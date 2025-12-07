import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import bn_init


class DifferentiableSparseHypergraph(nn.Module):
    """
    [Paper A Optimization] Differentiable Sparse Hypergraph Generator

    Integrated Fixes:
    1. Dense Start Initialization (Prevents initial deadlock).
    2. Soft-Hard STE (Guarantees gradient flow for binary decisions).
    3. Differentiable Masking (Replaces masked_fill).
    4. Robust Orthogonality Loss (MSE based).
    """

    def __init__(self, in_channels, num_hyperedges, ratio=8, use_virtual_conn=True, **kwargs):
        super(DifferentiableSparseHypergraph, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn  # [Param] 控制虚拟连接是否启用

        inter_channels = max(1, in_channels // ratio)

        # Feature Projection
        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.key_prototypes = nn.Parameter(torch.randn(inter_channels, num_hyperedges))

        # Stability
        self.ln = nn.LayerNorm(inter_channels)

        # Dynamic Threshold Generator
        self.threshold_net = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()  # Range (0, 1)
        )

        # [核心修复 1] 初始化偏置为负值 (如 -2.0)
        # 初始阈值较低 (sigmoid(-2) ≈ 0.12)，确保大部分 diff > 0，连接处于开启状态。
        # 避免训练初期因全 0 掩码导致模型无法学习。
        nn.init.constant_(self.threshold_net[0].bias, -2.0)

        # Cache for loss calculation and visualization
        self.last_mask = None
        self.last_prototypes = None
        self.last_h = None

    def forward(self, x):
        # [Control Logic] 如果禁用了虚拟连接，直接返回全零关联矩阵
        if not self.use_virtual_conn:
            N, C, T, V = x.shape
            return torch.zeros(N, V, self.num_hyperedges, device=x.device)

        N, C, T, V = x.shape

        # 1. Node Embedding
        q_node = self.query(x)
        q_node_pooled = q_node.mean(2).permute(0, 2, 1)  # (N, V, C')
        q_node_pooled = self.ln(q_node_pooled)

        # 2. Raw Affinity (Query * Prototype)
        scale = self.query.out_channels ** -0.5
        k = self.key_prototypes
        H_raw = torch.matmul(q_node_pooled, k) * scale

        # 3. Adaptive Thresholding
        thresh = self.threshold_net(x).mean(2).permute(0, 2, 1)

        # 限制范围防止梯度消失，但给予足够的动态空间
        thresh = torch.clamp(thresh, min=-2.0, max=2.0)

        # 4. Soft-Hard STE (核心修复 2)
        diff = H_raw - thresh

        # 使用 Sigmoid 控制梯度流的斜率 (Temperature=5.0)
        # mask_soft 在反向传播时提供非零梯度
        mask_soft = torch.sigmoid(diff * 5.0)
        mask_hard = (diff > 0).float()

        # 技巧：前向传播值等于 mask_hard，反向传播梯度等于 mask_soft
        mask = mask_hard - mask_soft.detach() + mask_soft

        # 5. Differentiable Masking (核心修复 3)
        # 弃用 masked_fill，改用加法/乘法掩码，保证 H_raw 的梯度回传
        # 当 mask=1 时，保持 H_raw；当 mask=0 时，给予极小的负值 (-1e4)
        # 这样 Softmax 后 mask=0 的位置概率为 0，但 H_raw 的梯度不会被截断
        H_masked = H_raw * mask + (1 - mask) * -1e4

        # 6. Softmax Normalization
        H_final = torch.softmax(H_masked, dim=-1)

        # Save state for Loss and Visualization
        self.last_mask = mask
        self.last_prototypes = k
        self.last_h = H_final

        return H_final

    def get_loss(self):
        """
        Returns (sparsity_loss, orthogonality_loss)
        """
        if not self.use_virtual_conn or self.last_mask is None:
            return torch.tensor(0.0, device=self.key_prototypes.device), \
                torch.tensor(0.0, device=self.key_prototypes.device)

        # 1. Sparsity Loss: Minimize the ratio of active connections
        loss_sparsity = torch.mean(self.last_mask)

        # 2. Orthogonality Loss (核心修复 4: MSE 版本)
        k = self.last_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)  # (M, M)

        identity = torch.eye(gram.shape[0], device=gram.device)
        off_diagonal = gram * (1 - identity)

        # 使用 MSE (均方误差) 替代 Fro 范数，数值更稳定且小于 1，防止梯度爆炸
        loss_ortho = torch.mean(off_diagonal ** 2)

        return loss_sparsity, loss_ortho


class unit_hypergcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True, **kwargs):
        super(unit_hypergcn, self).__init__()

        # 传递 kwargs (包含 use_virtual_conn) 给 DifferentiableSparseHypergraph
        self.dhg = DifferentiableSparseHypergraph(in_channels, num_hyperedges, **kwargs)

        self.conv_v2e = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_e = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        bn_init(self.bn, 1e-5)

    def forward(self, x):
        N, C, T, V = x.shape

        # 1. 生成关联矩阵 H (N, V, M)
        H = self.dhg(x)

        # 2. Hypergraph Convolution: Node -> Edge -> Node

        # [Step A] Node to Hyperedge (V -> E)
        # H_norm_v: 归一化 (Degree of Hyperedge)
        # 加上 epsilon 防止除零 (当 use_virtual_conn=False 时 H 全为 0)
        H_norm_v = H / (H.sum(dim=1, keepdim=True) + 1e-5)

        x_v2e_feat = self.conv_v2e(x)
        # Einsum: (N, C, T, V) * (N, V, M) -> (N, C, T, M)
        x_edge = torch.einsum('nctv,nve->ncte', x_v2e_feat, H_norm_v)

        # [Step B] Edge Feature Transformation
        x_e_feat = self.conv_e(x_edge)

        # [Step C] Hyperedge to Node (E -> V)
        # H_norm_e: 归一化 (Degree of Vertex)
        H_norm_e = H / (H.sum(dim=2, keepdim=True) + 1e-5)
        # Einsum: (N, C, T, M) * (N, M, V) -> (N, C, T, V)
        x_node = torch.einsum('ncte,nev->nctv', x_e_feat, H_norm_e.transpose(1, 2))

        # 3. Residual & Activation
        y = self.bn(x_node)
        y = y + self.down(x)
        y = self.relu(y)
        return y