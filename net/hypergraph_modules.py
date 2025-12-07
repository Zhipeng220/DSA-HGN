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
        self.use_virtual_conn = use_virtual_conn

        inter_channels = max(1, in_channels // ratio)

        # 1. Feature Projection
        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.key_prototypes = nn.Parameter(torch.randn(inter_channels, num_hyperedges))

        # 2. Stability
        self.ln = nn.LayerNorm(inter_channels)

        # 3. Dynamic Threshold Generator
        self.threshold_net = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()  # Range (0, 1)
        )

        # [核心修复 1] 初始化偏置为 -5.0
        # Sigmoid(-5.0) ≈ 0.006，初始阈值极低，确保初期 H_raw > thresh，
        # 让 Mask=1，从而保证梯度能传导到 key_prototypes。
        nn.init.constant_(self.threshold_net[0].bias, -5.0)

        # Cache for loss calculation
        self.last_mask = None
        self.last_prototypes = None
        self.last_h = None

    def forward(self, x):
        # 如果禁用虚拟连接，返回全零矩阵
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
        # thresh shape: (N, V, 1) - 每个节点都有自己的阈值
        thresh = self.threshold_net(x).mean(2).permute(0, 2, 1)

        # 限制阈值范围，防止数值极化
        thresh = torch.clamp(thresh, min=0.0, max=1.0)

        # 4. Soft-Hard STE (Straight-Through Estimator)
        diff = H_raw - thresh

        # Sigmoid 温度系数=5.0，控制反向传播时的梯度斜率
        mask_soft = torch.sigmoid(diff * 5.0)
        mask_hard = (diff > 0).float()

        # 前向传播用 mask_hard (0/1)，反向传播用 mask_soft (梯度)
        mask = mask_hard - mask_soft.detach() + mask_soft

        # 5. Differentiable Masking
        # 使用加法/乘法掩码代替 masked_fill，保持计算图连通性
        # mask=1 -> H_raw; mask=0 -> -1e4 (Softmax后趋近0)
        H_masked = H_raw * mask + (1 - mask) * -1e4

        # 6. Softmax Normalization
        H_final = torch.softmax(H_masked, dim=-1)

        # Save state for Loss
        self.last_mask = mask
        self.last_prototypes = k
        self.last_h = H_final

        return H_final

    def get_loss(self, target_density=0.5):
        """
        Returns (sparsity_loss, orthogonality_loss)
        Args:
            target_density (float): 期望保留的连接比例
        """
        if not self.use_virtual_conn or self.last_mask is None:
            dev = self.key_prototypes.device
            return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)

        # --- 1. Sparsity Loss ---
        # 计算当前连接密度
        current_density = torch.mean(self.last_mask)
        # 仅当密度高于目标时才惩罚 (ReLU)，避免过度稀疏化
        loss_sparsity = F.relu(current_density - target_density)

        # --- 2. Orthogonality Loss ---
        # 鼓励超边原型正交，捕捉多样化特征
        k = self.last_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)

        identity = torch.eye(gram.shape[0], device=gram.device)
        off_diagonal = gram * (1 - identity)
        loss_ortho = torch.mean(off_diagonal ** 2)

        return loss_sparsity, loss_ortho


class unit_hypergcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True, **kwargs):
        super(unit_hypergcn, self).__init__()

        # 将 kwargs (如 use_virtual_conn) 传递给核心模块
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

        # 初始化 BN，防止梯度消失
        bn_init(self.bn, 1e-5)

    def forward(self, x):
        N, C, T, V = x.shape

        # 1. 生成动态关联矩阵 H (N, V, M)
        H = self.dhg(x)

        # 2. Hypergraph Convolution: Node -> Edge -> Node

        # [Step A] Node to Hyperedge (V -> E)
        # 归一化 H (按节点度数)
        # 添加 epsilon 防止除零 (当 Mask 全为 0 时)
        H_norm_v = H / (H.sum(dim=1, keepdim=True) + 1e-5)

        x_v2e_feat = self.conv_v2e(x)
        # Einsum: (N, C, T, V) * (N, V, M) -> (N, C, T, M)
        x_edge = torch.einsum('nctv,nve->ncte', x_v2e_feat, H_norm_v)

        # [Step B] Edge Feature Transformation
        x_e_feat = self.conv_e(x_edge)

        # [Step C] Hyperedge to Node (E -> V)
        # 归一化 H (按超边度数)
        H_norm_e = H / (H.sum(dim=2, keepdim=True) + 1e-5)
        # Einsum: (N, C, T, M) * (N, M, V) -> (N, C, T, V)
        x_node = torch.einsum('ncte,nev->nctv', x_e_feat, H_norm_e.transpose(1, 2))

        # 3. Residual & Activation
        y = self.bn(x_node)
        y = y + self.down(x)
        y = self.relu(y)
        return y