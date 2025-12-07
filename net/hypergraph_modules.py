import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import bn_init


class DifferentiableSparseHypergraph(nn.Module):
    """
    [Optimization] Top-K + Softmax Attention Hypergraph Generator
    Replaces unstable STE/Thresholding with robust Top-K selection.
    This fixes gradient vanishing and loss oscillation issues.
    """

    def __init__(self, in_channels, num_hyperedges, ratio=8, use_virtual_conn=True, k_neighbors=10, **kwargs):
        super(DifferentiableSparseHypergraph, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn

        # [NEW] Top-K 参数: 每个节点连接的超边数量
        # 默认值为 10，或者取超边总数的一半，取较小值以保证稀疏性
        self.k_neighbors = min(k_neighbors, num_hyperedges)

        inter_channels = max(1, in_channels // ratio)
        self.inter_channels = inter_channels

        # 1. Feature Projection
        self.query = nn.Conv2d(in_channels, inter_channels, 1)

        # 2. Prototypes (保持正交初始化)
        prototypes = torch.randn(inter_channels, num_hyperedges)
        if inter_channels >= num_hyperedges:
            q, _ = torch.linalg.qr(prototypes)
            self.key_prototypes = nn.Parameter(q.contiguous())
        else:
            q, _ = torch.linalg.qr(prototypes.T)
            self.key_prototypes = nn.Parameter(q.T.contiguous())

        print(f"[INIT] Top-K Sparse Hypergraph. K={self.k_neighbors}, M={num_hyperedges}")

        # [REMOVED] 移除了 threshold_net，不再需要学习阈值

        # Cache
        self.last_h = None

    def forward(self, x):
        if not self.use_virtual_conn:
            N, C, T, V = x.shape
            return torch.zeros(N, V, self.num_hyperedges, device=x.device)

        N, C, T, V = x.shape

        # 1. Node Embedding
        q_node = self.query(x)  # (N, C', T, V)
        q_node_pooled = q_node.mean(2)  # (N, C', V)

        # L2 Normalization (重要：防止点积数值过大)
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=1)  # (N, C', V)
        q_node_pooled = q_node_pooled.permute(0, 2, 1)  # (N, V, C')

        # 2. Raw Affinity
        k = self.key_prototypes  # (C', M)
        scale = self.inter_channels ** -0.5
        H_raw = torch.matmul(q_node_pooled, k) * scale  # (N, V, M)

        # =======================================================
        # [CORE CHANGE] Top-K Selection + Softmax Attention
        # =======================================================

        # 3. Top-K Selection (解决稀疏性问题，防止图全连或全断)
        # 为每个节点 V 选择亲和力最高的 k 个超边 M
        # val: (N, V, k), idx: (N, V, k)
        topk_val, topk_idx = torch.topk(H_raw, k=self.k_neighbors, dim=-1)

        # 4. Softmax Attention (解决梯度消失问题)
        # 仅对选中的 k 个连接进行 Softmax，保证梯度流畅回传
        topk_val = torch.softmax(topk_val, dim=-1)

        # 5. Reconstruct Sparse Matrix (N, V, M)
        # 创建全零矩阵，将计算出的 Attention 值填回对应位置
        H_final = torch.zeros_like(H_raw)
        H_final.scatter_(-1, topk_idx, topk_val)

        # =======================================================

        # Save state
        self.last_h = H_final

        return H_final  # (N, V, M)

    def get_loss(self, target_density=None):
        """
        不再需要 Sparsity Loss，因为 Top-K 机制已经硬性保证了稀疏度。
        保留 Orthogonality Loss 以促进原型的多样性。
        """
        if not self.use_virtual_conn:
            dev = self.key_prototypes.device
            return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)

        # 1. Sparsity Loss -> 0.0
        loss_sparsity = torch.tensor(0.0, device=self.key_prototypes.device)

        # 2. Orthogonality Loss
        k = self.key_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)

        identity = torch.eye(gram.shape[0], device=gram.device)
        off_diagonal = gram * (1 - identity)
        loss_ortho = torch.mean(off_diagonal ** 2)

        return loss_sparsity, loss_ortho


class unit_hypergcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True, **kwargs):
        super(unit_hypergcn, self).__init__()

        # [MODIFIED] 将 kwargs 传递给 dhg，以便传入 k_neighbors 等参数
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

        # 1. 生成动态关联矩阵 H (N, V, M)
        H = self.dhg(x)

        # 2. Hypergraph Convolution
        # 增加 epsilon 防止除零 (MPS/Float16 兼容性)
        H_norm_v = H / (H.sum(dim=1, keepdim=True) + 1e-5)  # (N, V, M)

        x_v2e_feat = self.conv_v2e(x)  # (N, C, T, V)
        x_edge = torch.einsum('nctv,nve->ncte', x_v2e_feat, H_norm_v)  # (N, C, T, E)

        x_e_feat = self.conv_e(x_edge)  # (N, C_out, T, E)

        H_norm_e = H / (H.sum(dim=2, keepdim=True) + 1e-5)  # (N, V, M)
        x_node = torch.einsum('ncte,nev->nctv', x_e_feat, H_norm_e.transpose(1, 2))  # (N, C_out, T, V)

        # 3. Residual & Activation
        y = self.bn(x_node)
        y = y + self.down(x)
        y = self.relu(y)
        return y