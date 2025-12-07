import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import bn_init


class DifferentiableSparseHypergraph(nn.Module):
    """
    [Paper A Optimization] Differentiable Sparse Hypergraph Generator

    🔥 CRITICAL FIX: Scale-Aware Thresholding + Dimension Fix
    """

    def __init__(self, in_channels, num_hyperedges, ratio=8, use_virtual_conn=True, **kwargs):
        super(DifferentiableSparseHypergraph, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.in_channels = in_channels
        self.use_virtual_conn = use_virtual_conn

        inter_channels = max(1, in_channels // ratio)
        self.inter_channels = inter_channels

        # 1. Feature Projection
        self.query = nn.Conv2d(in_channels, inter_channels, 1)

        # 🔥 FIX 1: Orthogonal Initialization
        prototypes = torch.randn(inter_channels, num_hyperedges)
        if inter_channels >= num_hyperedges:
            q, _ = torch.linalg.qr(prototypes)
            self.key_prototypes = nn.Parameter(q.contiguous())
        else:
            q, _ = torch.linalg.qr(prototypes.T)
            self.key_prototypes = nn.Parameter(q.T.contiguous())

        print(f"[INIT] Prototypes initialized with orthogonal constraint. "
              f"Shape: {self.key_prototypes.shape}")

        # 🔥 FIX 2: 简化归一化 - 不使用 LayerNorm/BatchNorm
        # 直接使用 L2 归一化，避免维度问题

        # 🔥 FIX 3: Scale-Aware Threshold Generator (简化版)
        self.threshold_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (N, C, T, V) -> (N, C, 1, 1)
            nn.Flatten(),  # (N, C)
            nn.Linear(in_channels, 1),  # (N, 1)
            nn.Tanh()  # 输出 [-1, 1]
        )

        # 初始化为负值
        nn.init.constant_(self.threshold_net[2].bias, -1.0)

        print(f"[INIT] Scale-aware threshold network initialized")

        # Cache
        self.last_mask = None
        self.last_prototypes = None
        self.last_h = None
        self.debug_counter = 0

    def forward(self, x):
        if not self.use_virtual_conn:
            N, C, T, V = x.shape
            return torch.zeros(N, V, self.num_hyperedges, device=x.device)

        N, C, T, V = x.shape

        # 1. Node Embedding
        q_node = self.query(x)  # (N, C', T, V)
        q_node_pooled = q_node.mean(2)  # (N, C', V)

        # 🔥 FIX 4: L2 Normalization (保持维度清晰)
        q_node_pooled = F.normalize(q_node_pooled, p=2, dim=1)  # (N, C', V)
        q_node_pooled = q_node_pooled.permute(0, 2, 1)  # (N, V, C')

        # 2. Raw Affinity
        k = self.key_prototypes  # (C', M)
        scale = self.inter_channels ** -0.5
        H_raw = torch.matmul(q_node_pooled, k) * scale  # (N, V, M)

        # 🔥 FIX 5: 生成阈值 (确保维度正确)
        thresh = self.threshold_net(x)  # (N, 1)
        thresh = thresh.view(N, 1, 1)  # (N, 1, 1) - 显式 reshape

        # 限制阈值范围
        thresh = torch.clamp(thresh, min=-0.5, max=0.5)

        # 4. Soft-Hard STE
        diff = H_raw - thresh  # (N, V, M) - (N, 1, 1) -> (N, V, M) 广播正确

        mask_soft = torch.sigmoid(diff * 5.0)
        mask_hard = (diff > 0).float()
        mask = mask_hard - mask_soft.detach() + mask_soft

        # 5. Differentiable Masking
        H_masked = H_raw * mask + (1 - mask) * -1e4

        # 6. Softmax Normalization
        H_final = torch.softmax(H_masked, dim=-1)  # (N, V, M)

        # 🔥 确保输出维度正确
        assert H_final.shape == (N, V, self.num_hyperedges), \
            f"H_final shape mismatch: {H_final.shape} vs expected ({N}, {V}, {self.num_hyperedges})"

        # Save state
        self.last_mask = mask
        self.last_prototypes = k
        self.last_h = H_final

        # Debug
        if self.training:
            self.debug_counter += 1
            if self.debug_counter % 500 == 0:
                print(f"[HYPERGRAPH DEBUG @iter {self.debug_counter}]")
                print(f"  - H_raw range: [{H_raw.min().item():.4f}, {H_raw.max().item():.4f}]")
                print(f"  - Threshold value: {thresh.mean().item():.4f}")
                print(f"  - Diff range: [{diff.min().item():.4f}, {diff.max().item():.4f}]")
                print(f"  - Mask mean: {mask.mean().item():.4f} (target: 0.3~0.7)")
                print(f"  - H_final entropy: {self._compute_entropy(H_final):.4f}")
                print(f"  - Prototype norm: {k.norm(dim=0).mean().item():.4f}")

        return H_final  # 确保返回 (N, V, M)

    def _compute_entropy(self, H):
        """计算超图分布的熵"""
        H_normalized = H / (H.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(H_normalized * torch.log(H_normalized + 1e-8)).sum(dim=-1)
        return entropy.mean()

    def get_loss(self, target_density=0.5):
        if not self.use_virtual_conn or self.last_mask is None:
            dev = self.key_prototypes.device
            return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)

        # 1. Sparsity Loss
        current_density = torch.mean(self.last_mask)
        loss_sparsity = F.relu(current_density - target_density)

        # 2. Orthogonality Loss
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

        assert H.dim() == 3, f"H should be 3D, got {H.shape}"

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
