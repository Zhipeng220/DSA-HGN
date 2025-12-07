import math
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


# =============================================================================
# Basic Modules (TCN, GCN)
# =============================================================================

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=False),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out = out + res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

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
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-5)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y = y + self.down(x)
        y = self.relu(y)

        return y


# =============================================================================
# Hypergraph Modules (Optimization: STE + Orthogonality + Parameter Passthrough)
# =============================================================================

class BinaryStep(torch.autograd.Function):
    """
    [Paper A Optimization] Straight-Through Estimator (STE)
    前向：阶跃函数实现硬剪枝 (Hard Pruning)。
    反向：梯度直通 (HardTanh)，允许端到端训练。
    """

    @staticmethod
    def forward(ctx, input):
        # input > 0 输出 1，否则输出 0
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 使用 HardTanh 模拟梯度，防止梯度消失，使得不可导的 step 函数可训练
        return F.hardtanh(grad_output)


class DifferentiableSparseHypergraph(nn.Module):
    """
    [Paper A Optimization] Differentiable Sparse Hypergraph Generator

    Features:
    1. Gated Differentiable Binarization (via STE): 实现真正的 0/1 剪枝。
    2. Adaptive Thresholding: 动态自适应阈值。
    3. Virtual Connection Toggle: 支持通过 use_virtual_conn 参数关闭模块（消融实验）。
    4. Visualization Support: 缓存 last_h 用于导出拓扑图。
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

        self.binary_step = BinaryStep.apply

        # Cache for loss calculation and visualization
        self.last_mask = None
        self.last_prototypes = None
        self.last_h = None  # 🚨 [关键修复] 用于保存关联矩阵，供 visualization 使用

    def forward(self, x):
        # [Control Logic] 如果禁用了虚拟连接，直接返回全零关联矩阵
        # 此时 Hypergraph 不会对节点特征进行混合，等同于移除了该模块
        if not self.use_virtual_conn:
            N, C, T, V = x.shape
            # 返回全0矩阵，表示无任何节点与超边相连
            # 注意: 需要与 x 在同一设备上
            return torch.zeros(N, V, self.num_hyperedges, device=x.device)

        N, C, T, V = x.shape

        # 1. Node Embedding
        q_node = self.query(x)
        q_node_pooled = q_node.mean(2).permute(0, 2, 1)  # (N, V, C')
        q_node_pooled = self.ln(q_node_pooled)

        # 2. Raw Affinity (Query * Prototype)
        scale = self.query.out_channels ** -0.5
        k = self.key_prototypes
        # H_raw: (N, V, M)
        H_raw = torch.matmul(q_node_pooled, k) * scale

        # 3. Adaptive Thresholding
        # thresh: (N, 1, 1, V) -> (N, V, 1)
        thresh = self.threshold_net(x).mean(2).permute(0, 2, 1)

        # 4. Gated Differentiable Binarization (STE)
        # 计算差值：Similarity - Threshold
        diff = H_raw - thresh
        mask = self.binary_step(diff)  # 0 or 1

        # 5. Apply Mask
        # Retain raw scores where mask is 1, else prune heavily (set to -inf)
        # Note: Softmax(-inf) -> 0. We use a large negative number (-1e9).
        H_masked = H_raw.masked_fill(mask == 0, -1e9)

        # 6. Softmax Normalization
        H_final = torch.softmax(H_masked, dim=-1)

        # Save state for Loss and Visualization
        self.last_mask = mask
        self.last_prototypes = k
        self.last_h = H_final  # 🚨 [关键修复] 保存 H_final

        return H_final

    def get_loss(self):
        """
        Returns (sparsity_loss, orthogonality_loss)
        """
        # 如果禁用了虚拟连接，或者还没跑过 forward，不计算相关 Loss
        if not self.use_virtual_conn or self.last_mask is None:
            return torch.tensor(0.0), torch.tensor(0.0)

        # 1. Sparsity Loss: Minimize the ratio of active connections
        # 优化 mask 中 1 的比例，使其尽可能稀疏
        loss_sparsity = torch.mean(self.last_mask)

        # 2. Orthogonality Loss: Minimize cosine similarity between prototypes
        # 强制超边原型关注不同的语义特征，避免 Mode Collapse
        k = self.last_prototypes
        k_norm = F.normalize(k, p=2, dim=0)
        gram = torch.matmul(k_norm.T, k_norm)  # (M, M)
        identity = torch.eye(gram.shape[0], device=gram.device)
        # Minimize off-diagonal elements
        loss_ortho = torch.norm(gram * (1 - identity), p='fro')

        return loss_sparsity, loss_ortho

class unit_hypergcn(nn.Module):
    # [Modified] 接收 **kwargs 并传给 dhg
    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True, **kwargs):
        super(unit_hypergcn, self).__init__()

        # 传递 kwargs (包含 use_virtual_conn)
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
        H = self.dhg(x)

        # H_norm_v 计算时要小心，如果 H 全为 0 (use_virtual_conn=False)
        # H.sum(dim=1) 会是 0，加 epsilon 防止除零
        H_norm_v = H / (H.sum(dim=1, keepdim=True) + 1e-5)

        x_v2e_feat = self.conv_v2e(x)
        x_edge = torch.einsum('nctv,nve->ncte', x_v2e_feat, H_norm_v)

        x_e_feat = self.conv_e(x_edge)

        H_norm_e = H / (H.sum(dim=2, keepdim=True) + 1e-5)
        x_node = torch.einsum('ncte,nev->nctv', x_e_feat, H_norm_e.transpose(1, 2))

        y = self.bn(x_node)
        y = y + self.down(x)
        y = self.relu(y)
        return y


class TCN_GCN_unit(nn.Module):
    # [Modified] 接收 **kwargs 并传给 hypergcn1
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2], num_hyperedges=16, **kwargs):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        self.hypergcn1 = unit_hypergcn(in_channels, out_channels, num_hyperedges=num_hyperedges, **kwargs)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=False)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        z_gcn = self.gcn1(x)
        z_hyp = self.hypergcn1(x)
        alpha = self.gate(x)

        z_fused = alpha * z_gcn + (1 - alpha) * z_hyp

        y = self.relu(self.tcn1(z_fused) + self.residual(x))
        return y


# =============================================================================
# Backbone Model
# =============================================================================

class Model(nn.Module):
    # [Modified] 接收 **kwargs 并传给每一层 TCN_GCN_unit
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_hyperedges=16,
                 base_channels=64, num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8],
                 pretrained=None, data_bn_type='VC', ch_ratio=2, **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = base_channels

        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive,
                               num_hyperedges=num_hyperedges, **kwargs)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               **kwargs)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               **kwargs)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               **kwargs)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive,
                               num_hyperedges=num_hyperedges, **kwargs)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               **kwargs)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               **kwargs)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive,
                               num_hyperedges=num_hyperedges, **kwargs)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                               **kwargs)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, num_hyperedges=num_hyperedges,
                                **kwargs)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, drop=False, return_features=False):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)

        z = self.l10(x)

        c_new = z.size(1)
        x_gap = z.view(N, M, c_new, -1)
        x_gap = x_gap.mean(3).mean(1)

        features_before_drop = x_gap
        x_out = self.drop_out(x_gap)

        if return_features:
            return x_out, z

        if drop:
            return features_before_drop, x_out
        else:
            return self.fc(x_out)

    def get_hypergraph_l1_loss(self):
        return torch.tensor(0.0, device=self.data_bn.weight.device)


class ChannelDifferentialBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels - 1, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_diff = x[:, 1:, :, :] - x[:, :-1, :, :]
        out = self.diff_conv(x_diff)
        return out


class DualBranchDSA_HGN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 **kwargs):
        super().__init__()

        self.st_branch = Model(num_class=num_class, num_point=num_point, num_person=num_person,
                               graph=graph, graph_args=graph_args, in_channels=in_channels, **kwargs)

        self.diff_prep = ChannelDifferentialBlock(in_channels)
        self.diff_branch = Model(num_class=num_class, num_point=num_point, num_person=num_person,
                                 graph=graph, graph_args=graph_args, in_channels=in_channels, **kwargs)

        base_channel = kwargs.get('base_channels', 64)
        feature_dim = base_channel * 4

        self.fusion_fc = nn.Linear(feature_dim * 2, num_class)

    def forward(self, x, drop=False, return_features=False):
        x_st = x

        N, C, T, V, M = x.shape
        x_reshaped = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x_diff = self.diff_prep(x_reshaped)
        x_diff = x_diff.view(N, M, C, T, V).permute(0, 2, 3, 4, 1).contiguous()

        feat_st, z_st = self.st_branch(x_st, return_features=True)
        feat_diff, z_diff = self.diff_branch(x_diff, return_features=True)

        feat_fused = torch.cat([feat_st, feat_diff], dim=1)

        if return_features:
            return feat_fused, z_st

        out = self.fusion_fc(feat_fused)
        return out

    def get_hypergraph_l1_loss(self):
        return torch.tensor(0.0)


class HypergraphAttentionFusion(nn.Module):
    def __init__(self, in_channels, num_streams=4):
        super().__init__()
        self.num_streams = num_streams

        self.attn_conv = nn.Sequential(
            nn.Linear(in_channels * num_streams, in_channels * num_streams // 2),
            nn.ReLU(),
            nn.Linear(in_channels * num_streams // 2, num_streams),
            nn.Softmax(dim=1)
        )

    def forward(self, features_list):
        features_stack = torch.stack(features_list, dim=1)
        features_cat = torch.cat(features_list, dim=1)

        attn_weights = self.attn_conv(features_cat)

        attn_weights = attn_weights.unsqueeze(-1)
        fused_feature = (features_stack * attn_weights).sum(dim=1)

        return fused_feature, attn_weights


class MultiStreamDSA_HGN(nn.Module):
    def __init__(self, model_args, num_class=14, streams=['joint', 'bone', 'joint_motion', 'bone_motion']):
        super().__init__()
        self.streams = streams
        self.num_streams = len(streams)

        self.backbones = nn.ModuleList([
            DualBranchDSA_HGN(num_class=num_class, **model_args)
            for _ in range(self.num_streams)
        ])

        base_channel = model_args.get('base_channels', 64)
        feature_dim = base_channel * 4 * 2

        self.hafm = HypergraphAttentionFusion(feature_dim, num_streams=self.num_streams)
        self.fc = nn.Linear(feature_dim, num_class)

        self.bone_pairs = []
        if 'graph' in model_args:
            Graph = import_class(model_args['graph'])
            graph_args = model_args.get('graph_args', {})
            graph = Graph(**graph_args)
            if hasattr(graph, 'inward'):
                self.bone_pairs = graph.inward
            else:
                print("Warning: Graph does not have 'inward' attribute. Bone stream will be zero.")

    def forward(self, x_joint):
        inputs = []
        inputs.append(x_joint)  # Stream 1: Joint

        x_bone = None

        if self.num_streams > 1:
            x_bone = torch.zeros_like(x_joint)
            if self.bone_pairs:
                for v1, v2 in self.bone_pairs:
                    x_bone[:, :, :, v1, :] = x_joint[:, :, :, v1, :] - x_joint[:, :, :, v2, :]
            inputs.append(x_bone)

        if self.num_streams > 2:
            x_jm = torch.zeros_like(x_joint)
            x_jm[:, :, :-1, :, :] = x_joint[:, :, 1:, :, :] - x_joint[:, :, :-1, :, :]
            inputs.append(x_jm)

        if self.num_streams > 3:
            if x_bone is None:
                x_bone = torch.zeros_like(x_joint)
                if self.bone_pairs:
                    for v1, v2 in self.bone_pairs:
                        x_bone[:, :, :, v1, :] = x_joint[:, :, :, v1, :] - x_joint[:, :, :, v2, :]

            x_bm = torch.zeros_like(x_bone)
            x_bm[:, :, :-1, :, :] = x_bone[:, :, 1:, :, :] - x_bone[:, :, :-1, :, :]
            inputs.append(x_bm)

        features = []
        for i, backbone in enumerate(self.backbones):
            if i < len(inputs):
                feat, _ = backbone(inputs[i], return_features=True)
                features.append(feat)

        fused_feat, attn = self.hafm(features)
        out = self.fc(fused_feat)

        return out