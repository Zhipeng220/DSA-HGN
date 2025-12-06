import math
import pdb
import numpy as np
import torch
import torch.nn as nn
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
# Basic Modules (TCN, GCN, Hypergraph)
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
        # 提高 epsilon 以防 MPS 不稳定
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


class DynamicHypergraphGenerator(nn.Module):
    def __init__(self, in_channels, num_hyperedges, ratio=8):
        super(DynamicHypergraphGenerator, self).__init__()
        self.num_hyperedges = num_hyperedges
        self.in_channels = in_channels

        inter_channels = max(1, in_channels // ratio)

        self.query = nn.Conv2d(in_channels, inter_channels, 1)

        # [OPTIMIZATION] LayerNorm 用于稳定 Attention 的输入分布
        # 这是解决数值不稳定性(NaN)最根本的方法
        self.ln = nn.LayerNorm(inter_channels)

        self.hyperedge_prototypes = nn.Parameter(torch.randn(inter_channels, num_hyperedges))

    def forward(self, x):
        N, C, T, V = x.shape
        q_node = self.query(x)
        q_node_pooled = q_node.mean(2).permute(0, 2, 1)  # (N, V, C')

        # [OPTIMIZATION] 归一化输入特征
        q_node_pooled = self.ln(q_node_pooled)

        # Scaled Dot-Product Attention
        scale = self.query.out_channels ** -0.5
        H = torch.matmul(q_node_pooled, self.hyperedge_prototypes) * scale

        # [OPTIMIZATION] 由于有 LayerNorm，H 的数值范围被限制在合理区间
        # 不再需要 nan_to_num 和 clamp 等硬补丁

        H = torch.softmax(H, dim=-1)
        return H


class unit_hypergcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=16, residual=True):
        super(unit_hypergcn, self).__init__()

        self.dhg = DynamicHypergraphGenerator(in_channels, num_hyperedges)

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

        # [FIX] 提高 BN epsilon
        bn_init(self.bn, 1e-5)

    def forward(self, x):
        N, C, T, V = x.shape
        H = self.dhg(x)

        # 使用较小的 epsilon 即可，因为 H 来自稳定的 Softmax
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
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2], num_hyperedges=16):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        self.hypergcn1 = unit_hypergcn(in_channels, out_channels, num_hyperedges=num_hyperedges)

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
# Backbone Model (Single Stream, Single Branch)
# =============================================================================

class Model(nn.Module):
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

        # Hardcoded structure (10 stages, inflate at 5,8)
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive,
                               num_hyperedges=num_hyperedges)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, num_hyperedges=num_hyperedges)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive,
                               num_hyperedges=num_hyperedges)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, num_hyperedges=num_hyperedges)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, num_hyperedges=num_hyperedges)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive,
                               num_hyperedges=num_hyperedges)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, num_hyperedges=num_hyperedges)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, num_hyperedges=num_hyperedges)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, drop=False, return_features=False):
        # [OPTIMIZATION] 移除了入口处的 nan_to_num 补丁
        # 如果输入数据有问题，应由数据预处理或 Feeder 解决

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

        z = self.l10(x)  # (N*M, C_feat, T_feat, V)

        c_new = z.size(1)
        x_gap = z.view(N, M, c_new, -1)
        x_gap = x_gap.mean(3).mean(1)  # (N, C_feat)

        features_before_drop = x_gap
        x_out = self.drop_out(x_gap)  # (N, C_feat)

        if return_features:
            return x_out, z

        if drop:
            return features_before_drop, x_out
        else:
            return self.fc(x_out)

    def get_hypergraph_l1_loss(self):
        l1_loss = 0.0
        count = 0
        for m in self.modules():
            if isinstance(m, DynamicHypergraphGenerator):
                l1_loss += torch.sum(torch.abs(m.hyperedge_prototypes))
                count += 1

        if count > 0:
            return l1_loss / count
        else:
            return torch.tensor(0.0, device=self.data_bn.weight.device)


# =============================================================================
# Innovation 1: Dual-Branch Differential Channel Design
# =============================================================================

class ChannelDifferentialBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels - 1, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (N, C, T, V)
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
        # x: (N, C, T, V, M)
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
        loss_st = self.st_branch.get_hypergraph_l1_loss()
        loss_diff = self.diff_branch.get_hypergraph_l1_loss()
        return (loss_st + loss_diff) / 2


# =============================================================================
# Innovation 2: Hypergraph Attention Fusion Module (HAFM)
# =============================================================================

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
        features_stack = torch.stack(features_list, dim=1)  # (N, num_streams, C)
        features_cat = torch.cat(features_list, dim=1)  # (N, num_streams * C)

        attn_weights = self.attn_conv(features_cat)  # (N, num_streams)

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

        # 2. Bone Stream
        if self.num_streams > 1:
            x_bone = torch.zeros_like(x_joint)
            if self.bone_pairs:
                for v1, v2 in self.bone_pairs:
                    x_bone[:, :, :, v1, :] = x_joint[:, :, :, v1, :] - x_joint[:, :, :, v2, :]
            inputs.append(x_bone)

        # 3. Joint Motion Stream
        if self.num_streams > 2:
            x_jm = torch.zeros_like(x_joint)
            x_jm[:, :, :-1, :, :] = x_joint[:, :, 1:, :, :] - x_joint[:, :, :-1, :, :]
            inputs.append(x_jm)

        # 4. Bone Motion Stream
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
