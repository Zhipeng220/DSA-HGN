import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FT_Processor(Processor):
    """
        Processor for Finetune Evaluation (Paper A Supervised Training).
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)

        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        # [FIX] Paper A: 推荐使用 AdamW 配合超图训练
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                eps=1e-4)  # 防止除零
        else:
            raise ValueError()

    def adjust_lr(self):
        # [FIX] 只要设置了 step 就执行衰减，不再局限于 SGD
        if self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        l1_loss_value = []  # [Paper A] 记录 L1 Loss

        for data, label in loader:
            self.global_step += 1
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # [保留原始逻辑] 处理 bone 数据流
            # 如果 feeder 没有处理 bone，这里会手动计算
            if self.arg.stream == 'bone':
                if self.arg.train_feeder_args.get('bone', False):
                    pass
                else:
                    from net.utils.graph import Graph
                    graph = Graph(self.arg.model_args['graph_args']['layout'])
                    bone = torch.zeros_like(data)
                    for v1, v2 in graph.Bones:
                        bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                    data = bone

            # forward
            output = self.model(data)

            # [Paper A] 核心修改：计算分类损失 + L1 稀疏损失
            loss_ce = self.loss(output, label)

            # 获取 L1 Loss (如果模型支持)
            loss_l1 = torch.tensor(0.0).to(self.dev)
            if self.arg.lambda_l1 > 0:
                if hasattr(self.model, 'get_hypergraph_l1_loss'):
                    loss_l1 = self.model.get_hypergraph_l1_loss()
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'get_hypergraph_l1_loss'):
                    loss_l1 = self.model.module.get_hypergraph_l1_loss()

            # 总损失
            loss = loss_ce + (self.arg.lambda_l1 * loss_l1)

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # [FIX] 梯度裁剪 (Paper A 必备，防止超图梯度爆炸)
            if self.arg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip_norm)

            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_ce'] = loss_ce.data.item()
            self.iter_info['loss_l1'] = loss_l1.data.item()  # 记录 L1
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

            loss_value.append(self.iter_info['loss'])
            l1_loss_value.append(self.iter_info['loss_l1'])

            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_l1'] = np.mean(l1_loss_value)  # 记录 Epoch 级 L1

        self.show_epoch_info()
        self.train_writer.add_scalar('loss', self.epoch_info['mean_loss'], epoch)
        self.train_writer.add_scalar('loss_l1', self.epoch_info['mean_l1'], epoch)

    def test(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # [保留原始逻辑] 处理 bone 数据流
            if self.arg.stream == 'bone':
                if self.arg.test_feeder_args.get('bone', False):
                    pass
                else:
                    from net.utils.graph import Graph
                    graph = Graph(self.arg.model_args['graph_args']['layout'])
                    bone = torch.zeros_like(data)
                    for v1, v2 in graph.Bones:
                        bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                    data = bone

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if 'train' or 'test' in self.arg.phase:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if 'train' or 'test' in self.arg.phase:
            self.label = np.concatenate(label_frag)

        self.eval_info['mean_loss'] = np.mean(loss_value)
        self.show_eval_info()

        # show top-k accuracy
        for k in self.arg.show_topk:
            self.show_topk(k)
        self.show_best(1)  # 默认显示 Top-1 最佳

        # [FIX] 只有在评估间隔时才写入 TensorBoard
        # if epoch % self.arg.eval_interval == 0:
        self.eval_log_writer(epoch)

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')

        # [Paper A] 新增参数
        parser.add_argument('--lambda_l1', type=float, default=0.0001, help='Weight for Hypergraph L1 Sparsity loss')
        parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm')

        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')

        return parser