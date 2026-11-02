"""固定社区评测版 ResNet 网络；只允许由独立推理进程导入。"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    """保持上游参数名及计算顺序的单维残差块，输入和输出均为批次张量。"""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """按输入／输出通道数和步长创建卷积及归一化层，参数名与上游保持一致。"""
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(3,),
                               stride=(stride,), padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=(3,),
                               stride=(1,), padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=(1,), stride=(stride,), bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对编码特征执行两次卷积与残差连接，不修改输入张量。"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResnetModel(nn.Module):
    """三角色各自加载独立权重；z 为 N×40×54，x 为 N×15，评分为 N×1。"""
    def __init__(self) -> None:
        """构造固定40通道网络；三种身份使用相同结构，独立加载各自权重。"""
        super().__init__()
        self.in_planes = 80
        #input 1*54*41
        self.conv1 = nn.Conv1d(40, 80, kernel_size=(3,),
                               stride=(2,), padding=1, bias=False) #1*27*80

        self.bn1 = nn.BatchNorm1d(80)

        self.layer1 = self._make_layer(BasicBlock, 80, 2, stride=2)#1*14*80
        self.layer2 = self._make_layer(BasicBlock, 160, 2, stride=2)#1*7*160
        self.layer3 = self._make_layer(BasicBlock, 320, 2, stride=2)#1*4*320
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(320 * BasicBlock.expansion * 4 + 15 * 4, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """按固定数量构造残差阶段，仅首块使用指定步长，不修改上游层顺序。"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, x: torch.Tensor, return_value: bool = False, flags=None, debug=False) -> dict:
        """复现上游原始评分；推理使用 return_value=True，不把评分解释为胜率。"""
        out = F.relu(self.bn1(self.conv1(z)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1,2)
        out = torch.cat([x,x,x,x,out], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = F.leaky_relu_(self.linear4(out))
        if return_value:
            return dict(values=out)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out,dim=0)[0]
            return dict(action=action, max_value=torch.max(out))
