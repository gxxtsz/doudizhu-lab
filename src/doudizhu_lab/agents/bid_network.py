"""固定社区叫抢网络的推理结构；仅由独立模型进程导入 PyTorch。

结构移植自 Vincentzyx/DouZero_For_HLDDZ_FullAuto 的 BidModel.Net2，
提交 6d20491da216c989d23cf2883498758fb78e9f8d，按 Apache-2.0 保留来源。
原始代码与许可证见 vendor/bidding；这里不包含训练、随机参数回退或设备接口。
"""

import torch
from torch import nn
from torch.nn import functional as F


class BidNetwork(nn.Module):
    """严格复现 Net2 的参数名、尺寸及前向计算；输入为按行展开的 4×15 计数矩阵。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.dense1 = nn.Linear(1020, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense5 = nn.Linear(128, 1)

    def forward(self, features):
        """输出上游原始叫牌评分；没有 Sigmoid，不能将其当作实际胜率。"""
        value = F.leaky_relu(self.conv1(features.unsqueeze(1))).flatten(1, 2)
        value = torch.cat((value, features), 1)
        for layer in (self.dense1, self.dense2, self.dense3, self.dense4):
            value = F.leaky_relu(layer(value))
        return self.dense5(value)
