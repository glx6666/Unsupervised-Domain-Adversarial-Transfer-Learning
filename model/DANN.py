import torch.nn as nn
from model.GRL import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """ResNet 18 的基本残差块 (1D)"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 残差连接
        out = F.relu(out)

        return out


class DANN_F(nn.Module):
    """ResNet18 for 1D Signal"""

    def __init__(self):
        super(DANN_F, self).__init__()
        self.in_channels = 64  # 初始通道数
        self.BN = nn.BatchNorm1d(1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv = nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # ResNet18 的 4 个层，每个有多个残差块
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应池化成 (batch, 512, 1)
        self.fc = nn.Linear(512, 256)  # 全连接层

    def _make_layer(self, out_channels, blocks, stride):
        """创建多个残差块"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels  # 更新通道数

        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.MultiHeadSelfAttention1D(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # (batch, 512, 1)
        x = torch.flatten(x, 1)  # (batch, 512)
        x = self.fc(x)  # (batch, num_classes)
        return x


class DANN_D(nn.Module):
    def __init__(self, alpha=1.0):
        super(DANN_D, self).__init__()
        self.grl = GradientReversalLayer(alpha)  # 加入梯度反转层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.grl(x)  # 应用梯度反转
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # 输出在 (0,1) 之间
        return x


class DANN_C(nn.Module):
    def __init__(self, num_class):
        super(DANN_C, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # 直接返回 logits，Softmax 在 loss 计算时自动处理


if __name__ == '__main__':
    if __name__ == '__main__':
        # 模拟一个 batch_size=64，1 通道，长度为 1024 的输入信号
        x = torch.randn(64, 1, 1024).cuda()

        # 初始化模型（输入为单通道）
        model = DANN_F().cuda()

        # 前向传播
        y = model(x)

        print("输出形状：", y.shape)  # 应该是 [64, 256]

