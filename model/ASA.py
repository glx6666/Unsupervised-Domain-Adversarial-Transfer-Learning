import torch
import torch.nn as nn
import torch.nn.functional as F

class ASA(nn.Module):
    def __init__(self, in_channels, reduction=8, sparsity_lambda=0.5):
        super(ASA, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.sparsity_lambda = sparsity_lambda

        # 通道注意力
        self.maxpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # soft thresholding 参数（可学习）
        self.threshold = nn.Parameter(torch.zeros(1, in_channels, 1))

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.size()
        #print(x.shape)

        # 1. 通道注意力
        y = self.maxpool(x)                 # -> [B, C]
        y = F.relu(self.fc1(y))                             # -> [B, C//r]
        y = torch.sigmoid(self.fc2(y))      # -> [B, C, 1]
        #print(y.shape)

        # 2. 权重乘以原特征（注意力增强）
        x_att = x * y                                        # [B, C, L]

        # 3. 稀疏化：自适应 soft thresholding
        x_sparse = torch.sign(x_att) * F.relu(torch.abs(x_att) - self.sparsity_lambda * self.threshold)

        return x_sparse

if __name__ == '__main__':
    asa = ASA(in_channels=64, reduction=8, sparsity_lambda=0.5)
    x = torch.randn(64, 64, 1024)  # 示例输入
    out = asa(x)
    print(out.shape)  # 应该输出: [16, 64, 128]
