import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GRL import *

class DomainExtractor(nn.Module):
    def __init__(self, alpha=1.0):
        super(DomainExtractor, self).__init__()
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

