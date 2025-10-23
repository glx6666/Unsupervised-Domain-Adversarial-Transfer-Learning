import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationExtractor(nn.Module):
    def __init__(self, num_class=10):
        super(ClassificationExtractor, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # 直接返回 logits，Softmax 在 loss 计算时自动处理
