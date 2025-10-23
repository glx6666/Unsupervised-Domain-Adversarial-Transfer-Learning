import torch
import torch.nn as nn
import torch.autograd as autograd

class GradientReversalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        """前向传播: 直接返回输入"""
        ctx.alpha = alpha  # 存储反转系数
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播: 反转梯度的符号并乘以 alpha"""
        return -ctx.alpha * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha  # 反转系数

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
