import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
from sklearn.metrics import accuracy_score
import time

class ResNetMultiLayer(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=False, out_layers=('layer2','layer3','layer4')):
        super().__init__()
        assert backbone in ['resnet18','resnet34','resnet50']
        if backbone == 'resnet18':
            net = models.resnet18(pretrained=pretrained)
            feat_dim_map = {'layer2':128, 'layer3':256, 'layer4':512}
        elif backbone == 'resnet34':
            net = models.resnet34(pretrained=pretrained)
            feat_dim_map = {'layer2':128, 'layer3':256, 'layer4':512}
        else:
            net = models.resnet50(pretrained=pretrained)
            feat_dim_map = {'layer2':512, 'layer3':1024, 'layer4':2048}

        # 移除最后的 avgpool 和 fc（我们自己使用）
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_layers = out_layers
        self.feat_dim_map = feat_dim_map

    def forward(self, x):
        """
        输入 x: (B, C, L)  — 注意：ResNet 是2D卷积网络，下面我们把一维信号扩成 pseudo-2D
        简单策略：将 1D 信号视作 (C=1, H=1, W=L)，通过 conv1(7x7) 处理时要保证形状兼容。
        更稳妥的方法是把 conv1 替换为 Conv1d 实现；但为了方便我这里采用以下方式：
        - 把输入从 (B,1,L) expand -> (B,3,L)（拷贝三通道），再 unsqueeze 成 (B,3,L,1) 或 (B,3,1,L) 以适配 torchvision conv2d。
        这里我们采用 (B,3,L,1) 并在 conv1 参数上依赖 padding/stride。实际使用时，推荐把 ResNet 的 conv1 改为 Conv2d(kernel=(1,7))
        """
        # 假定 x shape (B, 1, L)
        B = x.size(0)
        # 变成 (B,3,L,1)
        x2 = x.repeat(1,3,1)  # (B,3,L)
        x2 = x2.unsqueeze(3)  # (B,3,L,1)
        # Now treat width as H and height as W in conv2d: we will permute to (B,3,1,L) because torchvision conv expects (N,C,H,W)
        x2 = x2.permute(0,1,3,2)  # (B,3,1,L)

        # 走 ResNet 的前向（注意 conv1 expects kernel 7x7, works with H=1?)
        # 这 is a workaround; 更好的方法是定义自适应的 1D ResNet。
        out = self.conv1(x2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        feats = {}
        if 'layer2' in self.out_layers or 'layer3' in self.out_layers or 'layer4' in self.out_layers:
            out2 = self.layer2(out)
            feats['layer2'] = out2
            out3 = self.layer3(out2)
            feats['layer3'] = out3
            out4 = self.layer4(out3)
            feats['layer4'] = out4
        # 对每个层做全局池化并 flatten
        out_feats = []
        for layer in self.out_layers:
            f = feats[layer]  # (B, C_l, H_l, W_l)
            # 全局平均池化到 (B, C_l)
            f_pool = F.adaptive_avg_pool2d(f, (1,1)).view(B, -1)
            out_feats.append(f_pool)
        return out_feats  # list of tensors

# -------------------------
# JAN 网络（wrapper）：包含 backbone、多层 bottleneck、分类头
# -------------------------
class JAN(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', pretrained=False, out_layers=('layer2','layer3','layer4'), bottleneck_dim=256):
        super().__init__()
        self.backbone = ResNetMultiLayer(backbone=backbone, pretrained=pretrained, out_layers=out_layers)
        self.out_layers = out_layers
        # 对每个层创建一个 bottleneck 映射到相同的维度
        self.bottlenecks = nn.ModuleDict()
        for layer in out_layers:
            in_dim = self.backbone.feat_dim_map[layer]
            self.bottlenecks[layer] = nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True)
            )
        # 分类器（基于最后一层的 bottleneck）
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x):
        """
        返回:
         - logits (B, num_classes)
         - feats_for_jmmd: list of bottleneck outputs for each layer (顺序与 out_layers 一致)
        """
        multi_feats = self.backbone(x)  # list of raw pooled feats
        bottleneck_feats = []
        for layer, feat in zip(self.out_layers, multi_feats):
            b = self.bottlenecks[layer](feat)
            bottleneck_feats.append(b)
        # 最后一个 bottleneck 用作分类器输入
        logits = self.classifier(bottleneck_feats[-1])
        return logits, bottleneck_feats

if __name__ == '__main__':
    x = torch.randn(16, 1, 1024).cuda()
    model = JAN(7).cuda()
    y, z = model(x)
    print("logits:", y.shape)
    for i, f in enumerate(z):
        print(f"layer {model.out_layers[i]} bottleneck feat:", f.shape)