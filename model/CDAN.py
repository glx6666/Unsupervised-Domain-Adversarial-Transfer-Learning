import torch
import torch.nn as nn

# 初始化函数保持不变
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------- ResNet1D 基本块 ----------------------
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---------------------- ResNet1D 主体 ----------------------
class CDAN(nn.Module):
    def __init__(self,  use_bottleneck=True, bottleneck_dim=512, in_channels=3):
        super(CDAN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1D, 64, 2)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.bottleneck = nn.Linear(512 * BasicBlock1D.expansion, bottleneck_dim)
            #self.fc = nn.Linear(bottleneck_dim, num_classes)
            self.__in_features = bottleneck_dim
        else:
            #self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.__in_features = 512 * BasicBlock1D.expansion

        self.apply(init_weights)

    def _make_layer(self, BasicBlock1D, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * BasicBlock1D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * BasicBlock1D.expansion),
            )
        layers = []
        layers.append(BasicBlock1D(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        return x

    def output_num(self):
        return self.__in_features

# ---------------------- ResNet1D-18 构造函数 ----------------------


# ---------------------- 域对抗网络保持不变 ----------------------
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ad_layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1, 3, 1024).to(device)  # 输入放到 GPU
    model = CDAN(in_channels=3).to(device)  # 模型也放到 GPU

    y = model(x)
    print(y.shape)
