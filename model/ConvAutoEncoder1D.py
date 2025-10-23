import torch
import torch.nn as nn

class ConvAutoEncoder1D(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder1D, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # [B, 16, 1250]
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # [B, 32, 625]
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, 313]
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 32, 625]
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # [B, 16, 1250]
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),  # [B, 1, 2500]
            nn.Tanh()  # 保证输出幅度 [-1, 1]，可换成 ReLU
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x[:, :, :2500]  # 强制裁剪多余长度
        return x


if __name__ == '__main__':
    x = torch.randn(64,1,2500).cuda()
    model = ConvAutoEncoder1D().cuda()
    y = model(x)
    print(y.shape)
