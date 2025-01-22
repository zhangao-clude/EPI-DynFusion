import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 压缩
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 激发
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 将特征图大小从 (B, C, L) -> (B, C)
        y = self.fc(y).view(b, c, 1)  # 将特征图大小从 (B, C) -> (B, C, 1)
        return x * y.expand_as(x)  # 按元素相乘，对应每个通道进行加权调整

# 示例用法
# 假设输入大小为 (batch_size, channels, length)
if __name__ == '__main__':

    input_tensor = torch.randn(1, 100, 194)  # (B, C, L)
    se_layer = SE1D(channel=100, reduction=16)
    output_tensor = se_layer(input_tensor)
    print(output_tensor.shape)
