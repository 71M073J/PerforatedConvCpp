import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, out_channels):
        super(UNet, self).__init__()
        self.conv1 = self.contract_block(3, 64, 3)
        self.conv2 = self.contract_block(64, 128, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv2_ex = self.expand_block(128, 64, 3)
        self.upconv2 = self.contract_block(128, 64, 3)
     
        self.upconv1 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        y2 = self.upconv2_ex(x2)
        y2 = torch.cat((x1, y2), dim=1)
        y2 = self.upconv2(y2)

        y1 = self.upconv1(y2)
        return y1

    def contract_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def expand_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", stride=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )