import torch
from torch import nn
#from perforateCustomNet import perforate_net_perfconv, perforate_net_downActivUp
from Architectures.PerforatedConv2d import PerforatedConv2d, DownActivUp
class UNet(nn.Module):
    def __init__(self, out_channels):
        super(UNet, self).__init__()
        self.conv1 = self.contract_block1(3, 64, 5)
        self.conv2 = self.contract_block2(64, 128, 5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ups = self.upscale_block(128, 64)
        self.upconv2_ex = self.expand_block(192, 64, 3, 2)
        self.upconv2 = self.contract_block2(128, 64, 3)
        self.upconv1 = self.expand_block(64, 64, kernel_size=3, padding=1)
        self.outconv = nn.Conv2d(64+3, out_channels, kernel_size=1)
    def forward(self, X):
        x_ = X
        x1 = self.conv1(X)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x22 = self.maxpool(x2)
        x3 = self.ups(x22)
        y2 = self.upconv2_ex(torch.cat((x2, x3), dim=1))
        y2 = torch.cat((x1, y2), dim=1)
        y2 = self.upconv2(y2)
        y2 = self.upconv1(y2)
        y2 = torch.cat((x_, y2), dim=1)
        y1 = self.outconv(y2)
        return y1
    def upscale_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def contract_block1(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=4, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def contract_block2(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def expand_block(self, in_channels, out_channels, kernel_size,padding):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class UNetPerf(nn.Module):
    def __init__(self, out_channels):
        super(UNetPerf, self).__init__()
        self.conv1 = self.contract_block1(3, 64, 5)
        self.conv2 = self.contract_block2(64, 128, 5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ups = self.upscale_block(128, 64)
        self.upconv2_ex = self.expand_block(192, 64, 3, 2)
        self.upconv2 = self.contract_block2(128, 64, 3)

        self.upconv1 = self.expand_block(64, 64, kernel_size=3, padding=1)
        self.outconv = nn.Conv2d(64+3, out_channels, kernel_size=1)

    def forward(self, X):
        x_ = X
        x1 = self.conv1(X)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x22 = self.maxpool(x2)
        x3 = self.ups(x22)
        y2 = self.upconv2_ex(torch.cat((x2, x3), dim=1))
        y2 = torch.cat((x1, y2), dim=1)
        y2 = self.upconv2(y2)
        y2 = self.upconv1(y2)
        y2 = torch.cat((x_, y2), dim=1)
        y1 = self.outconv(y2)
        return y1
    def upscale_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def contract_block1(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            PerforatedConv2d(in_channels, out_channels, kernel_size, stride=2, padding=4, dilation=2, perf_stride=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            PerforatedConv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1, perf_stride=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def contract_block2(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            PerforatedConv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1, perf_stride=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            PerforatedConv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1, perf_stride=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def expand_block(self, in_channels, out_channels, kernel_size,padding):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            PerforatedConv2d(in_channels, out_channels, kernel_size, padding=padding, stride=1, dilation=1, perf_stride=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

if __name__ == "__main__":
    net = UNet(3)
    net(torch.ones((1, 3, 256, 256)))
