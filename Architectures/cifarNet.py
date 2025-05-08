import torch
from torch import nn

def block(in_channels, out_channels, n):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU())
    for i in range(n):
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)
class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.activ = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=1, stride=2)#out=30
        self.bn1 = nn.BatchNorm2d(64)
        #mp1 = nn.MaxPool2d(3, stride=2, padding=1)#15

        self.b1 = block(64, 128, 1)
        self.b2 = block(128, 256, 1)
        self.linl = nn.Linear(16*256, 256)
        self.linnorm = nn.BatchNorm1d(256)
        self.linout = nn.Linear(256, num_classes)
        self.sm = nn.Softmax(1)

    def forward(self, x):
        x = self.activ(self.bn1(self.conv1(x)))
        x = self.b1(x)
        x = self.b2(x)
        x = self.activ(self.linnorm(self.linl(x.view(x.shape[0], -1))))
        x = self.sm(self.linout(x))
        return x

if __name__ == "__main__":
    net = CifarNet()
    net(torch.rand((2, 3, 32, 32)))