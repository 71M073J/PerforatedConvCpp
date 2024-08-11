import torch
from torch import nn
import numpy as np

from .PerforatedConv2d import PerforatedConv2d
class SimpleConv(nn.Module):
    def _reset(self):
        self.eval()
        self(torch.zeros(self.in_size, device=self.conv1[0].weight.device))
        self.train()
        return self
    def __init__(self, in_channels, out_channels, perforation_mode: list = None,
            grad_conv: bool = True, in_size=None):
        super(SimpleConv, self).__init__()
        if in_size is None:
            self.in_size = (1, in_channels, 256, 256)
        else:
            self.in_size = in_size
        self.perforation = perforation_mode
        if type(self.perforation) == tuple:
            self.perforation = [self.perforation] * 12
        elif type(self.perforation) not in [list, np.ndarray]:
            raise NotImplementedError("Provide the perforation mode")
        if 12 != len(self.perforation):
            raise ValueError(
                f"The perforation list length should equal the number of conv layers ({12}), given was {len(self.perforation)}")
        self.conv1 = self.contract_block(in_channels, 8, 7, 3, perforation_mode=self.perforation[0:2], grad_conv=grad_conv)
        self.conv2 = self.contract_block(8, 16, 3, 1, perforation_mode=self.perforation[2:4], grad_conv=grad_conv)
        self.conv3 = self.contract_block(16, 32, 3, 1, perforation_mode=self.perforation[4:6], grad_conv=grad_conv)

        self.upconv3 = self.expand_block(32, 16, 3, 1, perforation_mode=self.perforation[6:8], grad_conv=grad_conv)
        self.upconv2 = self.expand_block(16 * 2, 8, 3, 1, perforation_mode=self.perforation[8:10], grad_conv=grad_conv)
        self.upconv1 = self.expand_block(8 * 2, out_channels, 3, 1, perforation_mode=self.perforation[10:12], grad_conv=grad_conv)

    def forward(self, X):
        conv1 = self.conv1(X)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding, perforation_mode: list = None,
            grad_conv: bool = True,):
        contract = nn.Sequential(
            PerforatedConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,perforation_mode=perforation_mode[0], grad_conv=grad_conv
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            PerforatedConv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,perforation_mode=perforation_mode[1], grad_conv=grad_conv
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding, perforation_mode: list = None,
            grad_conv: bool = True):
        expand = nn.Sequential(
            PerforatedConv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding,perforation_mode=perforation_mode[0], grad_conv=grad_conv
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            PerforatedConv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding,perforation_mode=perforation_mode[1], grad_conv=grad_conv
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return expand


    def _set_perforation(self, perf):
        if type(perf) == tuple:
            perf = [perf] * len(self._get_n_calc())
        self.perforation = perf
        cnt = 0
        for i in self.children():
            if type(i) == torch.nn.Sequential:
                for j in i:
                    if type(j) == PerforatedConv2d:
                        j.perf_stride = perf[cnt]
                        j.recompute = True
                        cnt += 1
            elif type(i) == PerforatedConv2d:
                i.perf_stride = perf[cnt]
                i.recompute = True
                cnt += 1

        #self._reset()
        return self

    def _get_perforation(self):
        perfs = []
        for i in self.children():
            if type(i) == torch.nn.Sequential:
                for j in i:
                    if type(j) == PerforatedConv2d:
                        perfs.append(j.perf_stride)
            elif type(i) == PerforatedConv2d:
                perfs.append(i.perf_stride)
        self.perforation = perfs
        return perfs
    def _get_n_calc(self):
        perfs = []
        for i in self.children():
            if type(i) == torch.nn.Sequential:
                for j in i:
                    if type(j) == PerforatedConv2d:
                        perfs.append(j.calculations)
            elif type(i) == PerforatedConv2d:
                perfs.append(i.calculations)
        #perfs.append(ll.downsample[0].calculations)

        return perfs
