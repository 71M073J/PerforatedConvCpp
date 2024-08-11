import torch
from torch import nn
from typing import Any, Callable, List, Optional, Type, Union

from torch import Tensor
from torch.nn import Sequential
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from .PerforatedConv2d import PerforatedConv2d
import numpy as np


class UNet(nn.Module):
    def _reset(self):
        self.eval()
        self(torch.zeros(self.in_size, device=self.conv1[0].weight.device))
        self.train()
        return self

    def __init__(self, out_channels, perforation_mode = None,
                 grad_conv: bool = True, in_size=(1, 3, 256, 256)):
        super(UNet, self).__init__()
        self.in_size = in_size
        self.perforation = perforation_mode
        if type(self.perforation) == tuple:
            self.perforation = [self.perforation] * 7
        elif type(self.perforation) not in [list, np.ndarray]:
            raise NotImplementedError("Provide the perforation mode")
        if 7 != len(self.perforation):
            raise ValueError(
                f"The perforation list length should equal the number of conv layers ({7}), given was {len(self.perforation)}")
        self.conv1 = self.contract_block(3, 64, 3, perforation_mode=self.perforation[0:2], grad_conv=grad_conv)
        self.conv2 = self.contract_block(64, 128, 3, perforation_mode=self.perforation[2:4], grad_conv=grad_conv)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv2_ex = self.expand_block(128, 64, 3, perforation_mode=self.perforation[4], grad_conv=grad_conv)
        self.upconv2 = self.contract_block(128, 64, 3, perforation_mode=self.perforation[5:7], grad_conv=grad_conv)

        self.upconv1 = PerforatedConv2d(64, out_channels, kernel_size=1,
                                        perforation_mode=(1, 1))  # always 1,1, since kernel size 1

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        y2 = self.upconv2_ex(x2)
        y2 = torch.cat((x1, y2), dim=1)
        y2 = self.upconv2(y2)

        y1 = self.upconv1(y2)
        return y1

    def contract_block(self, in_channels, out_channels, kernel_size, perforation_mode: list,
                       grad_conv: bool = True, ):
        return nn.Sequential(
            PerforatedConv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1,
                             perforation_mode=perforation_mode[0], grad_conv=grad_conv),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            PerforatedConv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1,
                             perforation_mode=perforation_mode[1], grad_conv=grad_conv),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def expand_block(self, in_channels, out_channels, kernel_size, perforation_mode: list,
                     grad_conv: bool = True, ):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            PerforatedConv2d(in_channels, out_channels, kernel_size, padding="same", stride=1, dilation=1,
                             perforation_mode=perforation_mode, grad_conv=grad_conv),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

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

        # self._reset()
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
        # perfs.append(ll.downsample[0].calculations)

        return perfs
