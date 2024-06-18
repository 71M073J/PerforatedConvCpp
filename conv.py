from torch import nn
import torch
import torch.nn.functional as F
import perfconv
from torch.utils.cpp_extension import load

class DeviceError(Exception):
    pass
# perfconv_cpu = load('perfconv_cpu', sources=['perfconv_cpu.cpp'], verbose=True)
class ConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, params):
                    
        (dW, dH), (padW, padH), is_bias, perf_stride, device, (dil1, dil2), groups, upscale_conv, strided_backward = params

        kW, kH = weights.shape[2], weights.shape[3]

        outputs = \
            perfconv.forward(input, weights, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
                                     is_bias, device, dil1, dil2, groups, upscale_conv)[0]

        ctx.params = (dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward)
        variables = [input, weights, bias]
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):

    
        input, weights, bias = ctx.saved_tensors

        dW, dH, padW, padH, is_bias, device, dil1 ,dil2, groups, perf_stride, strided_backward = ctx.params
        kW, kH = weights.shape[2], weights.shape[3]

        gradInput, gradWeight, gradBias = perfconv.backward(input, gradOutput, weights,
                                kW, kH, dW, dH,
                                perf_stride[0],perf_stride[1], padW, padH,
                                is_bias, device, dil1, dil2, groups, strided_backward)

        return gradInput, gradWeight, gradBias, None


class PerforatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,1), padding=(0,0),
                 stride=1, dilation=1, groups=1, is_bias=True, device=None,
                 silent=False, perf_stride=(1,1), upscale_conv=False, strided_backward=False):
        super(PerforatedConv2d, self).__init__()
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        elif type(kernel_size) == tuple:
            self.kernel_size = kernel_size
        else:
            raise TypeError(f"Incorrect kernel_size type: {type(kernel_size)}, with data: {kernel_size}")
        if type(padding) == int:
            self.padding = (padding, padding)
        elif type(padding) == tuple:
            self.padding = padding
        else:
            raise TypeError(f"Incorrect padding type: {type(padding)}, with data: {padding}")
        if type(dilation) == int:
            self.dilation = (dilation, dilation)
        elif type(dilation) == tuple:
            self.dilation = dilation
        else:
            raise TypeError(f"Incorrect dilation type: {type(dilation)}, with data: {dilation}")
        if type(stride) == int:
            self.stride = (stride, stride)
        elif type(stride) == tuple:
            self.stride = stride
        else:
            raise TypeError(f"Incorrect stride type: {type(stride)}, with data: {stride}")
        self.is_bias = is_bias
        if type(perf_stride) == int:
            self.perf_stride = (perf_stride, perf_stride)
        elif type(perf_stride) == tuple:
            self.perf_stride = perf_stride
        else:
            raise TypeError(f"Incorrect perf_stride type: {type(perf_stride)}, with data: {perf_stride}")
        #self.params = self.stride[0], self.stride[1], self.padding[0], self.padding[1], is_bias

        self.upscale_conv = upscale_conv
        self.strided_backward = strided_backward

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1], device=self.device))
        self.bias = nn.Parameter(torch.empty(out_channels, device=self.device))
        self._initialize_weights()
        if not silent:
            print("put logic for deciding whether to use torch impl or not and which perf stride into c++?")

        self.out_x = 0
        self.out_y = 0
        self.n1 = 0
        self.n2 = 0
        self.mod1 = 1
        self.mod2 = 1
        self.recompute = True
        self.calculations = 0
        self.in_shape = None
        self.do_offsets = False
        self.hard_limit = (self.kernel_size[0] == 1 and self.kernel_size[1] == 1)

    def set_perf(self, perf):
        self.perf_stride = perf
        self.recompute = True

    # noinspection PyTypeChecker
    def _do_recomputing(self, shape):
        tmp = 0
        self.out_x = int(
            (shape[-2] - ((self.kernel_size[0] - 1) * self.dilation[0]) + 2 * self.padding[
                0] - 1) // self.stride[0] + 1)
        tmp_stride1 = self.perf_stride[0] + 1
        while tmp <= 1:
            tmp_stride1 -= 1
            if tmp_stride1 == 0:
                tmp_stride1 = 1
                break
            tmp = int((shape[-2] - ((self.kernel_size[0] - 1) * self.dilation[0]) + 2 *
                       self.padding[0] - 1) // (self.stride[0] * tmp_stride1) + 1)

        tmp = 0
        self.out_y = int(
            (shape[-1] - ((self.kernel_size[1] - 1) * self.dilation[1]) + 2 * self.padding[
                1] - 1) // self.stride[1] + 1)
        tmp_stride2 = self.perf_stride[1] + 1
        while tmp <= 1:
            tmp_stride2 -= 1
            if tmp_stride2 == 0:
                tmp_stride2 = 1
                break
            tmp = int((shape[-1] - ((self.kernel_size[1] - 1) * self.dilation[1]) + 2 *
                       self.padding[1] - 1) // (self.stride[1] * tmp_stride2) + 1)
        self.perf_stride = (tmp_stride1, tmp_stride2)

        self.mod1 = ((self.out_x - 1) % self.perf_stride[0]) + 1
        self.mod2 = ((self.out_y - 1) % self.perf_stride[1]) + 1
        self.recompute = False
        # in_channels * out_channels * h * w * filter_size // stride1 // stride2
        self.calculations = ((self.in_channels * self.out_channels *
                              (shape[-2] - self.kernel_size[0] // 2 * 2 + self.padding[0] * 2) *
                              (shape[-1] - self.kernel_size[1] // 2 * 2 + self.padding[1] * 2) *
                              self.kernel_size[0] * self.kernel_size[1]) //
                             self.stride[0]) // self.stride[1] // \
                            self.perf_stride[0] // self.perf_stride[1], \
            f"{self.in_channels}x" \
            f"{(shape[-2] - self.kernel_size[0] // 2 * 2 + self.padding[0] * 2)}x" \
            f"{(shape[-1] - self.kernel_size[1] // 2 * 2 + self.padding[1] * 2)}x" \
            f"{self.out_channels}x{self.kernel_size[0]}x{self.kernel_size[1]}//" \
            f"{self.stride[0]}//{self.stride[1]}//{self.perf_stride[0]}//{self.perf_stride[1]}"



    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.is_bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):

        if self.hard_limit:
            self.perf_stride = (1,1)
        if self.recompute:
            self._do_recomputing(input.shape)

        if input.device != self.weight.device:
            raise DeviceError(f"Expected both input and weight to be on the same device, got {input.device} and {self.weight.device}.")
        if self.perf_stride != (1, 1):
            return ConvFunction.apply(input, self.weight, self.bias,
                                      (self.stride, self.padding, self.is_bias,self.perf_stride
                                      ,self.device, self.dilation, self.groups, self.upscale_conv, self.strided_backward))
        else:
            #print("using torch impl")
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)