from typing import Callable

from torch import nn
import torch
import torch.nn.functional as F
import perfconv
from torch.utils.cpp_extension import load

#TODO in cpp file, add the dtype option as a param
class DeviceError(Exception):
    pass


# perfconv_cpu = load('perfconv_cpu', sources=['perfconv_cpu.cpp'], verbose=True)
class ConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, params):
        (dW, dH), (padW, padH), is_bias, perf_stride, device, (
        dil1, dil2), groups, upscale_conv, strided_backward, verbose, original_back = params
        # strided_backward = True
        kW, kH = weights.shape[2], weights.shape[3]
        # try:
        outputs = \
            perfconv.forward(input, weights, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
                             is_bias, device, dil1, dil2, groups, upscale_conv, verbose)[0]
        # except Exception:
        #    print(traceback.format_exc())
        #    print(input.shape, weights.shape, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
        #                                 is_bias, device, dil1, dil2, groups, upscale_conv)
        #    quit()
        ctx.params = (dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, original_back)
        variables = [input, weights, bias]
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        input, weights, bias = ctx.saved_tensors

        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, original_back = ctx.params
        kW, kH = weights.shape[2], weights.shape[3]

        gradInput, gradWeight, gradBias = perfconv.backward(input, gradOutput, weights,
                                                            kW, kH,  # kernel
                                                            dW, dH,  # stride
                                                            perf_stride[0], perf_stride[1], padW, padH,
                                                            is_bias, device, dil1, dil2, groups, strided_backward,
                                                            verbose, original_back)

        return gradInput, gradWeight, gradBias, None
class ConvFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, params):
        (dW, dH), (padW, padH), is_bias, perf_stride, device, (
        dil1, dil2), groups, upscale_conv, strided_backward, verbose, activ = params
        # strided_backward = True
        kW, kH = weights.shape[2], weights.shape[3]
        # try:
        outputs, outW, outH = perfconv.strided_down(input, weights, bias, kW, kH, dW, dH,
                                                    perf_stride[0], perf_stride[1], padW, padH,
                                                    is_bias, device, dil1, dil2, groups, upscale_conv, verbose)
        outputs = activ(outputs)
        outputs = perfconv.upscale(outputs, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
                             is_bias, device, dil1, dil2, groups, upscale_conv, verbose, outW, outH)[0]
        # except Exception:
        #    print(traceback.format_exc())
        #    print(input.shape, weights.shape, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
        #                                 is_bias, device, dil1, dil2, groups, upscale_conv)
        #    quit()
        ctx.params = (dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose)
        variables = [input, weights, bias]
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        input, weights, bias = ctx.saved_tensors

        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose = ctx.params
        kW, kH = weights.shape[2], weights.shape[3]

        gradInput, gradWeight, gradBias = perfconv.backward(input, gradOutput, weights,
                                                            kW, kH,  # kernel
                                                            dW, dH,  # stride
                                                            perf_stride[0], perf_stride[1], padW, padH,
                                                            is_bias, device, dil1, dil2, groups, strided_backward,
                                                            verbose)

        return gradInput, gradWeight, gradBias, None

class Upsample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, params):
        (dW, dH), (padW, padH), is_bias, perf_stride, device, (
            dil1, dil2), groups, upscale_conv, strided_backward, verbose, kW, kH = params
        # strided_backward = True
        # try:
        outputs = \
            perfconv.upsample(input, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
                              is_bias, device, dil1, dil2, groups, upscale_conv, verbose)[0]
        # except Exception:
        #    print(traceback.format_exc())
        #    print(input.shape, weights.shape, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
        #                                 is_bias, device, dil1, dil2, groups, upscale_conv)
        #    quit()
        ctx.params = (
        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, kW, kH)
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, kH, kW = ctx.params

        return gradOutput[:, :, perf_stride[0], perf_stride[1]], None


class PerforatedConv2d(nn.Module):
    def __repr__(self):
        return f"PerforatedConv2d({self.in_channels}, {self.out_channels}, perforation_mode={self.perf_stride})"
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0),
                 dilation=1, groups=1, bias=True, device=None, padding_mode=None,
                 perf_stride=None, upscale_conv=False, strided_backward=None, perforation_mode=None,
                 grad_conv=None, verbose=False, original_back=False):
        self.original_back = original_back
        super(PerforatedConv2d, self).__init__()
        self.verbose = verbose
        if strided_backward is None:
            if grad_conv is None:
                strided_backward = True
            else:
                strided_backward = grad_conv

        if not (padding_mode is None):
            if padding_mode != "zeros":
                raise ValueError(f"Unsupported padding mode \"{padding_mode}\", only supports zeros or None")
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.groups = groups
        self.in_channels = in_channels
        if self.in_channels % self.groups != 0:
            raise ValueError(f'in_channels {in_channels} must be divisible by groups {self.groups}')
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
        if self.stride[0] == 0 or self.stride[1] == 0:
            raise ValueError(f"Incorrect stride value: Cannot be zero, is {self.stride}")
        if perf_stride is None:
            if perforation_mode is not None:
                perf_stride = perforation_mode
            else:
                perf_stride = (1, 1)
        if type(perf_stride) == int:
            self.perf_stride = (perf_stride, perf_stride)
        elif type(perf_stride) == tuple:
            self.perf_stride = perf_stride
        else:
            raise TypeError(f"Incorrect perf_stride type: {type(perf_stride)}, with data: {perf_stride}")
        # self.params = self.stride[0], self.stride[1], self.padding[0], self.padding[1], is_bias

        self.upscale_conv = upscale_conv
        self.strided_backward = strided_backward

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // self.groups, self.kernel_size[0], self.kernel_size[1],
                        device=self.device))
        self.is_bias = bias
        if self.is_bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=self.device))
        else:
            self.bias = None
        self._initialize_weights()

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
        self.jitter = False
        self.hard_limit = (self.kernel_size[0] == 1 and self.kernel_size[1] == 1)

    def set_perf(self, perf):
        self.perf_stride = perf
        self.recompute = True

    # noinspection PyTypeChecker
    def _do_recomputing(self, shape):
        tmp = 0
        self.out_x = int((shape[-2] - ((self.kernel_size[0] - 1) * self.dilation[0]) + 2 * self.padding[0] - 1)
                         // self.stride[0] + 1)
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
        if input.device != self.weight.device:
            raise DeviceError(
                f"Expected both input and weight to be on the same device, got {input.device} and {self.weight.device}.")
        if self.hard_limit:
            self.perf_stride = (1, 1)
        if self.recompute:
            self._do_recomputing(input.shape)


        if self.perf_stride != (1, 1):
            #jitter = 0
            #if self.jitter:
            #    jitter = (self.mod1 - self.n1) % self.mod1, (self.mod2 - self.n2) % self.mod2
            #    if self.do_offsets:
            #        self.n1 = (self.n1 + 1) % self.mod1
            #        if self.n1 == 0:
            #            self.n2 = (self.n2 + 1) % self.mod2  # legit offseti
            #    print("trying to jitter")
            return ConvFunction.apply(input, self.weight, self.bias,
                                      (self.stride, self.padding, self.is_bias, self.perf_stride
                                       , self.device, self.dilation, self.groups, self.upscale_conv,
                                       self.strided_backward,
                                       self.verbose, self.original_back))
        else:
            # print("using torch impl")
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DownActivUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0),
                 dilation=1, groups=1, bias=True, device=None, padding_mode=None,
                 perf_stride=None, upscale_conv=False, strided_backward=None, perforation_mode=None,
                 grad_conv=None, verbose=False, activation: Callable = torch.nn.ReLU, original_back=False):
        super(DownActivUp, self).__init__()
        self.original_back = original_back
        self.verbose = verbose
        self.activation = activation()
        if strided_backward is None:
            if grad_conv is None:
                strided_backward = True
            else:
                strided_backward = grad_conv

        if not (padding_mode is None):
            if padding_mode != "zeros":
                raise ValueError(f"Unsupported padding mode \"{padding_mode}\", only supports zeros or None")
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.groups = groups
        self.in_channels = in_channels
        if self.in_channels % self.groups != 0:
            raise ValueError(f'in_channels {in_channels} must be divisible by groups {self.groups}')
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
        if self.stride[0] == 0 or self.stride[1] == 0:
            raise ValueError(f"Incorrect stride value: Cannot be zero, is {self.stride}")
        if perf_stride is None:
            if perforation_mode is not None:
                perf_stride = perforation_mode
            else:
                perf_stride = (1, 1)
        if type(perf_stride) == int:
            self.perf_stride = (perf_stride, perf_stride)
        elif type(perf_stride) == tuple:
            self.perf_stride = perf_stride
        else:
            raise TypeError(f"Incorrect perf_stride type: {type(perf_stride)}, with data: {perf_stride}")
        # self.params = self.stride[0], self.stride[1], self.padding[0], self.padding[1], is_bias

        self.upscale_conv = upscale_conv
        self.strided_backward = strided_backward

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // self.groups, self.kernel_size[0], self.kernel_size[1],
                        device=self.device))
        self.is_bias = bias
        if self.is_bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=self.device))
        else:
            self.bias = None
        self._initialize_weights()

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
        self.out_x = int((shape[-2] - ((self.kernel_size[0] - 1) * self.dilation[0]) + 2 * self.padding[0] - 1)
                         // self.stride[0] + 1)
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
            self.perf_stride = (1, 1)
        if self.recompute:
            self._do_recomputing(input.shape)

        if input.device != self.weight.device:
            raise DeviceError(
                f"Expected both input and weight to be on the same device, got {input.device} and {self.weight.device}.")
        if self.perf_stride != (1, 1):
            return ConvFunction.apply(input, self.weight, self.bias,
                                      (self.stride, self.padding, self.is_bias, self.perf_stride
                                       , self.device, self.dilation, self.groups, self.upscale_conv,
                                       self.strided_backward,
                                       self.verbose, self.original_back))
        else:
            # print("using torch impl")
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


if __name__ == "__main__":
    p = PerforatedConv2d
    c = p(64, 128, 3, perf_stride=2)
    h = c(torch.ones((2, 64, 10, 10)))
