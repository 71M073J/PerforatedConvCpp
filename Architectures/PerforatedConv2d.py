import math
from torch import nn
import torch
import torch.nn.functional as F
import perfconv
#from torch.utils.cpp_extension import load

class DeviceError(Exception):
    pass


# perfconv_cpu = load('perfconv_cpu', sources=['perfconv_cpu.cpp'], verbose=True)
class ConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, params):
        (dW, dH), (padW, padH), is_bias, perf_stride, device, (
        dil1, dil2), groups, upscale_conv, strided_backward, verbose, original_conv_back = params
        # strided_backward = True
        kW, kH = weights.shape[2], weights.shape[3]
        # try:
        outputs = \
            perfconv.forward_newUp(input, weights, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
                             is_bias, device, dil1, dil2, groups, upscale_conv, verbose)[0]
        # except Exception:
        #    print(traceback.format_exc())
        #    print(input.shape, weights.shape, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
        #                                 is_bias, device, dil1, dil2, groups, upscale_conv)
        #    quit()
        ctx.params = (dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, original_conv_back)
        variables = [input, weights, bias]
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        input, weights, bias = ctx.saved_tensors

        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, original_conv_back = ctx.params
        kW, kH = weights.shape[2], weights.shape[3]
        gradInput, gradWeight, gradBias = perfconv.backward(input, gradOutput, weights,
                                                            kW, kH,  # kernel
                                                            dW, dH,  # stride
                                                            perf_stride[0], perf_stride[1], padW, padH,
                                                            is_bias, device, dil1, dil2, groups, strided_backward,
                                                            verbose, original_conv_back, False)

        return gradInput, gradWeight, gradBias, None

class Up(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, params):
        (dW, dH), (padW, padH), is_bias, perf_stride, device, (
            dil1, dil2), groups, upscale_conv, strided_backward, verbose, (kW, kH), outW, outH = params
        # strided_backward = True
        # try:

        outputs = \
            perfconv.upscale_newUp(input, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
                              is_bias, device, dil1, dil2, groups, upscale_conv, verbose, outW, outH)[0]
        # except Exception:
        #    print(traceback.format_exc())
        #    print(input.shape, weights.shape, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
        #                                 is_bias, device, dil1, dil2, groups, upscale_conv)
        #    quit()
        ctx.params = (
        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, kW, kH)
        return outputs#.clone()

    @staticmethod
    def backward(ctx, gradOutput):
        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, kH, kW = ctx.params
        return gradOutput[:, :, ::perf_stride[0], ::perf_stride[1]], None

class Down(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, params):
        (dW, dH), (padW, padH), is_bias, perf_stride, device, (
            dil1, dil2), groups, upscale_conv, strided_backward, verbose, original_conv_back, kW, kH = params
        # strided_backward = True
        # try:

        outputs = \
            perfconv.strided_down(input, weights, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
                              is_bias, device, dil1, dil2, groups, upscale_conv, verbose)[0]
        # except Exception:
        #    print(traceback.format_exc())
        #    print(input.shape, weights.shape, bias, kW, kH, dW, dH, perf_stride[0], perf_stride[1], padW, padH,
        #                                 is_bias, device, dil1, dil2, groups, upscale_conv)
        #    quit()
        variables = [input, weights, bias]
        ctx.save_for_backward(*variables)
        ctx.params = (
        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, original_conv_back, kW, kH)
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        dW, dH, padW, padH, is_bias, device, dil1, dil2, groups, perf_stride, strided_backward, verbose, original_conv_back, kH, kW = ctx.params
        input, weights, bias = ctx.saved_tensors
        gradInput, gradWeight, gradBias = perfconv.backward(input, gradOutput, weights,
                                                            kW, kH,  # kernel
                                                            dW, dH,  # stride
                                                            perf_stride[0], perf_stride[1], padW, padH,
                                                            is_bias, device, dil1, dil2, groups, strided_backward,
                                                            verbose, original_conv_back, True#noDownscale
                                                            )

        return gradInput, gradWeight, gradBias, None

class PerforatedConv2d(nn.Module):
    def __repr__(self):
        return f"PerforatedConv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, perforation_mode={self.perf_stride})"
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0),
                 dilation=1, groups=1, bias=True, device=None, padding_mode=None,
                 perf_stride=None, upscale_conv=False, strided_backward=None, perforation_mode=None,
                 grad_conv=None, verbose=False, original_conv_back=False, init_weights=True):
        self.original_conv_back = original_conv_back
        super(PerforatedConv2d, self).__init__()
        self.verbose = verbose
        #TODO
        strided_backward = True


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
        elif type(padding) == str:
            if padding == "same":
                self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)
            elif padding == "none":
                self.padding = (0,0)
            else:
                raise ValueError("String padding modes other than \"same\" and \"none\" are not supported")
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
        if bias is None:
            bias = False
        if type(bias) == bool:
            self.is_bias = bias
            if bias:
                self.bias = nn.Parameter(torch.empty(out_channels, device=self.device))
            else:
                self.bias = None
        else:
            self.bias = nn.Parameter(torch.clone(bias))
            self.is_bias = True
        if init_weights:
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
        self.inputshape = (1,1,0,0)

    def _get_calculations(self):
        # in_channels * out_channels * h * w * filter_size // stride1 // stride2
        self.calculations = ((((self.in_channels * self.out_channels * self.inputshape[0] *
                              (self.inputshape[-2] - self.kernel_size[0] // 2 * self.dilation[0] * 2 + self.padding[0] * 2) *
                              (self.inputshape[-1] - self.kernel_size[1] // 2 * self.dilation[1] * 2 + self.padding[1] * 2) *
                              self.kernel_size[0] * self.kernel_size[1]) // self.groups) //
                             self.stride[0]) // self.stride[1] //
                            self.perf_stride[0] // self.perf_stride[1])
        self.calculations += ((self.out_channels * self.inputshape[0] * #+interpolacija
                            (self.inputshape[-2] - self.kernel_size[0] // 2 * self.dilation[0] * 2 + self.padding[0] * 2) *
                            (self.inputshape[-1] - self.kernel_size[1] // 2 * self.dilation[1] * 2 + self.padding[1] * 2))\
                            //self.stride[0]//self.stride[1]//
                            self.perf_stride[0]//self.perf_stride[1]
                            * (self.perf_stride[0] *self.perf_stride[1] - 1))


            #f"{self.in_channels}x" \
            #f"{(self.inputshape[-2] - self.kernel_size[0] // 2 * 2 + self.padding[0] * 2)}x" \
            #f"{(self.inputshape[-1] - self.kernel_size[1] // 2 * 2 + self.padding[1] * 2)}x" \
            #f"{self.out_channels}x{self.kernel_size[0]}x{self.kernel_size[1]}//" \
            #f"{self.stride[0]}//{self.stride[1]}//{self.perf_stride[0]}//{self.perf_stride[1]}"
    def set_perf(self, perf):
        self.perf_stride = perf
        self.recompute = True

    # noinspection PyTypeChecker
    def _do_recomputing(self, shape):
        self.inputshape = shape
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
        self.recompute = False



    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

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
                                       self.verbose, self.original_conv_back))
        else:
            # print("using torch impl")
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DownActivUp(nn.Module):
    def __repr__(self):
        return f"DownActivUp({self.in_channels}, {self.out_channels}, {self.kernel_size}, perforation_mode={self.perf_stride}, activ={self.activ.__repr__()})"
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0),
                 dilation=1, groups=1, bias=True, device=None, padding_mode=None, activation=torch.nn.Identity(),
                 perf_stride=None, upscale_conv=False, strided_backward=None, perforation_mode=None,
                 grad_conv=None, verbose=False, original_conv_back=False, init_weights=True, up=True, down=True):
        self.original_conv_back = original_conv_back
        super(DownActivUp, self).__init__()
        self.up = up
        self.down = down
        self.verbose = verbose
        if strided_backward is None:
            if grad_conv is None:
                strided_backward = True
            else:
                strided_backward = grad_conv
        self.activ = activation
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
        elif type(padding) == str:
            if padding == "same":
                self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)
            elif padding == "none":
                self.padding = (0,0)
            else:
                raise ValueError("String padding modes other than \"same\" and \"none\" are not supported")
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
        if bias is None:
            bias = False
        if type(bias) == bool:
            self.is_bias = bias
            if bias:
                self.bias = nn.Parameter(torch.empty(out_channels, device=self.device))
            else:
                self.bias = None
        else:
            self.bias = nn.Parameter(torch.clone(bias))
            self.is_bias = True
        if init_weights:
            self._initialize_weights()
        self.is_bias = False
        self.bias = None
        if type(bias) == bool:
            self.is_bias = bias
            if bias:
                self.bias = nn.Parameter(torch.empty(out_channels, device=self.device))
            else:
                self.bias = None
        elif bias is not None:
            self.bias = nn.Parameter(torch.clone(bias))
            self.is_bias = True
        if init_weights:
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
        self.inputshape = (1,1,0,0)

    def set_perf(self, perf):
        self.perf_stride = perf
        self.recompute = True
    def _get_calculations(self):
        # in_channels * out_channels * h * w * filter_size // stride1 // stride2
        self.calculations = ((((self.in_channels * self.out_channels * self.inputshape[0] *
                              (self.inputshape[-2] - self.kernel_size[0] // 2 * self.dilation[0] * 2 + self.padding[0] * 2) *
                              (self.inputshape[-1] - self.kernel_size[1] // 2 * self.dilation[1] * 2 + self.padding[1] * 2) *
                              self.kernel_size[0] * self.kernel_size[1]) // self.groups) //
                             self.stride[0]) // self.stride[1] //
                            self.perf_stride[0] // self.perf_stride[1])
        self.calculations += ((self.out_channels * self.inputshape[0] * #+interpolacija
                            (self.inputshape[-2] - self.kernel_size[0] // 2 * self.dilation[0] * 2 + self.padding[0] * 2) *
                            (self.inputshape[-1] - self.kernel_size[1] // 2 * self.dilation[1] * 2 + self.padding[1] * 2))\
                            //self.stride[0]//self.stride[1]//
                            self.perf_stride[0]//self.perf_stride[1]
                            * (self.perf_stride[0] *self.perf_stride[1] - 1))

    # noinspection PyTypeChecker
    def _do_recomputing(self, shape):
        self.inputshape = shape
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


    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

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
            if self.down:
                x = Down.apply(input, self.weight, self.bias, (self.stride, self.padding, self.is_bias, self.perf_stride
                                       , self.device, self.dilation, self.groups, self.upscale_conv,
                                       self.strided_backward,
                                       self.verbose, self.original_conv_back, self.kernel_size[0], self.kernel_size[1]))
            else:
                x = torch.rand((input.shape[0], self.out_channels, (input.shape[2] - 2*(self.kernel_size[0]//2))//2, (input.shape[3] - 2*(self.kernel_size[0]//2))//2))
            x = self.activ(x)
            if self.up:
                return Up.apply(x, (self.stride, self.padding, self.is_bias, self.perf_stride, self.device, self.dilation,
                                self.groups, self.upscale_conv, self.strided_backward, self.verbose, self.kernel_size,
                                self.out_x, self.out_y))
            else:
                return x
            #return ConvFunction.apply(input, self.weight, self.bias,
            #                          (self.stride, self.padding, self.is_bias, self.perf_stride
            #                           , self.device, self.dilation, self.groups, self.upscale_conv,
            #                           self.strided_backward,
            #                           self.verbose, self.original_conv_back))
        else:
            # print("using torch impl")
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


if __name__ == "__main__":
    p = PerforatedConv2d
    c = p(64, 128, 3, perf_stride=2)
    h = c(torch.ones((2, 64, 10, 10)))
