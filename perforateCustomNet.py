from Architectures.PerforatedConv2d import PerforatedConv2d
import torch


from types import MethodType


def replace_module(net, from_class, perforation_mode, pretrained):
    for name, submodule in net.named_children():
        if type(submodule) == from_class:
            original = getattr(net, name)
            new = PerforatedConv2d(original.in_channels, original.out_channels, original.kernel_size,
                                   original.stride, original.padding, original.dilation, original.groups,
                                   original.bias, original.device, perforation_mode=perforation_mode)
            if pretrained:
                with torch.no_grad():
                    new.weight = torch.nn.Parameter(torch.clone(original.weight))
                    if original.bias:
                        new.bias = torch.nn.Parameter(torch.clone(original.bias))
            setattr(net, name, new)
        elif len(list(submodule.named_children())) != 0:
            replace_module(submodule, from_class, perforation_mode=perforation_mode)


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list
def _get_perforation(self, part=None):
    if part is None:
        part = self
    convs = []
    for submodule in part.children():
        if type(submodule) == PerforatedConv2d:
            convs.append(submodule.perf_stride)
        elif len(list(submodule.children())) != 0:
            convs.append(self._get_perforation(submodule))
    return convs#flatten_list(convs)

def _set_perforation(self, perfs, part=None,start_n=[0]):
    if part is None:
        part = self
    for submodule in part.children():
        if type(submodule) == PerforatedConv2d:
            submodule.perf_stride = perfs[start_n[0]]
            submodule.recompute = True
            start_n[0] += 1
        elif len(list(submodule.children())) != 0:
            self._set_perforation(perfs, submodule, start_n)
    return self#flatten_list(convs)

def _reset(self):
    self.eval()
    self(torch.zeros(self.in_size, device=next(self.children()).device))
    self.train()
    return self

def _get_n_calc(self, part=None):
    if part is None:
        part = self
    convs = []
    for submodule in part.children():
        if type(submodule) == PerforatedConv2d:
            convs.append(submodule.calculations)
        elif len(list(submodule.children())) != 0:
            convs.append(self._get_n_calc(submodule))
    return convs#flatten_list(convs)

def add_functs(net):
    net._get_perforation = MethodType(_get_perforation, net)
    net._set_perforation = MethodType(_set_perforation, net)
    net._reset = MethodType(_reset, net)
    net._get_n_calc = MethodType(_get_n_calc, net)

def perforate_net(net, from_class=torch.nn.Conv2d, perforation_mode=(2,2), pretrained=False):
    replace_module(net, from_class=from_class, perforation_mode=perforation_mode, pretrained=pretrained)
    add_functs(net)
