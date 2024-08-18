from Architectures.PerforatedConv2d import PerforatedConv2d, DownActivUp
import torch


from types import MethodType
import numpy as np


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
def _get_total_n_calc(net, perf_compare=(2,2)):
    default_perf = net._get_perforation()
    calculationsBase = net._set_perforation((1, 1))._reset()._get_n_calc()
    calculationsOther = net._set_perforation(perf_compare)._reset()._get_n_calc()

    base_n = np.array([x[0] for x in calculationsBase])
    base_o = np.array([x[0] for x in calculationsOther])
    #n_diff = sum([int(calculationsBase[i][0] == calculationsOther[i][0]) for i in range(len(calculationsBase))])
    net._set_perforation(default_perf)
    return f"{(base_n != base_o).sum()} out of {len(calculationsBase)} layers perforated, " \
           f"going from {base_n.sum()} to {base_o.sum()} operations ({(int(10000 * ( base_o.sum().astype(float)/base_n.sum().astype(float)))/100)}% of size).\n" \
           f"(Counting only Conv layers)"

def replace_module_perfconv(net, from_class, perforation_mode, pretrained):
    for name, submodule in net.named_children():
        if type(submodule) == from_class:
            original = getattr(net, name)
            new = PerforatedConv2d(original.in_channels, original.out_channels, original.kernel_size,
                                   original.stride, original.padding, original.dilation, original.groups,
                                   original.bias, original.weight.device, perforation_mode=perforation_mode)
            if pretrained:
                with torch.no_grad():
                    new.weight = torch.nn.Parameter(torch.clone(original.weight))
                    if original.bias:
                        new.bias = torch.nn.Parameter(torch.clone(original.bias))
            setattr(net, name, new)
        elif len(list(submodule.named_children())) != 0:
            replace_module_perfconv(submodule, from_class, perforation_mode=perforation_mode, pretrained=pretrained)


def replace_module_downActivUp(net, perforation_mode, pretrained, from_class=torch.nn.Conv2d, layers=None, activ_class=torch.nn.ReLU, start_n=0):
    raise NotImplementedError("i don't know how to dynamically adjust to operation order with different actiovations/batchnorms etc")
    print("this deletes ALL activations...how to implement=?")
    def get_layers(component):
        convs = []
        for submodule in component.children():
            if len(list(submodule.children())) != 0:
                convs.extend(flatten_list(get_layers(submodule)))
            else:
                convs.append(submodule)
        return flatten_list(convs)

    if layers is None:
        layers = get_layers(net)
        start_n = [0]
    for name, submodule in net.named_children():
        if type(submodule) == from_class:
            original = getattr(net, name)
            new = PerforatedConv2d(original.in_channels, original.out_channels, original.kernel_size,
                                   original.stride, original.padding, original.dilation, original.groups,
                                   original.bias, original.weight.device, perforation_mode=perforation_mode)
            if pretrained:
                with torch.no_grad():
                    new.weight = torch.nn.Parameter(torch.clone(original.weight))
                    if original.bias:
                        new.bias = torch.nn.Parameter(torch.clone(original.bias))
            setattr(net, name, new)
        elif len(list(submodule.named_children())) != 0:
            replace_module_perfconv(...)
        start_n[0]+=1

def _get_perforation(self, part=None):
    if part is None:
        part = self
    convs = []
    for submodule in part.children():
        if type(submodule) == PerforatedConv2d:
            convs.append(submodule.perf_stride)
        elif len(list(submodule.children())) != 0:
            convs.append(self._get_perforation(submodule))
    return flatten_list(convs)

def _set_perforation(self, perfs, part=None,start_n=0):
    if part is None:
        part = self
        start_n = [0]
    if type(perfs) == tuple:
        perfs = [perfs] * len(flatten_list(self._get_n_calc()))
    for submodule in part.children():
        if type(submodule) == PerforatedConv2d:
            submodule.perf_stride = perfs[start_n[0]]
            submodule.recompute = True
            start_n[0] += 1
        elif len(list(submodule.children())) != 0:
            self._set_perforation(perfs, submodule, start_n)
    self.perforation = perfs
    return self

def _reset(self):
    def recomp(net):
        for c in net.children():
            if type(c) == PerforatedConv2d:
                c.recompute = True
            else:
                recomp(c)
    recomp(self)
    def find_dev(net):
        for c in net.children():
            if list(c.children()) == [] and hasattr(c, "weight"):
                    return c.weight.device
            else:
                return find_dev(c)
        return "No device found?=?=?=?=?="
    self.eval()
    self(torch.zeros(self.in_size, device=find_dev(self)))
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
    return flatten_list(convs)

def add_functs(net):
    net._get_perforation = MethodType(_get_perforation, net)
    net._set_perforation = MethodType(_set_perforation, net)
    net._reset = MethodType(_reset, net)
    net._get_n_calc = MethodType(_get_n_calc, net)
    net._get_total_n_calc = MethodType(_get_total_n_calc, net)

def perforate_net_perfconv(net, from_class=torch.nn.Conv2d, perforation_mode=(2,2), pretrained=False, in_size=(1,3,512,512)):
    setattr(net, "in_size", in_size)
    replace_module_perfconv(net, from_class=from_class, perforation_mode=perforation_mode, pretrained=pretrained)
    add_functs(net)
    print(net.in_size)
    net._reset()
def perforate_net_downActivUp(net, from_class=torch.nn.Conv2d, perforation_mode=(2,2), pretrained=False, in_size=(1,3,512,512)):
    setattr(net, "in_size", in_size)
    replace_module_downActivUp(net, from_class=from_class, perforation_mode=perforation_mode, pretrained=pretrained)
    add_functs(net)
    print(net.in_size)
    net._reset()