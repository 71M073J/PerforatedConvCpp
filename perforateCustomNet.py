import copy

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


def _get_total_n_calc(net, perf_compare=(2, 2)):
    default_perf = net._get_perforation()
    calculationsBase = net._set_perforation((1, 1))._reset()._get_n_calc()
    calculationsOther = net._set_perforation(perf_compare)._reset()._get_n_calc()

    base_n = np.array([x[0] for x in calculationsBase])
    base_o = np.array([x[0] for x in calculationsOther])
    # n_diff = sum([int(calculationsBase[i][0] == calculationsOther[i][0]) for i in range(len(calculationsBase))])
    net._set_perforation(default_perf)
    return f"{(base_n != base_o).sum()} out of {len(calculationsBase)} layers perforated, " \
           f"going from {base_n.sum()} to {base_o.sum()} operations ({(int(10000 * (base_o.sum().astype(float) / base_n.sum().astype(float))) / 100)}% of size).\n" \
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
                    if original.bias is not None or (type(original.bias) == bool and original.bias):
                        new.bias = torch.nn.Parameter(torch.clone(original.bias))
            setattr(net, name, new)
        elif len(list(submodule.named_children())) != 0:
            replace_module_perfconv(submodule, from_class, perforation_mode=perforation_mode, pretrained=pretrained)


def replace_module_downActivUp_backup(net, perforation_mode, pretrained=False, from_class=torch.nn.Conv2d, layers=None,
                               start_n=0, replace_activs=False, verbose=False):
    # raise NotImplementedError("i don't know how to dynamically adjust to operation order with different actiovations/batchnorms etc")
    # print("this deletes ALL activations...how to implement=?")
    def get_layers(component):
        convs = []
        for submodule in component.named_children():
            if len(list(submodule[1].named_children())) != 0:
                convs.extend(flatten_list(get_layers(submodule[1])))
            else:
                convs.append(submodule)
        return flatten_list(convs)

    if layers is None:
        layers = get_layers(net)
        start_n = [0]
    if verbose:
        print([x for x in layers[start_n[0]:start_n[0] + min(len(list(net.named_children())), 10)]])
    for name, submodule in net.named_children():
        # print("Testing", str(submodule) if type(submodule) != torch.nn.Sequential else "Sequential(...)")
        if type(submodule) == from_class:
            replace_activs = True
            cnt = 1
            newActivs = []
            skipWhile = False
            if len(layers) <= start_n[0] + 1:
                print("We have reached end of layers array, skipping...")
                skipWhile = True

            if verbose:
                print(submodule, layers[start_n[0]][1])
                print("found layer", str(layers[start_n[0]][1]), "Looking for shape-preserving layers...")
            while not skipWhile:
                if any(map(str(layers[start_n[0] + cnt][1]).__contains__, ["Dropout", "Norm", "ReLU", "ELU",
                                                                           "Hardshrink", "Hardsigmoid", "Hardtanh",
                                                                           "Hardswish", "Sigmoid", "SiLU", "Mish",
                                                                           "Softplus", "Softshrink", "Softsign", "Tanh",
                                                                           "Threshold", "GLU"])):
                    if verbose:
                        print("Layer", str(layers[start_n[0] + cnt][1]), "found, adding to list")
                    newActivs.append(copy.deepcopy(layers[start_n[0] + cnt][1]))
                    cnt += 1
                else:
                    if verbose:
                        print("Layer", str(layers[start_n[0] + cnt][1]), "not shape preserving, apparently")
                    break
            if type(net) != torch.nn.Sequential:
                original = getattr(net, name)
            else:
                original = net[int(name)]
            if verbose:
                print("Final List of mid-layers:", newActivs)
            if not skipWhile:
                if verbose:
                    print("Creating new DAU layer")
                print("\n\n", newActivs, "\n\n")
                new = DownActivUp(original.in_channels, original.out_channels, original.kernel_size,
                                  original.stride, original.padding, original.dilation, original.groups,
                                  original.bias, original.weight.device, perforation_mode=perforation_mode,
                                  activation=torch.nn.Sequential(*newActivs))
                print(new.activ)
                print("\n\n")
            else:
                ...#output layer should? be non-perforated maybe
                new = PerforatedConv2d(original.in_channels, original.out_channels, original.kernel_size,
                                       original.stride, original.padding, original.dilation, original.groups,
                                       original.bias, original.weight.device, perforation_mode=perforation_mode)
            if verbose:
                print("New layer:", new, "for name", name, "on layer", net._get_name())
            if pretrained:
                print(pretrained, submodule, name)
                with torch.no_grad():
                    new.weight = torch.nn.Parameter(torch.clone(original.weight))
                    if type(original.bias) == bool:
                        if original.bias:
                            new.bias = torch.nn.Parameter(torch.clone(original.bias))
                    elif hasattr(original.bias, "shape"):
                        new.bias = torch.nn.Parameter(torch.clone(original.bias))
            if type(net) != torch.nn.Sequential:
                setattr(net, name, new)
            else:
                net[int(name)] = new
            start_n[0] += 1
        elif len(list(submodule.children())) == 0 and any(
                map(str(submodule).__contains__, ["Dropout", "Norm", "ReLU", "ELU",
                                                  "Hardshrink", "Hardsigmoid", "Hardtanh", "Hardswish", "Sigmoid",
                                                  "SiLU", "Mish",
                                                  "Softplus", "Softshrink", "Softsign", "Tanh", "Threshold", "GLU"])):
            start_n[0] += 1
            if replace_activs:
                if verbose:
                    print("Setting layer", submodule, "to Identity()...")
                #setattr(net, name, torch.nn.Identity())
                #TODO
            else:
                replace_activs = False
        elif len(list(submodule.children())) != 0:
            if verbose:
                print("Recursing deeper...")
            replace_module_downActivUp(submodule, perforation_mode, pretrained, from_class=from_class, layers=layers,
                                       start_n=start_n, replace_activs=replace_activs, verbose=verbose)
        else:
            #None of the valid replacement layers
            start_n[0] += 1

def replace_module_downActivUp(net, perforation_mode, pretrained=False, from_class=torch.nn.Conv2d, layers=None,
                               start_n=0, replace_activs=False, verbose=False, n_to_replace=0):
    # raise NotImplementedError("i don't know how to dynamically adjust to operation order with different actiovations/batchnorms etc")
    # print("this deletes ALL activations...how to implement=?")
    def get_layers(component):
        convs = []
        for submodule in component.named_children():
            if len(list(submodule[1].named_children())) != 0:
                convs.extend(flatten_list(get_layers(submodule[1])))
            else:
                convs.append(submodule)
        return flatten_list(convs)

    if layers is None:
        layers = get_layers(net)
        start_n = [0]
    if verbose:
        print("Layers applicable currently: ", [x for x in layers[start_n[0]+1:start_n[0] + min(len(list(net.named_children())), 10)]])
    for name, submodule in net.named_children():
        # print("Testing", str(submodule) if type(submodule) != torch.nn.Sequential else "Sequential(...)")
        if type(submodule) == from_class:
            replace_activs = True
            cnt = 1
            newActivs = []
            skipWhile = False
            if len(layers) <= start_n[0] + 1:
                if verbose:
                    print("We have reached end of layers array, skipping...")
                skipWhile = True

            if verbose:
                print(submodule, layers[start_n[0]][1])
                print("found layer", str(layers[start_n[0]][1]), "Looking for shape-preserving layers...")
            while not skipWhile:
                if start_n[0] + cnt >= len(layers):
                    if verbose:
                        print("At the end?=")
                        print(layers, start_n, cnt)
                    break
                if any(map(str(layers[start_n[0] + cnt][1]).__contains__, ["Dropout", "Norm", "ReLU", "ELU",
                                                                           "Hardshrink", "Hardsigmoid", "Hardtanh",
                                                                           "Hardswish", "Sigmoid", "SiLU", "Mish",
                                                                           "Softplus", "Softshrink", "Softsign", "Tanh",
                                                                           "Threshold", "GLU"])):
                    if verbose:
                        print("Layer", str(layers[start_n[0] + cnt][1]), "found, adding to list")
                    newActivs.append(copy.deepcopy(layers[start_n[0] + cnt][1]))
                    cnt += 1
                else:
                    if verbose:
                        print("Layer", str(layers[start_n[0] + cnt][1]), "not shape preserving, apparently")
                    break
            if type(net) != torch.nn.Sequential:
                original = getattr(net, name)
            else:
                original = net[int(name)]
            if verbose:
                print("Final List of mid-layers:", newActivs)
            if not skipWhile:
                if verbose:
                    print("Creating new DAU layer")
                new = DownActivUp(original.in_channels, original.out_channels, original.kernel_size,
                                  original.stride, original.padding, original.dilation, original.groups,
                                  original.bias, original.weight.device, perforation_mode=perforation_mode,
                                  activation=torch.nn.Sequential(*newActivs))

            else:
                ...#output layer should? be non-perforated maybe
                new = PerforatedConv2d(original.in_channels, original.out_channels, original.kernel_size,
                                       original.stride, original.padding, original.dilation, original.groups,
                                       original.bias, original.weight.device, perforation_mode=perforation_mode)
            if verbose:
                print("New layer:", new, "for name", name, "on layer", net._get_name())
            if pretrained:
                if verbose:
                    print(pretrained, submodule, name)
                with torch.no_grad():
                    new.weight = torch.nn.Parameter(torch.clone(original.weight))
                    if type(original.bias) == bool:
                        if original.bias:
                            new.bias = torch.nn.Parameter(torch.clone(original.bias))
                    elif hasattr(original.bias, "shape"):
                        new.bias = torch.nn.Parameter(torch.clone(original.bias))
            if type(net) != torch.nn.Sequential:
                setattr(net, name, new)
            else:
                net[int(name)] = new
            start_n[0] += 1
        elif len(list(submodule.children())) == 0 and any(
                map(str(submodule).__contains__, ["Dropout", "Norm", "ReLU", "ELU",
                                                  "Hardshrink", "Hardsigmoid", "Hardtanh", "Hardswish", "Sigmoid",
                                                  "SiLU", "Mish",
                                                  "Softplus", "Softshrink", "Softsign", "Tanh", "Threshold", "GLU"])):
            start_n[0] += 1
            if replace_activs:
                if verbose:
                    print("Setting layer", submodule, "to Identity()...")
                if type(net) != torch.nn.Sequential:
                    setattr(net, name, torch.nn.Identity())
                else:
                    net[int(name)] = torch.nn.Identity()
            else:
                replace_activs = False
        elif len(list(submodule.children())) != 0:
            if verbose:
                print("Recursing deeper into...", submodule)
            replace_module_downActivUp(submodule, perforation_mode, pretrained, from_class=from_class, layers=layers,
                                       start_n=start_n, replace_activs=replace_activs, verbose=verbose)
        else:
            #None of the valid replacement layers
            start_n[0] += 1

def _get_perforation(self, part=None):
    if part is None:
        part = self
    convs = []
    for submodule in part.children():
        if type(submodule) in [PerforatedConv2d, DownActivUp]:
            convs.append(submodule.perf_stride)
        elif len(list(submodule.children())) != 0:
            convs.append(self._get_perforation(submodule))
    return flatten_list(convs)


def _set_perforation(self, perfs, part=None, start_n=0, verbose=False):

    if part is None:
        if verbose:
            print("Setting perforation mode to ", perfs)
        part = self
        start_n = [0]
    if type(perfs) == tuple:
        perfs = [perfs] * len(flatten_list(self._get_perforation()))
    for submodule in part.children():
        if type(submodule) in [PerforatedConv2d, DownActivUp]:
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
            if type(c) in [PerforatedConv2d, DownActivUp]:
                c.recompute = True
            else:
                recomp(c)

    recomp(self)

    def find_dev(net):
        for c in net.children():
            if list(c.children()) == [] and hasattr(c, "weight"):
                return c.weight.device
            else:
                if hasattr(c, "weight"):
                    return c.weight.device
                return find_dev(c)
        if list(net.children()) == [] and hasattr(net, "weight"):
            return net.weight.device
        else:
            if hasattr(net, "weight"):
                return net.weight.device
            print(net, flush=True)
            raise AttributeError("No device found?")

    if self.training:
        # SPECIFIČNO DAU PROBLEM
        self.eval()
        self(torch.zeros(self.in_size, device=find_dev(self), requires_grad=False))
        self.train()
    else:
        self(torch.zeros(self.in_size, device=find_dev(self)))
    return self


def _get_n_calc(self, part=None):
    if part is None:
        part = self
    convs = []
    for submodule in part.children():
        if type(submodule) in [PerforatedConv2d, DownActivUp]:
            submodule._get_calculations()
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


def perforate_net_perfconv(net, from_class=torch.nn.Conv2d, perforation_mode=(2, 2), pretrained=True,
                           in_size=None):
    if in_size is None:
        in_size = (2,3,5,5)
    print("perforating network to ", perforation_mode, " perf")
    if len(in_size) == 2:
        in_size = (2, 3, in_size[0], in_size[1])
    setattr(net, "in_size", in_size)
    setattr(net, "is_perf", True)
    replace_module_perfconv(net, from_class=from_class, perforation_mode=perforation_mode, pretrained=pretrained)
    add_functs(net)
    print(net.in_size)
    net._reset()


def perforate_net_downActivUp(net, in_size=None, from_class=torch.nn.Conv2d, perforation_mode=(2, 2), pretrained=True,
                              verbose=False):
    if in_size is None:
        in_size = (2,3,5,5)
    if verbose:
        print("perforating network to ", perforation_mode, " DAU")
    if len(in_size) == 2:
        in_size = (2, 3, in_size[0], in_size[1])
    setattr(net, "in_size", in_size)
    setattr(net, "is_perf", True)
    replace_module_downActivUp(net, from_class=from_class, perforation_mode=perforation_mode, pretrained=pretrained,
                               verbose=verbose)
    add_functs(net)
    #print(net.in_size)
    net._reset()

class layerWrapper(torch.nn.Module):
    def __init__(self, layer):
        super(layerWrapper, self).__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x)
