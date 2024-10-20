# TODO torch environment to benchmark the model memory/cpu calls for all perf modes + comparisons between models
import torchvision.models
import json
# TODO 2, FLOPS per model
# from torchvision.models import resnet18, mobilenet_v2, mobilenet_v3_small
# from Architectures.UnetCustom import UNet as UNetCustom
# from agriadapt.dl_scripts.UNet import UNet
from perforateCustomNet import perforate_net_perfconv as perf
from perforateCustomNet import perforate_net_downActivUp as DAU
from Architectures.PerforatedConv2d import PerforatedConv2d as perfLayer
# TODO make function to eval SPEED, MEMORY, FLOPS of each network, that is called if
# "_best" file already exists so we can do stuff on already trained tests
# TODO: separate scripts for making images and scripts for training - why tf is one dependent on the other
# TODO: aaaaaaaaa
# net = mobilenet_v2(num_classes=6).to(device)
# dataset = "ucihar"
# perfDAU(net, in_size=in_size, perforation_mode=(2, 2),
#                     pretrained=pretrained)


# TODO replace in C code zeros initialised tensor with empty

import torch
from Architectures.PerforatedConv2d import DownActivUp as dauLayer
from torchinfo import summary
from perforateCustomNet import flatten_list
import numpy as np
def getTotalNumCalc(net, perf_compare=(2,2)):
    if not hasattr(net, "_set_perforation"): raise NotImplementedError(f"Net of type {type(net)} does not support perforation, try perforating the network first")
    calculationsBase = net._set_perforation((1, 1))._reset()._get_n_calc()
    calculationsOther = net._set_perforation(perf_compare)._reset()._get_n_calc()

    base_n = np.array([x[0] for x in calculationsBase])
    base_o = np.array([x[0] for x in calculationsOther])
    #n_diff = sum([int(calculationsBase[i][0] == calculationsOther[i][0]) for i in range(len(calculationsBase))])

    return f"{(base_n != base_o).sum()} out of {len(calculationsBase)} layers perforated, " \
           f"going from {base_n.sum()} to {base_o.sum()} operations ({(int(10000 * (1 - base_o.sum().astype(float)/base_n.sum().astype(float)))/100)}% improvement).\n" \
           f"(Counting only Conv layers)"


def get_multadds(net, mode="dau"):
    if mode == "dau":
        classname = "DownActivUp"
    elif mode == "perf":
        classname = "PerforatedConv2d"
    else:
        classname = ""
    nams = names(net)
    output = summary(net, input_size=ins, mode="train", depth=10,
                     col_names=["input_size", "output_size", "mult_adds", "kernel_size", "num_params"], verbose=0)
    ind = 0
    sum_mads = 0
    for i, line in enumerate(str(output).split("\n")):
        if line.split("─")[-1].startswith(classname):
            things = line.replace(", ", ",").split()
            if len(things) < 5:continue
            if line.startswith("L"): continue
            if line.startswith("T"): continue
            if line.startswith("E"): continue
            if line.startswith("P"): continue
            if line.startswith("I"): continue
            if line.startswith("F"): continue
            if line.startswith("N"): continue
            #print(things, nams)
            in_s = json.loads(things[-5])
            out_s = json.loads(things[-4])
            try:
                original = nams[ind]
                ind += 1
                standin = torch.nn.Conv2d(original.in_channels, original.out_channels, original.kernel_size,
                                          (original.stride[0]*original.perf_stride[0],original.stride[1]*original.perf_stride[1])
                                          , original.padding, original.dilation, original.groups,
                                          original.bias, device=original.weight.device)
                tempout = summary(standin, input_size=in_s, mode="train", col_names=["mult_adds"], verbose=0)

                n_multadds = int(str(tempout).split("\n")[3].split()[-1].replace(",", ""))
            except:
                try:
                    n_multadds = int([x for x in str(output).split("\n")[i].split(" ") if len(x) > 0][-3].replace(",", ""))
                except:
                    try:
                        n_multadds = int(
                            [x for x in str(output).split("\n")[i].split(" ") if len(x) > 0][-4].replace(",", ""))
                    except:continue

            #now for upscale
            if mode=="dau":
                sp = [x for x in line.split(" ") if len(x) > 0]
                try:
                    n = int(sp[-3].replace(",", ""))
                    n_multadds -= n
                    n_multadds += n / (original.perf_stride[0] * original.perf_stride[1])
                except:
                    for j in range(len([x for x in str(output).split("\n")[i+2:i+4] if x.startswith("│    " * (len([x for x in line.split(" ") if "│" in x]) + 1))])):
                        l = str(output).split("\n")[i + 2 + j]
                        sp = [x for x in l.split(" ") if len(x) > 0]
                        try:
                            n = int(sp[-3].replace(",", ""))
                            n_multadds -= n
                            n_multadds += n / (original.perf_stride[0] * original.perf_stride[1])
                        except:pass
                # n multadds from following layers that are also reduced in dau
                factor = original.perf_stride[0] < 3 and original.perf_stride[1] < 3 #if we do "optimised" interpolation
                if factor:
                    n_multadds += (out_s[0] * out_s[1] * out_s[2] * out_s[3])
                else:
                    n_multadds *= 2


            sum_mads += n_multadds


        else:
            if line.startswith("==="):continue
            if line.startswith("T"):continue
            if line.startswith("E"):continue
            if line.startswith("P"):continue
            if line.startswith("I"):continue
            if line.startswith("F"):continue
            if line.startswith("N"):continue
            #print("what the fuck",classname,line)
            #print("HELLO====???",line)
            sp = [x for x in line.split(" ") if len(x) > 0]
            #print(sp)
            #print(sp[-3].replace(",", ""))
            try:
                n = int(sp[-3].replace(",", ""))
                sum_mads += n
            except:pass
    return int(sum_mads)

if __name__ == "__main__":
    ins = (32, 3, 32, 32)
    net = torchvision.models.resnet18(num_classes=10)


    def names(part):
        convs = []

        #convs.append(submodule._get_name())
        for submodule in part.children():
            if type(submodule) in [dauLayer, perfLayer]:
                convs.append(submodule)
            elif len(list(submodule.children())) != 0:
                convs.append(names(submodule))
        return flatten_list(convs)


    DAU(net, in_size=ins, verbose=False)
    #perf(net, in_size=ins)
    print(get_multadds(net, mode="dau"))
    #summary(net, input_size=ins, mode="train", depth=10,
    #                 col_names=["input_size", "output_size", "mult_adds", "kernel_size", "num_params"])
    #print(output)
    # Numbers are Mult-adds
    # net_compare = torch.nn.Conv2d(3,32,3, stride=2)#6.45M/1.84/2.24
    # in_size = (32, 3, 32, 32)
    # net1 = perfLayer(3,32,3, perforation_mode=(2,2))#25.8M/7.37/7.77
    # net2 = dauLayer(3,32,3, perforation_mode=(2,2), up=False)#SHOULD BE #6.45M/1.84/2.24 - same as comparison
    # net3 = dauLayer(3,32,3, perforation_mode=(2,2), down=False)#just upscale
    # 1,2 - 0.5 * outsize tensor/0.25 insize tensor
    # 2,2 - 0.25 * outsize tensor/0.125 insize tensor
    # 2,3/3,3/+ - same as strided conv
    # net4 = dauLayer(3,32,3, perforation_mode=(2,2))#reduces all shape preserving ops by *(1/(prod(perf_modes)))
    # model = torchvision.models.resnet18(num_classes=10)
    # model2 = torchvision.models.resnet18(num_classes=10)
    # model3 = torchvision.models.resnet18(num_classes=10)
    # DAU(model2, in_size=in_size)
    # perf(model3, in_size=in_size)
    # summary(model, input_size=in_size)#1.18G
    # summary(model2, input_size=in_size)#0.22M
    # summary(model3, input_size=in_size)#1.18G
    # output = summary(net, input_size=, mode="train", col_names=["input_size", "output_size", "num_params","mult_adds"])



# speedtest actual networkov na cpu
