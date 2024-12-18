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


def get_multadds(net, mode="dau", perf_mode=(1,1), verbose=False):
    if mode == "dau":
        classname = "DownActivUp"
    elif mode == "perf":
        classname = "PerforatedConv2d"
    else:
        classname = "Conv2d"
    nams = names(net)
    if verbose:
        summary(net, input_size=ins, mode="train", depth=10,
                col_names=["input_size", "output_size", "mult_adds", "kernel_size", "num_params"], verbose=2)
    output = summary(net, input_size=ins, mode="train", depth=10,
                     col_names=["input_size", "output_size", "mult_adds", "kernel_size", "num_params"], verbose=0)
    ind = 0
    sum_mads = 0
    factorPerf = 1/(perf_mode[0]*perf_mode[1])
    for i, line in enumerate(str(output).split("\n")):
        if line.split("─")[-1].startswith(classname):
            if verbose:
                print("adding", line.split("─"))
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
            standin = None
            tempout = None
            try:
                original = nams[ind]
                ind += 1
                try:
                    stride = (original.stride[0] * original.perf_stride[0], original.stride[1] * original.perf_stride[1])
                    is_bias = original.is_bias
                except:
                    stride = original.stride
                    is_bias = True

                standin = torch.nn.Conv2d(original.in_channels, original.out_channels, original.kernel_size,
                                         stride, original.padding, original.dilation, original.groups,
                                          is_bias, device=original.weight.device)
                tempout = summary(standin, input_size=in_s, mode="train", col_names=["mult_adds"], verbose=0)
                if verbose:
                    print(standin)
                    print(tempout)

                n_multadds = int(str(tempout).split("\n")[3].split()[-1].replace(",", ""))
            except:
                try:
                    n_multadds = int([x for x in str(output).split("\n")[i].split(" ") if len(x) > 0][-3].replace(",", ""))
                except:
                    try:
                        n_multadds = int(
                            [x for x in str(output).split("\n")[i].split(" ") if len(x) > 0][-4].replace(",", ""))
                    except:
                        if verbose:
                            print("line failed", line, "input size:", things[-5])

                        raise ValueError(line, things)
            #now for upscale
            if mode=="dau" or mode == "perf":
                n_multadds += (1-factorPerf) * out_s[0]*out_s[1]*out_s[2]*out_s[3] * 2 #interpolation: 2x output size for all except corect nums

                if mode == "dau":
                    # n multadds from following layers that are also reduced in dau - we estimate 1 layer reduced
                    n_multadds -= (out_s[0] * out_s[1] * out_s[2] * out_s[3]) * (1-factorPerf)


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
            except:
                try:
                    sum_mads += int(sp[-3].replace(",", ""))
                except:
                    try:
                        sum_mads += int(sp[-4].replace(",", ""))
                    except:
                        pass
    return int(sum_mads)

if __name__ == "__main__":
    ins = (2, 3, 128, 128)


    def names(part):
        convs = []

        #convs.append(submodule._get_name())
        for submodule in part.children():
            if type(submodule) in [dauLayer, perfLayer]:
                convs.append(submodule)
            elif len(list(submodule.children())) != 0:
                convs.append(names(submodule))
        return flatten_list(convs)

    from agriadapt.segmentation.models.UNet import UNet as Agri
    from Architectures.UnetCustom import UNet as Custom

    #net = Custom(2)
    #DAU(net, in_size=ins)
    #print(get_multadds(net, mode="dau", perf_mode=(2,2), verbose=True))
    #quit()
    with open("UnetResults.txt", "w") as file:
        for netF in [Custom, Agri]:
            data = {}
            for pf in [DAU, perf]:
                mode = "perf" if pf == perf else "dau"
                print(netF, pf)
                net = netF(2)
                baseline = 0
                baseline = get_multadds(net, mode="none", verbose=False)
                print(baseline)
                data["None" + mode] = 100

                #quit()
                pf(net, in_size=ins)
                #DAU(net, in_size=ins)
                #print(get_multadds(net, mode="dau", perf_mode=(2,2), verbose=True))
                #quit()
                MAs = {}
                for i, j in [(3,3),(2,3),(3,2),(3,1),(1,3),(2,2),(2,1),(1,2),(1,1)]:
                    net._set_perforation((i, j))._reset()
                    MAs[(i,j)] = get_multadds(net, mode=mode, perf_mode=(i, j))
                #print(baseline)
                #print(MAs)
                coeffs = [0, 0.20029760897159576, 0.32178425043821335, 0.44327089190483093,0.5378847792744637, 0.6407210007309914, 0.7435572221875191, 0.8306062072515488, 0.9176551923155785, 1]
                ratios = np.diff(coeffs)
                data["(2, 2)_" + mode] = int(np.round((baseline/MAs[(2,2)])*100))
                data["(3, 3)_" + mode] = int(np.round((baseline/MAs[(3,3)])*100))
                data["2 by 2 equivalent_" + mode] = int(np.round((baseline/sum([MAs[x] * y for x, y in zip([(3,3),(2,3),(3,2),(3,1),(1,3),(2,2),(2,1),(1,2),(1,1)], ratios)]))*100))
                data["uniform random_" + mode] = int(np.round((baseline/(sum([MAs[x] for x in MAs])/len(MAs)))*100))
            print("\n".join([str((x, data[x])) for x in sorted([x for x in data])]))
    #if rn < 0.20029760897159576:
    #    perfs[i] = 3, 3
    #elif rn < 0.32178425043821335:
    #    perfs[i] = 2, 3
    #elif rn < 0.44327089190483093:
    #    perfs[i] = 3, 2
    ## elif rn < 0.5378847792744637: #ultra bad memory layout, skipping - 3,1 is the same time complexity
    ##    perfs[i] = 3,1
    #elif rn < 0.6407210007309914:
    #    perfs[i] = 1, 3
    #elif rn < 0.7435572221875191:
    #    perfs[i] = 2, 2
    #elif rn < 0.8306062072515488:
    #    perfs[i] = 1, 2
    #elif rn < 0.9176551923155785:
    #    perfs[i] = 2, 1
    #else:
    #    perfs[i] = 1, 1
    #print(get_multadds(net, mode="dau"))
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
