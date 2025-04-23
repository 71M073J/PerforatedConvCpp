# TODO torch environment to benchmark the model memory/cpu calls for all perf modes + comparisons between models
import time

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

def democode_for_memory_benchmarking():
    ### this is just demo code, as i did this manually, so i could avoid unknown memory allocations,
    ### which appear common when running this kind of benchmarking automatically
    import torch
    from Architectures.PerforatedConv2d import DownActivUp as dauLayer
    from torchinfo import summary
    from perforateCustomNet import flatten_list
    import numpy as np

    import torch
    from torchvision.models import mobilenet_v3_small
    from perforateCustomNet import perforate_net_perfconv as perf1
    from perforateCustomNet import perforate_net_downActivUp as perf2
    def calc():
        global network, train_data
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        pred = network(train_data)
        loss = loss_fn(pred, torch.ones_like(pred))
        loss.backward()

    def get_memory(msg=None):
        if msg is not None:
            print("memory for", msg, ":\n")
        print("current: ", torch.cuda.memory_allocated(), "Bytes")
        print("maximum usage since reset:", torch.cuda.max_memory_allocated(device=None), "Bytes")
        torch.cuda.reset_peak_memory_stats(device=None)

    train_data = torch.rand(2, 3, 32, 32, device="cuda")
    get_memory()
    network = mobilenet_v3_small(num_classes=10).cuda()
    calc()
    calc()
    get_memory()



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


def get_multadds(net, mode="dau", perf_mode=(1,1), verbose=False, ins=None, prevIOs=None):
    if mode == "dau":
        classname = "DownActivUp"
        cl = dauLayer
    elif mode == "perf":
        classname = "PerforatedConv2d"
        cl = perfLayer
    else:
        classname = "Conv2d"
    nams = names(net)
    if verbose:
        summary(net, input_size=ins, mode="train", depth=10,
                col_names=["input_size", "output_size", "kernel_size", "mult_adds", "num_params"], verbose=2)
    output = summary(net, input_size=ins, mode="train", depth=10,
                     col_names=["input_size", "output_size", "kernel_size", "mult_adds", "num_params"], verbose=0)

    #print(output)
    ind = 0
    insouts = []
    sum_mads = 0
    sum_MB = output.to_megabytes(output.total_output_bytes )# if classname == "Conv2d" else 0
    #+ output.total_input + output.total_param_bytes
    #print(perf_mode)
    #print(output)
    factorPerf = 1/(perf_mode[0]*perf_mode[1])
    for i, line in enumerate(str(output).split("\n")):
        if line.split("─")[-1].startswith(classname + ":"):#if conv2d-type layer
            if verbose:
                print("adding", line.split("─"))
            things = line.replace(", ", ",").split()

            in_s = json.loads(things[-5])
            out_s = json.loads(things[-4])
            insouts.append((in_s, out_s))
            if classname == "Conv2d": #for non-perforated networks
                n_multadds = int(things[-2].replace(",", ""))
            else:
                original = nams[ind]  # just daulayers or perflayers
                ind += 1
                #print("Strides:", original.stride, original.perf_stride)
                stride = (original.stride[0] * original.perf_stride[0], original.stride[1] * original.perf_stride[1])
                #is_bias = original.is_bias

                if out_s[-1] < original.perf_stride[0] or out_s[-2] < original.perf_stride[1]:
                    stride = original.stride

                #stand-in is just strided convolution
                #print("making a stand-in")
                standin = torch.nn.Conv2d(original.in_channels, original.out_channels, original.kernel_size,
                                         stride, original.padding, original.dilation, original.groups,
                                          original.is_bias, device=original.weight.device)

                tempout = summary(standin, input_size=prevIOs[ind-1][0], mode="train", col_names=["mult_adds"], verbose=0)

                n_multadds = int(str(tempout).split("\n")[3].split()[-1].replace(",", ""))
                sum_MB -= tempout.to_megabytes(tempout.total_output_bytes) #REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

                trueStandin = cl(original.in_channels, original.out_channels, original.kernel_size, original.stride,
                                 original.padding, original.dilation, original.groups, original.is_bias,
                                 device=original.weight.device, perf_stride=perf_mode)
                tempout2 = summary(trueStandin, input_size=prevIOs[ind-1][0], mode="train", verbose=0)
                sum_MB += tempout2.to_megabytes(tempout2.total_output_bytes)
                #Now for upscale

                n_multadds += (1-factorPerf) * out_s[0]*out_s[1]*out_s[2]*out_s[3] * 2 #interpolation: 2x output size for all except corect indices
                #print(n_multadds)
                if mode == "dau":
                    # n multadds from following layers that are also reduced in dau - we estimate 1 layer reduced
                    n_multadds -= (out_s[0] * out_s[1] * out_s[2] * out_s[3]) * (1-factorPerf) #relu/BN are unary operations per input, so this is actually accurate
                    sum_MB -= tempout2.to_megabytes(tempout.total_output_bytes)#*4 for bytes??


            sum_mads += n_multadds#

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
            things = line.replace(", ", ",").split()
            if things[0] == "Layer":continue
            if things[-2] != "--":
                sum_mads += int(things[-2].replace(",", ""))

            #print(sp[-3].replace(",", ""))

    return int(sum_mads), sum_MB, insouts


def get_mem_nums():
    from Architectures.UnetCustom import UNet as UNetCustom
    from agriadapt.segmentation.models.UNet import UNet
    from perforateCustomNet import perforate_net_perfconv as perfPerf
    from perforateCustomNet import perforate_net_downActivUp as perfDAU
    from torchvision.models import resnet18
    from torchvision.models import mobilenet_v2
    from torchvision.models import mobilenet_v3_small
    import gc
    device = "cuda"
    architectures = [

        [[(resnet18, "resnet18"), (mobilenet_v2, "mobnetv2"), (mobilenet_v3_small, "mobnetv3s")], ["cifar", "ucihar"],
         [32]],
        # "cinic" takes too long to run, ~45sec per epoch compared to ~9 for cifar ,so it would be about 2 hour training per config, maybe later
        [[(UNetCustom, "unet_custom"), (UNet, "unet_agri"), ], ["agri"], [128, 256, 512]],
    ]
    memory_usages_bytes = {}
    train_data = None
    from benchmarking import get_perfs
    for version in architectures:  # classigication, segmetnationg
        for dataset in version[1]:
            for model, modelname in version[0]:
                for img in version[2]:
                    in_size = (2, 3, img, img)
                    alreadyNoPerf = False
                    for perforation in (None, 2, 3, "random", "2by2_equivalent"):
                        perf = (perforation, perforation)
                        if perforation is None:
                            if alreadyNoPerf:
                                continue
                            else:
                                alreadyNoPerf = True

                        for perf_type in ["perf", "dau", ]:
                            if perforation is None:
                                if perf_type == "dau":
                                    continue
                                else:
                                    perf_type = None

                            #memfirst0 = torch.cuda.max_memory_allocated(device=None)
                            #print(memfirst0)
                            #assert memfirst0 == 0

                            if "agri" in dataset:
                                net = model(2).to(device)
                            else:
                                sz = 6 if dataset == "ucihar" else 10
                                net = model(num_classes=sz).to(device)

                            pretrained = True  # keep default network init
                            if perf[0] is not None:  # do we want to perforate? # Is the net not already perforated?
                                if type(perf[0]) != str:  # is perf mode not a string
                                    if "dau" in perf_type.lower():
                                        perfDAU(net, in_size=in_size, perforation_mode=perf,
                                                pretrained=pretrained)
                                    elif "perf" in perf_type.lower():
                                        perfPerf(net, in_size=in_size, perforation_mode=perf,
                                             pretrained=pretrained)
                                    net._set_perforation(perf)
                                else:  # it is a string

                                    if "dau" in perf_type.lower():
                                        perfDAU(net, in_size=in_size, perforation_mode=(2, 2),
                                                pretrained=pretrained)
                                    elif "perf" in perf_type.lower():
                                        perfPerf(net, in_size=in_size, perforation_mode=(2, 2),
                                                 pretrained=pretrained)

                                    n_conv = len(net._get_perforation())
                                    perfs = get_perfs(perf[0], n_conv)
                                    net._set_perforation(perfs)
                                    del perfs, n_conv
                            else:
                                print("Perforating base net for noperf training...")
                                perfPerf(net, in_size=in_size, perforation_mode=(2, 2), pretrained=pretrained)
                                net._set_perforation((1, 1))

                            # continue
                            if hasattr(net, "_reset"):
                                net._reset()
                            net = net.to(device)
                            #del net
                            #time.sleep(2)
                            #gc.collect()

                            torch.cuda.reset_peak_memory_stats(device=None)
                            memfirst = torch.cuda.memory_allocated(device=None)
                            print(memfirst)
                            #del train_data
                            loss_fn = torch.nn.CrossEntropyLoss()
                            train_data = torch.rand(in_size, device=device)
                            torch.cuda.reset_peak_memory_stats(device=None)
                            memprev = torch.cuda.max_memory_allocated(device=None)
                            infer = net(train_data)
                            loss = loss_fn(infer, torch.ones_like(infer))
                            loss.backward()
                            mem = torch.cuda.max_memory_allocated(device=None)
                            memory_usages_bytes[modelname + str(perf_type) + str(perforation)] = (mem - memprev, None if perf_type is None else (mem - memprev)/memory_usages_bytes[modelname + "NoneNone"][0])
                            del net, train_data, loss_fn, infer, loss
                            gc.collect()
                            #time.sleep(20)
                            #gc.collect()
                            torch.cuda.reset_peak_memory_stats(device=None)
                            memafter = torch.cuda.memory_allocated(device=None)
                            print(memafter)
                            #assert memafter == 0

    print("\n".join([str((x, memory_usages_bytes[x])) for x in memory_usages_bytes]))


if __name__ == "__main__":
    get_mem_nums() #TODO WEIRD BASELINE NUMBERS??
    quit()

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

    from torchvision.models import resnet18, mobilenet_v2, mobilenet_v3_small

    ins = None
    with open("UnetResults.txt", "w") as file:
        for netF, nn in [(mobilenet_v3_small, 32),(resnet18, 32), (mobilenet_v2, 32), (Custom, 128), (Agri, 128), (Custom, 256), (Agri, 256), (Custom, 512), (Agri, 512), ]:

            data = {}
            print(netF, nn)
            print(netF, nn, file=file)
            for pf in [perf, DAU, ]:
                mode = "perf" if pf == perf else "dau"
                if netF in [Custom, Agri]:
                    net = netF(2)
                    ins = (2, 3, nn, nn)
                else:
                    net = netF(num_classes=10)
                    ins = (2, 3, 32, 32)

                nams_baseline = names(net)
                #print("Getting baseline...")
                baseline, MBs_baseline, outp = get_multadds(net, mode="none", verbose=False, ins=ins)
                #print(baseline)
                data["None" + mode] = 100
                #print("perforating...")
                pf(net, in_size=ins)#Perforate
                #print("Getting perfs...")
                MAs = {}
                MBs = {}
                for i, j in [(3,3),(2,3),(3,2),(3,1),(1,3),(2,2),(2,1),(1,2),(1,1)]:
                    net._set_perforation((i, j))._reset()
                    MAs[(i,j)], MBs[(i,j)], _ = get_multadds(net, mode=mode, perf_mode=(i, j), ins=ins, prevIOs=outp)
                #print(baseline)
                #print(MAs)
                coeffs = [0, 0.20029760897159576, 0.32178425043821335, 0.44327089190483093,0.5378847792744637, 0.6407210007309914, 0.7435572221875191, 0.8306062072515488, 0.9176551923155785, 1]
                ratios = np.diff(coeffs)
                data["(2, 2)_" + mode] = int(np.round((MAs[(2,2)]/baseline)*100)),int(np.round((MBs[(2,2)]/MBs_baseline)*100))
                data["(3, 3)_" + mode] = int(np.round((MAs[(3,3)]/baseline)*100)),int(np.round((MBs[(3,3)]/MBs_baseline)*100))
                data["2 by 2 equivalent_" + mode] = int(np.round((sum([MAs[x] * y for x, y in zip([(3,3),(2,3),(3,2),(3,1),(1,3),(2,2),(2,1),(1,2),(1,1)], ratios)])/baseline)*100)),\
                    int(np.round((sum([MBs[x] * y for x, y in zip([(3,3),(2,3),(3,2),(3,1),(1,3),(2,2),(2,1),(1,2),(1,1)], ratios)])/MBs_baseline)*100))
                data["uniform random_" + mode] = int(np.round(((sum([MAs[x] for x in MAs])/baseline/len(MAs)))*100)),int(np.round(((sum([MBs[x] for x in MBs])/MBs_baseline/len(MBs)))*100))
            print("\n".join([str((x, data[x])) for x in sorted([x for x in data])]))
            print("\n".join([str((x, data[x])) for x in sorted([x for x in data])]), file=file)

