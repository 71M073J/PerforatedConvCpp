import os.path
import random
import time
import copy
from typing import Union
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models.resnet
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score
# from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v2, mobilenet_v3_small
import torchvision.transforms.v2 as transforms
from agriadapt.segmentation.data.data import ImageDataset as AgriDataset
from torch.distributions import Categorical
from contextlib import ExitStack
from torchinfo import summary
from pytorch_cinic.dataset import CINIC10
from ucihar import UciHAR
# from Architectures.PerforatedConv2d import PerforatedConv2d
# from Architectures.mobilenetv2 import MobileNetV2
from perforateCustomNet import perforate_net_perfconv as perfPerf
from perforateCustomNet import perforate_net_downActivUp as perfDAU
# from Architectures.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small, MobileNetV3
# from Architectures.resnet import resnet152, resnet18, ResNet
from Architectures.UnetCustom import UNet as UNetCustom
from agriadapt.dl_scripts.UNet import UNet
from torch import argmax, where, cat, stack
import agriadapt.segmentation.settings as settings
import agriadapt.segmentation.data.data as dd
from benchmarking import benchmark, get_datasets, get_perfs, calculate_segmentation_metrics


metrics = {
    "iou": BinaryJaccardIndex,
    "precision": BinaryPrecision,
    "recall": BinaryRecall,
    "f1score": BinaryF1Score,
}
seed = 123
g = torch.Generator(device="cpu")
g.manual_seed(123)
torch.manual_seed(123)
np.random.seed(123)
num_workers = 1


device = "cpu"
architectures = [
    [[(resnet18, "resnet18"), (mobilenet_v2, "mobnetv2"), (mobilenet_v3_small, "mobnetv3s")],
     ["cifar", "ucihar"], [32]],
    # "cinic" takes too long to run, ~45sec per epoch compared to ~9 for cifar ,so it would be about 2 hour training per config, maybe later
    [[(UNet, "unet_agri"), (UNetCustom, "unet_custom")], ["agri"], [128, 256, 512]]
]

eval_modes = []

def profile_net(net, op, data_loader, vary_perf, n_conv, perforation_mode, run_name, prefix, loss_fn):

    for device in ["cuda", "cpu"]:
        net.train()
        net.to(device)
        results = {}
        train_accs = []
        losses = []
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
        ) as p:
            for i, (batch, classes) in enumerate(data_loader):
                if vary_perf is not None and n_conv > 0 and type(perforation_mode[0]) == str:
                    perfs = get_perfs(perforation_mode[0], n_conv)
                    net._set_perforation(perfs)
                    # net._reset()

                batch = batch.to(device)
                classes = classes.to(device)
                pred = net(batch)
                loss = loss_fn(pred, classes)
                loss.backward()
                losses.append(loss.item())

                op.step()
                op.zero_grad()
                if type(data_loader.dataset) in [torchvision.datasets.CIFAR10, CINIC10, UciHAR]:
                    # print("Should be here")
                    acc = (F.softmax(pred.detach(), dim=1).argmax(dim=1) == classes).cpu()
                    train_accs.append(torch.sum(acc) / batch_size)
                else:
                    calculate_segmentation_metrics(classes, pred, run_name, metrics, device, results)
                    acc = torch.mean(torch.tensor(results[f"{run_name}/iou/weeds"]))
                    train_accs.append(acc)
                break

        with open(f"./{prefix}/{curr_file}_{device}.txt", "w") as ff:
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1), file=ff)
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))


for version in architectures:  # classigication, segmetnationg
    for dataset in version[1]:

        loss_fn = torch.nn.CrossEntropyLoss()
        if dataset == "agri":
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9])).to(device)
            lr = 0.01
            max_epochs = 10
            batch_size = 32
        elif dataset == "ucihar":
            max_epochs = 10
            batch_size = 32
            lr = 0.01
        else:
            max_epochs = 10
            batch_size = 32
            lr = 0.01

        for model, modelname in version[0]:
            for img in version[2]:
                # model = UNet
                # img = 128
                # dataset = "agri"
                img_res = (img, img)
                in_size = (2, 3, img, img)
                if img == 128:
                    batch_size = 32
                elif img == 256:
                    batch_size = 16
                elif img == 512:
                    batch_size = 4

                alreadyNoPerf = False
                for perforation in (None, 2, 3, "random", "2by2_equivalent"):
                    perf = (perforation, perforation)
                    if perforation is None:
                        if alreadyNoPerf:
                            continue
                        else:
                            alreadyNoPerf = True

                    for perf_type in ["perf", "dau"]:
                        if perforation is None:
                            if perf_type == "dau":
                                continue
                            else:
                                perf_type = None
                        if not os.path.exists("./allTests/cpu"):
                            os.makedirs("./allTests/cpu")
                        prefix = "allTests/cpu"
                        name = f"{modelname}_{dataset}_{img}_{perforation}_{perf_type}"
                        curr_file = f"{name}"
                        if not os.path.exists(f"./allTests/profiling/{curr_file}_cuda.txt"):
                            print("RUNNING PROFILING...")
                            vary_perf = None
                            if type(perforation) == str:
                                vary_perf = True
                            else:
                                vary_perf = None
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
                                else:  # it is a string
                                    if "dau" in perf_type.lower():
                                        perfDAU(net, in_size=in_size, perforation_mode=(2, 2),
                                                pretrained=pretrained)
                                    elif "perf" in perf_type.lower():
                                        perfPerf(net, in_size=in_size, perforation_mode=(2, 2),
                                                 pretrained=pretrained)
                            else:
                                print("Perforating base net for noperf training...")
                                perfPerf(net, in_size=in_size, perforation_mode=(2, 2), pretrained=pretrained)
                                net._set_perforation((1, 1))
                            pref = "allTests/profiling"
                            n_conv = len(net._get_n_calc())
                            if not os.path.exists(pref):
                                os.makedirs(f"./{pref}")
                            op = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0005)
                            train_loader, valid_loader, test_loader = get_datasets(dataset, batch_size, True,
                                                                                   image_resolution=img_res)
                            profile_net(net, op=op, data_loader=train_loader, n_conv=n_conv, vary_perf=vary_perf,
                                        perforation_mode=perf, run_name=curr_file, prefix=pref, loss_fn=loss_fn)
                        if os.path.exists(f"./{prefix}/{curr_file}_best.txt"):
                            with open(f"./{prefix}/{curr_file}_best.txt", "r") as pread:
                                try:
                                    l = float(pread.readline().split("Validation acc (None):")[1].split("'")[0])

                                    if "resnet" not in modelname and l < 0.15 and "unet" not in modelname and perf_type != "dau":
                                        # not learning, not resnet
                                        print(f"RE-running run {curr_file}")

                                    else:
                                        print("file for", curr_file, "already exists, skipping...")
                                        continue
                                except:
                                    pass


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
                            else:  # it is a string
                                if "dau" in perf_type.lower():
                                    perfDAU(net, in_size=in_size, perforation_mode=(2, 2),
                                            pretrained=pretrained)
                                elif "perf" in perf_type.lower():
                                    perfPerf(net, in_size=in_size, perforation_mode=(2, 2),
                                             pretrained=pretrained)
                        else:
                            print("Perforating base net for noperf training...")
                            perfPerf(net, in_size=in_size, perforation_mode=(2, 2), pretrained=pretrained)
                            net._set_perforation((1, 1))
                        # if (dataset == "cifar" and perf_type == "dau") or mo:
                        #    lr /= 10
                        print("net:", modelname)
                        print("Dataset:", dataset)
                        print(max_epochs, "epochs")
                        print("perforation mode", perf)
                        print("perforation type:", perf_type)
                        print("batch_size:", batch_size)
                        print("loss fn", loss_fn)
                        print("eval modes", eval_modes)
                        print("Learning rate:", lr)
                        print("run name:", curr_file)
                        # continue
                        net._reset()
                        net.to(device)

                        # TODO make function to eval SPEED, MEMORY, FLOPS of each network, that is called if
                        # "_best" file already exists so we can do stuff on already trained tests

                        # TODO: separate scripts for making images and scripts for training - why tf is one dependent on the other
                        # TODO: aaaaaaaaa

                        op = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0005)
                        train_loader, valid_loader, test_loader = get_datasets(dataset, batch_size, True,
                                                                               image_resolution=img_res)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=max_epochs)

                        run_name = f"{curr_file}"
                        if not os.path.exists(f"./{prefix}/"):
                            os.mkdir(f"./{prefix}")
                        if not os.path.exists(f"./{prefix}/imgs"):
                            os.mkdir(f"./{prefix}/imgs")
                        print("starting run:", curr_file)
                        with open(f"./{prefix}/{curr_file}.txt", "w") as f:
                            best_out, confs, metrics = benchmark(net, op, scheduler, train_loader=train_loader,
                                                                 valid_loader=valid_loader, test_loader=test_loader,
                                                                 max_epochs=max_epochs, device=device,
                                                                 perforation_mode=perf,
                                                                 run_name=run_name, batch_size=batch_size,
                                                                 loss_function=loss_fn, prefix=prefix,
                                                                 eval_modes=eval_modes, in_size=in_size,
                                                                 dataset=dataset,
                                                                 perforation_type=perf_type, file=f, summarise=False)


                        with open(f"./{prefix}/{curr_file}_best.txt", "w") as ff:
                            print(best_out, file=ff)

