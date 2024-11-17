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

    for dev in ["cuda", "cpu"]:
        net.train()
        net.to(dev)
        results = {}
        train_accs = []
        losses = []
        loss_fn = loss_fn.to(dev)
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ], profile_memory=True,
        ) as p:
            for i, (batch, classes) in enumerate(data_loader):
                if vary_perf is not None and n_conv > 0 and type(perforation_mode[0]) == str:
                    perfs = get_perfs(perforation_mode[0], n_conv)
                    net._set_perforation(perfs)
                    # net._reset()

                batch = batch.to(dev)
                classes = classes.to(dev)
                pred = net(batch)
                if pred.device != classes.device:
                    loss = loss_fn(pred.cpu(), classes.cpu())
                else:
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
                    calculate_segmentation_metrics(classes, pred, run_name, metrics, dev, results)
                    acc = torch.mean(torch.tensor(results[f"{run_name}/iou/weeds"]))
                    train_accs.append(acc)
                break

        with open(f"./{prefix}/{curr_file}_{dev}.txt", "w") as ff:
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1), file=ff)
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))


for version in architectures:  # classigication, segmetnationg
    for dataset in version[1]:

        loss_fn = torch.nn.CrossEntropyLoss()
        if dataset == "agri":
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9])).to(device)

            lr = 0.5
            max_epochs = 300
            batch_size = 32
        elif dataset == "ucihar":
            max_epochs = 100
            batch_size = 32
            lr = 0.01
        else:
            max_epochs = 200
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

                        prefix = "allTests_last"
                        if not os.path.exists(f"./{prefix}/cpu"):
                            os.makedirs(f"./{prefix}/cpu")
                        prefix = prefix + "/cpu"
                        name = f"{modelname}_{dataset}_{img}_{perforation}_{perf_type}"
                        curr_file = f"{name}"
                        if not os.path.exists(f"./{prefix}/profiling/{curr_file}_cuda.txt"):
                            print("RUNNING PROFILING...")


                            pref = f"{prefix}/profiling"

                            if not os.path.exists(pref):
                                os.makedirs(f"./{pref}")
                            profile_net(net, op=op, data_loader=train_loader, n_conv=n_conv, vary_perf=vary_perf,
                                        perforation_mode=perf, run_name=curr_file, prefix=pref, loss_fn=loss_fn)

