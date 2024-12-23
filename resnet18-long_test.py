import time

import torch
import torch.nn as nn
import random
import os
import torchvision.models
from pytorch_cinic.dataset import CINIC10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from main import test_net

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    g = torch.Generator(device='cuda').manual_seed(0)
    augment = True
    tf = [transforms.ToTensor(), ]
    tf_test = [transforms.ToTensor(), ]
    data = "cifar"
    dataset1, dataset2, dataset3 = None, None, None
    bs = 64
    if augment:
        tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
                                            transforms.RandomResizedCrop(size=32)
                                            ]),
                   transforms.RandomHorizontalFlip()])

    if data == "cifar":

        tf.extend([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tf_test.extend([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tf = transforms.Compose(tf)
        tf_test = transforms.Compose(tf_test)
        dataset1 = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=tf), batch_size=bs, shuffle=True, num_workers=4)

        dataset2 = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=tf_test), batch_size=bs, shuffle=False,
            num_workers=4)


    from Architectures.resnet import resnet18 as perfresnet
    from Architectures.mobilenetv3 import mobilenet_v3_small as perfmobilenetv3
    #from Architectures.mobilenetv3 import mobilenet_v3_large
    from Architectures.mobilenetv2 import mobilenet_v2 as perfmobilenetv2
    from torchvision.models import resnet18, mobilenet_v2, mobilenet_v3_small
    from perforateCustomNet import perforate_net_downActivUp as DAU
    for arch, name in [
                       (resnet18, "DAUresnet"), (mobilenet_v3_small, "DAUmobnetv3"), (mobilenet_v2, "DAUmobnetv2"),
                        (perfresnet, "perfresnet"), (perfmobilenetv3, "perfmobnetv3"), (perfmobilenetv2, "perfmobnetv2"),]:
        for perf in [(1,1),(2,2),(3,3),"random", "2by2_equivalent"]:
            vary_perf=None
            if type(perf) == str:
                perf = (1,1)
                vary_perf=perf
            eval_mode = [None, (1,1),(2,2),(3,3)]
            net = None
            op = None
            if name.startswith("DAU"):
                #eval_mode = [None]
                #if perf[0] != 2:
                #    continue
                net = arch(num_classes=10)
                DAU(net, (32, 32), perforation_mode=perf, pretrained=True)
                op = torch.optim.SGD(net.parameters(), momentum=0.9, lr=0.1, weight_decay=0.0005)
            else:
                #continue
                net = arch(num_classes=10, perforation_mode=perf, grad_conv=True)
                op = torch.optim.SGD(net.parameters(), momentum=0.9, lr=0.1, weight_decay=0.0005)
            # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(op, [100, 150, 175], gamma=0.1)
            #op = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
            lr_scheduler = None
            make_imgs = False
            prefix = "testFixPerf"
            epochs = 200
            if type(op) == torch.optim.SGD:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=epochs)
                if name.startswith("DAU") and False:
                    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(op, [torch.optim.lr_scheduler.LinearLR(op, start_factor=0.0001, total_iters=2),
                                      #torch.optim.lr_scheduler.LinearLR(op, start_factor=0.01, total_iters=1),
                                      #torch.optim.lr_scheduler.LinearLR(op, start_factor=0.01, total_iters=1),
                                                                              torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=epochs-2)], milestones=[2])
            else:
                epochs = 10
                eval_mode = [None]
                prefix = "adam_test"
                make_imgs = True
            #eval_mode=(2,2)
            rs = 0
            perfmode = str(perf[0])+"x"+str(perf[0]) if type(perf[0]) == int else perf
            curr_file = f"{name}_{perfmode}"
            if not os.path.exists(f"./{prefix}/"):
                os.mkdir(f"./{prefix}")
            print("starting run:", curr_file)
            if os.path.exists(f"./{prefix}/{curr_file}_best.txt"):
                print("file for", curr_file, "already exists, skipping...")
                continue
            with open(f"./{prefix}/{curr_file}.txt", "w") as f:
                t_0 = time.time()
                results = test_net(net, batch_size=bs, epochs=epochs, do_profiling=False, summarise=True, verbose=False,
                         make_imgs=make_imgs, plot_loss=False, vary_perf=vary_perf,
                         file=f, eval_mode=eval_mode,device="cuda",
                         run_name=curr_file, dataset=dataset1, dataset2=dataset2, dataset3=dataset3, op=op,
                         lr_scheduler=lr_scheduler, validate=False if data == "cifar" else True,
                                   grad_clip=10 if name.startswith("DAU") else None)
                print(results)
                t1 = time.time()
                print(t1 - t_0, "Seconds elapsed for network", curr_file)
                rs = [float(x) for x in results]
                with open(f"./{prefix}/{curr_file}_best.txt", "w") as ffs:
                    print(rs, file=ffs)
                    print(t1 - t_0, "Seconds elapsed for network", curr_file, file=ffs)




