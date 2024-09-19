import torch
import torchvision
from perforateCustomNet import perforate_net_perfconv, perforate_net_downActivUp
from benchmarking import get_datasets, benchmark
from Architectures.resnet import resnet18
import numpy as np
import random
device = "cuda:0"
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

if __name__ == "__main__":
    for i in range(2):
        if i == 0:
            net = torchvision.models.resnet18(num_classes=10)
            perforate_net_downActivUp(net, perforation_mode=(2,2), in_size=(32,32))
            #print(net)
            run_name = "function_perf"
        else:
            continue
            net = resnet18(num_classes=10, perforation_mode=(2,2))
            #print(net)
            run_name = "manual_perf"
        ######### brez tega dvojega je funcperf == manual
        perforation_mode = (2,2)
        perforation_type="dau"
        #########
        net.to(device)
        img_res = (32,32)
        in_size=(2,3,32,32)
        train_l, valid_l, _ = get_datasets("cifar", batch_size=32, augment=True, image_resolution=img_res)
        op = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=200)
        benchmark(net, op, scheduler=scheduler, max_epochs=5, train_loader=train_l, valid_loader=valid_l, device=device,
                  run_name=run_name, perforation_mode=perforation_mode, eval_modes=[None, (2,2), (3,3)], summarise=False,
                  in_size=in_size, perforation_type=perforation_type, batch_size=32)