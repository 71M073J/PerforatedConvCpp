#TODO torch environment to benchmark the model memory/cpu calls for all perf modes + comparisons between models
import torchvision.models

#TODO 2, FLOPS per model
#from torchvision.models import resnet18, mobilenet_v2, mobilenet_v3_small
#from Architectures.UnetCustom import UNet as UNetCustom
#from agriadapt.dl_scripts.UNet import UNet
from perforateCustomNet import perforate_net_perfconv as perf
from perforateCustomNet import perforate_net_downActivUp as DAU
from Architectures.PerforatedConv2d import PerforatedConv2d as perfLayer
                            #TODO make function to eval SPEED, MEMORY, FLOPS of each network, that is called if
                            # "_best" file already exists so we can do stuff on already trained tests
                            #TODO: separate scripts for making images and scripts for training - why tf is one dependent on the other
                            #TODO: aaaaaaaaa
                            #net = mobilenet_v2(num_classes=6).to(device)
                            #dataset = "ucihar"
                            #perfDAU(net, in_size=in_size, perforation_mode=(2, 2),
                            #                     pretrained=pretrained)


#TODO replace in C code zeros initialised tensor with empty

import torch
from Architectures.PerforatedConv2d import DownActivUp as dauLayer
from torchinfo import summary

if __name__ == "__main__":
    #Numbers are Mult-adds
    net_compare = torch.nn.Conv2d(3,32,3, stride=2)#6.45M/1.84/2.24
    in_size = (32, 3, 32, 32)
    net1 = perfLayer(3,32,3, perforation_mode=(2,2))#25.8M/7.37/7.77
    net2 = dauLayer(3,32,3, perforation_mode=(2,2), up=False)#SHOULD BE #6.45M/1.84/2.24 - same as comparison
    net3 = dauLayer(3,32,3, perforation_mode=(2,2), down=False)#just upscale
    # 1,2 - 0.5 * outsize tensor/0.25 insize tensor
    # 2,2 - 0.25 * outsize tensor/0.125 insize tensor
    # 2,3/3,3/+ - same as strided conv
    net4 = dauLayer(3,32,3, perforation_mode=(2,2))#reduces all shape preserving ops by *(1/(prod(perf_modes)))
    model = torchvision.models.resnet18(num_classes=10)
    model2 = torchvision.models.resnet18(num_classes=10)
    model3 = torchvision.models.resnet18(num_classes=10)
    DAU(model2, in_size=in_size)
    perf(model3, in_size=in_size)
    summary(model, input_size=in_size)#1.18G
    summary(model2, input_size=in_size)#0.22M
    summary(model3, input_size=in_size)#1.18G
# speedtest actual networkov na cpu