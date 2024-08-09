import time

#from Architectures.resnet import resnet18
#from Architectures.mobilenetv2 import MobileNetV2
#from Architectures.mobilenetv3 import mobilenet_v3_small
#from Architectures.PerforatedConv2d import PerforatedConv2d
from conv import PerforatedConv2d
import torch
#device = "cuda:0"
#perforation_mode = (3,1)
#device = "cpu"
#a = PerforatedConv2d(256,64,3, device=device, perforation_mode=(2,2))
for device in ["cuda:0", "cpu"]:
    print("Pytorch testing")
    a = torch.nn.Conv2d(64, 64, 3, device=device)
    b = torch.nn.Conv2d(64, 64, 3, device=device)
    c = torch.nn.Conv2d(64, 64, 3, device=device)
    d = torch.nn.Conv2d(64, 64, 3, device=device)

    h = torch.rand((32, 64, 64, 64), device=device)
    output = a(h)
    print("Loop starting for torch impl", flush=True)
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(200):
        # print("adsda")
        output = d(c(b(a(h))))
        # output.sum().backward()
    # del output
    torch.cuda.synchronize()
    print("Forward time:", (time.time() - t1)/10, "seconds")

    h = output.sum()
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(200):
        # print("adsda")
        # output = a(h)
        h.backward(retain_graph=True)

    torch.cuda.synchronize()
    print("backward time:", (time.time() - t1)/10, "seconds")
    for x in range(1, 4):
        for y in range(1, 4):
            perforation_mode = (x, y)
            print("\nTesting perf", perforation_mode, "...---------------------------------\n")



            a = PerforatedConv2d(64,64,3, device=device, perforation_mode=perforation_mode)
            b = PerforatedConv2d(64, 64,3, device=device, perforation_mode=perforation_mode)
            c = PerforatedConv2d(64, 64,3, device=device, perforation_mode=perforation_mode)
            d = PerforatedConv2d(64, 64,3, device=device, perforation_mode=perforation_mode)

            h = torch.rand((32, 64, 64, 64), device=device)
            output = a(h)
            print("Loop starting for my impl", flush=True)
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(20 if device.startswith("cp") else 20):
                #print("adsda")
                output = d(c(b(a(h))))
                #output.sum().backward()
            #del output
            torch.cuda.synchronize()
            print("Forward time:", time.time() - t1, "seconds")

            h = output.sum()
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(20 if device.startswith("cp") else 20):
                #print("adsda")
                #output = a(h)
                h.backward(retain_graph=True)

            torch.cuda.synchronize()
            print("backward time:", time.time() - t1, "seconds")
quit()
