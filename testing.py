import time

#from Architectures.resnet import resnet18
#from Architectures.mobilenetv2 import MobileNetV2
#from Architectures.mobilenetv3 import mobilenet_v3_small
#from Architectures.PerforatedConv2d import PerforatedConv2d
from conv import PerforatedConv2d
import torch
device = "cuda:0"
device = "cpu"
#a = PerforatedConv2d(256,64,3, device=device, perforation_mode=(2,2))
a = torch.nn.Conv2d(256,64,3, device=device)
#b = nn.Conv2d(32,32,3, padding=2, padding_mode="zeros", device="cuda:0")
#for i in range(100):
#    a(tt)
#    b(tt)
h = torch.rand((32, 256, 64, 64), device=device)
output = a(h)
print("Loop starting", flush=True)
t1 = time.time()
for i in range(20):
    #print("adsda")
    output = a(h)
    output.sum().backward()
print(time.time() - t1, "seconds")
quit()
