import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from perforateCustomNet import perforate_net_perfconv, perforate_net_downActivUp
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):

        return self.conv1(x)


if __name__ == "__main__":
    #If this doesnt work, we run the entire run for one network on RPi
    for perforation in [None, "DAU", "perf"]:
        net = SimpleNet()
        if perforation == "DAU":
            perforate_net_downActivUp(net, in_size=(2,3,9,9))
        elif perforation == "perf":
            perforate_net_perfconv(net, in_size=(2,3,9,9))
        net.conv1.weight = torch.nn.Parameter(torch.ones_like(net.conv1.weight, dtype=torch.float) / (3 * 2 * 3 * 3))
        net.conv1.bias = torch.nn.Parameter(torch.ones_like(net.conv1.bias, dtype=torch.float) / (3*3))
        data = torch.ones((2, 3, 9, 9), dtype=torch.float)
        data[:, 0, ::2, :] = 1
        data[:, 0, 1::2, :] = 2
        data[:, 0, :, ::2] = 2
        data[:, 0, :, 1::2] = 3

        data[:, 1, ::2, :] = 0
        data[:, 1, 1::2, :] = 3
        data[:, 1, :, ::2] = 1
        data[:, 1, :, 1::2] = 2

        data[:, 2, ::2, :] = -1
        data[:, 2, 1::2, :] = -2
        data[:, 2, :, ::2] = -2
        data[:, 2, :, 1::2] = -3

        data.requires_grad = True

        dataCompare = torch.ones((2, 3, 9, 9), dtype=torch.float)
        dataCompare[:, 0, ::2, :] = 0.8
        dataCompare[:, 0, 1::2, :] = 0.2
        dataCompare[:, 0, :, ::2] = -0.4
        dataCompare[:, 0, :, 1::2] = 0.3


        dataCompare[:, 1, ::2, :] = 1
        dataCompare[:, 1, 1::2, :] = -0.2
        dataCompare[:, 1, :, ::2] = -0.4
        dataCompare[:, 1, :, 1::2] = 0.5

        dataCompare[:, 2, ::2, :] = 1
        dataCompare[:, 2, 1::2, :] = 0.2
        dataCompare[:, 2, :, ::2] = 0.4
        dataCompare[:, 2, :, 1::2] = 0.5

        dataCompare[:, 1, :6, :5] += 0.5
        #loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.CrossEntropyLoss()

        res = net(data)
        suma = loss_fn(res, dataCompare)
        suma.backward()
        verbose = False
        if verbose:
            print(res)
            print("\n---\n")
            print(suma)
            print(net.conv1.weight.grad)
            print(data.grad * 1000)
            print(net.conv1.weight.shape)

        base = f"./RpiComparison/{perforation}/"
        if not os.path.exists(base):
            os.makedirs(base)

        if os.path.exists(f"{base}weightGrad.bin"):
            weightGradCheck = torch.tensor(np.fromfile(f"{base}weightGrad.bin", dtype=np.float32)).view(net.conv1.weight.grad.shape)
            dataGradCheck = torch.tensor(np.fromfile(f"{base}dataGrad.bin", dtype=np.float32)).view(data.grad.shape)
            resCheck = torch.tensor(np.fromfile(f"{base}res.bin", dtype=np.float32)).view(res.shape)

            print("Checking weightgrad similarity...")
            print(weightGradCheck - net.conv1.weight.grad)
            print((weightGradCheck - net.conv1.weight.grad).mean(), "mean weightgrad diff, Checking datagrad similarity...")
            print(dataGradCheck - data.grad)
            print((dataGradCheck - data.grad).mean(), "mean datagrad diff, Checking result similarity...")
            print(resCheck - res)
            print((resCheck - res).mean(), "mean res diff")
        else:
            print("No existing compare files found, skipping...")

        if not os.path.exists(f"{base}weightGrad.bin"):
            net.conv1.weight.grad.detach().numpy().tofile(f"{base}weightGrad.bin")
            data.grad.detach().numpy().tofile(f"{base}dataGrad.bin")
            res.detach().numpy().tofile(f"{base}res.bin")





        fig, ax = plt.subplots(1, 5)
        ax[0].imshow(res.detach()[0].transpose(0, 1).transpose(1, 2))
        #ax[1].imshow(res.grad.detach()[0].transpose(0, 1).transpose(1, 2))
        ax[1].imshow((1 + net.conv1.weight.grad.detach()[0])/2)
        ax[2].imshow((1 + net.conv1.weight.grad.detach()[1])/2)
        ax[3].imshow((1 + net.conv1.weight.grad.detach()[2])/2)
        ax[-1].imshow(data.grad.detach()[0].transpose(0, 1).transpose(1, 2)*1000 + 0.2)
        plt.savefig(f"{base}CompareGrads.png")
        plt.show()

