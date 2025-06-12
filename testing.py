import time
import torch
from perforateCustomNet import perforate_net_perfconv, perforate_net_downActivUp


if __name__ == "__main__":
    print("hhhh")
    from Architectures.UnetCustom import UNet
    net = UNet(2)
    net.cuda()
    #perforate_net_perfconv(net, in_size=(2,3,128,128))._set_perforation((1,1))._reset()
    op = torch.optim.SGD(net.parameters(), lr=0.001)
    lossfn = torch.nn.CrossEntropyLoss()
    vals = torch.rand(4, 3, 128,128, device="cuda")
    res = torch.rand(4, 2, 128, 128, device="cuda")
    for i in range(10):
        for i in range(10):
            r = net(vals)
            l = lossfn(r, res)
            l.backward()
            op.step()
            op.zero_grad()
            #print(i)
        t0 = time.time()
        for i in range(100):
            r = net(vals)
            l = lossfn(r, res)
            l.backward()
            op.step()
            op.zero_grad()
        t1 = time.time()
        print(t1 - t0, "seconds elapsed")

else:
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



                a = DAU(64,64,3, device=device, perforation_mode=perforation_mode)
                b = DAU(64, 64,3, device=device, perforation_mode=perforation_mode)
                c = DAU(64, 64,3, device=device, perforation_mode=perforation_mode)
                d = DAU(64, 64,3, device=device, perforation_mode=perforation_mode)

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
