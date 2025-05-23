import os

import torch
from Architectures.PerforatedConv2d import PerforatedConv2d as Conv
import time
import numpy as np

c1 = Conv
c2 = torch.nn.Conv2d
ks = 3
cnt = 0
for device in ["cuda", "cpu"]:
    for stride in [2, 3]:
        for bs in [1, 2, 4, 16, 64]:
            for channels in [2, 4, 8, 32, 64, 128, 256, 512, 1024]:

                path = f"./speedtests/{device}/{ks}/{stride}/{bs}/{channels}"
                if not os.path.exists(path):
                    os.makedirs(path)
                #print(path)
                lines = []
                if os.path.exists(f"./speedtests/{device}/{ks}/{stride}/{bs}/{channels}/data.txt"):
                    continue
                for imgsz in [4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512]:
                    if bs * channels * imgsz * imgsz > 1e7:
                        continue
                    device = "cpu"
                    #device = "cuda"
                    print(device)
                    stride = 2
                    bs = 4
                    channels = 256
                    imgsz = 128
                    for back in [True, False]:
                        cnt += 1
                        #continue
                        n = bs * channels * imgsz * imgsz
                        print(n)

                        c22 = c2(channels, channels, ks).to(device)#torch
                        c33 = c1(channels, channels, ks, perf_stride=(stride, stride), strided_backward=back).to(device)

                        data = torch.rand((bs, channels, imgsz, imgsz), device=device)
                        if device != "cpu":
                            for i in range(100):
                                c22(data)
                                c33(data)

                        times2 = []
                        times3 = []
                        for i in range(100):
                            torch.cuda.synchronize()
                            t0 = time.time()
                            h2 = c22(data)#torch
                            l2 = h2.sum()
                            l2.backward()
                            torch.cuda.synchronize()
                            t1 = time.time()
                            times2.append(t1 - t0)#torch
                            torch.cuda.synchronize()
                            t03 = time.time()
                            h3 = c33(data)
                            l3 = h3.sum()
                            l3.backward()
                            torch.cuda.synchronize()
                            t3 = time.time()
                            times3.append(t3 - t03)
                        del c22, c33
                        m1 = np.median([times2]) * 1000 #torch
                        m2 = np.median([times3]) * 1000
                        # print(f"({device}) for {channels} channels, {bs} batch size {imgsz} image size, {ks} kernel, "
                        #      f"torch conv took {m1} ms")
                        # print(f"({device}) for {channels} channels, {bs} batch size {imgsz} image size, {ks} kernel, "
                        #      f"{stride}, {stride} perfconv took {m2} ms")
                        lines.append(f"Image size {imgsz} px: {m2} ms, Pytorch conv speed: {m1} ms\n")
                    print(lines)
                    quit()
                with open(f"./speedtests/{device}/{ks}/{stride}/{bs}/{channels}/data.txt", "w") as f:
                    f.writelines(lines)
print(cnt)