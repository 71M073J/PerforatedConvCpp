import torch
from conv import PerforatedConv2d as Conv
import traceback


def test():
    if False:
        c = Conv(1, 1, 1, perf_stride=2)
        with torch.no_grad():
            c.weight = torch.nn.Parameter(torch.ones_like(c.weight))
        test1 = True
        if test1:
            print("error test")
            c(torch.arange(0, 1 * 6 * 8, 1, dtype=torch.float32).reshape(1, 1, 6, 8))
            try:
                print(6, 6)
                print(c(torch.arange(0, 1 * 6 * 8, 1, dtype=torch.float32).reshape(1, 1, 6, 8)))
            except:
                print(6, 6, "failed")

        c = Conv(1, 1, 3, perf_stride=(3, 2))
        c2 = torch.nn.Conv2d(1, 1, 3)
        with torch.no_grad():
            c.weight = torch.nn.Parameter(torch.ones_like(c.weight) / 9)
            c.bias = torch.nn.Parameter(torch.zeros_like(c.bias))

            c2.weight = torch.nn.Parameter(torch.ones_like(c2.weight) / 9)
            c2.bias = torch.nn.Parameter(torch.zeros_like(c2.bias))
        try_sym = False
        if try_sym:
            try:
                print(6, 6)
                print(c(torch.arange(0, 1 * 36, 1, dtype=torch.float32).reshape(1, 1, 6, 6)))
            except:
                print(6, 6, "failed")
        try:
            print(5, 8)
            print(c(torch.arange(0, 1 * 9 * 11, 1, dtype=torch.float32).reshape(1, 1, 9, 11)))
            print(c2(torch.arange(0, 1 * 9 * 11, 1, dtype=torch.float32).reshape(1, 1, 9, 11)))
        except Exception:
            print(traceback.format_exc())
        try:
            print(8, 5)
            print(c(torch.arange(0, 1 * 9 * 11, 1, dtype=torch.float32).reshape(1, 1, 11, 9)))
            print(c2(torch.arange(0, 1 * 9 * 11, 1, dtype=torch.float32).reshape(1, 1, 11, 9)))
        except Exception:
            print(traceback.format_exc())
        c = Conv(1, 1, 3, perf_stride=(2, 3))
        c2 = torch.nn.Conv2d(1, 1, 3)
        with torch.no_grad():
            c.weight = torch.nn.Parameter(torch.ones_like(c.weight) / 9)
            c.bias = torch.nn.Parameter(torch.zeros_like(c.bias))

            c2.weight = torch.nn.Parameter(torch.ones_like(c2.weight) / 9)
            c2.bias = torch.nn.Parameter(torch.zeros_like(c2.bias))
        try_sym = False
        if try_sym:
            try:
                print(6, 6)
                ot = c(torch.arange(0, 1 * 36, 1, dtype=torch.float32).reshape(1, 1, 6, 6))
                l = ot.sum()
                l.backward()
                print(ot)
            except:
                print(6, 6, "failed")
        try:
            print(5, 8)
            print(c(torch.arange(0, 1 * 9 * 11, 1, dtype=torch.float32).reshape(1, 1, 9, 11)))
            # print(c2(torch.arange(0, 1*9*11, 1, dtype=torch.float32).reshape(1,1,9,11)))
        except Exception:
            print(traceback.format_exc())
    print("STARTING BRUTE FORCE TEST")
    torch.set_printoptions(linewidth=1231231)
    for inx in [9, 10, 11, 12]:
        for iny in [9, 10, 11, 12]:
            for p1 in [1, 2, 3]:
                for p2 in [1, 2, 3]:
                    for pad in [0, 1]:
                        for bias in [True, False]:
                            print(inx, iny, p1, p2, pad)
                            try:
                                c = Conv(1, 1, 3, perf_stride=(p1, p2), padding=pad, silent=True, bias=bias)
                                c2 = torch.nn.Conv2d(1, 1, 3, padding=pad, bias=bias)
                                with torch.no_grad():
                                    c.weight = torch.nn.Parameter(torch.ones_like(c.weight))
                                    c2.weight = torch.nn.Parameter(torch.ones_like(c2.weight))
                                    if bias:
                                        c.bias = torch.nn.Parameter(torch.zeros_like(c.bias))
                                        c2.bias = torch.nn.Parameter(torch.zeros_like(c2.bias))
                                t1 = c(torch.arange(0, 1 * inx * iny, 1, dtype=torch.float32).reshape(1, 1, inx, iny))
                                l = t1.sum()
                                l.backward()
                                t2 = c2(torch.arange(0, 1 * inx * iny, 1, dtype=torch.float32).reshape(1, 1, inx, iny))
                                diff = t1 - t2
                                if diff[:, :, p1+1:p1 * 2+1, p2+1:p2 * 2+1].sum() > 0.1:
                                    print(diff[:, :, :p1 * 2, :p2 * 2].sum())
                                    print(diff,"\n", t1.long(), "\n",t2.long())
                                    print(inx, iny, p1, p2, pad)
                            except Exception:
                                print(traceback.format_exc())
                                print(inx, iny, p1, p2, pad)
                                quit()


if __name__ == "__main__":
    test()
