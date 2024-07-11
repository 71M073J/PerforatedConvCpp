import torch
from conv import PerforatedConv2d as Conv
import traceback


def test():
    c = Conv(4, 2, 3, perf_stride=(2, 2), padding=0, bias=True,
                                                         stride=1, groups=1, strided_backward=True)

    #c.eval()
    h = c(torch.ones((2, 4, 19, 19)))
    h.sum().backward()
    #quit()
    print("STARTING BRUTE FORCE TEST")
    torch.set_printoptions(linewidth=1231231)
    for back_stride in [True, False]:
        for groups in [2,1]:
            for inx in [17, 18, 19]:
                for iny in [17, 18, 19]:
                    for p1 in [1, 2, 3]:
                        for p2 in [1, 2, 3]:
                            for pad in [0, 1]:
                                for bias in [True, False]:
                                    for ks in [7, 3]:
                                        for stride in [2, 1]:
                                            try:
                                                #TODO Try just eval modes nekaj pri 2x2 ga zjebe
                                                #TODO on teleport run python shell, make a resnet object and run it in eval2 mode for easier debugging
                                                c = Conv(4, 2, ks, perf_stride=(p1, p2), padding=pad, bias=bias,
                                                         stride=stride, groups=groups, strided_backward=back_stride)#.cuda()
                                                op = torch.optim.SGD(c.parameters(), lr=0.01)
                                                c2 = torch.nn.Conv2d(4, 2, ks, padding=pad, bias=bias, stride=stride, groups=groups)#.cuda()
                                                with torch.no_grad():
                                                    c.weight = torch.nn.Parameter(torch.ones_like(c.weight))
                                                    c2.weight = torch.nn.Parameter(torch.ones_like(c2.weight))
                                                    if bias:
                                                        c.bias = torch.nn.Parameter(torch.zeros_like(c.bias))
                                                        c2.bias = torch.nn.Parameter(torch.zeros_like(c2.bias))
                                                t1 = c(torch.arange(0, 4 * inx * iny, 1, dtype=torch.float32).reshape(1, 4, inx, iny))#.cuda())
                                                l = t1.sum()

                                                l.backward()
                                                op.step()
                                                t2 = c2(torch.arange(0, 4 * inx * iny, 1, dtype=torch.float32).reshape(1, 4, inx, iny))#.cuda())
                                                diff = t1 - t2
                                                #continue
                                                if diff[:, :, p1+1:p1 * 2+1, p2+1:p2 * 2+1].sum() > 0.1:
                                                    print(diff[:, :, :p1 * 2, :p2 * 2].sum())
                                                    print(diff,"\n", t1.long(), "\n",t2.long())
                                                    print(inx, iny, p1, p2, pad)
                                            except Exception:

                                                print(back_stride, groups, inx, iny, p1, p2, pad, bias, ks, stride)
                                                print(traceback.format_exc())
                                                #print(inx, iny, p1, p2, pad)
                                                quit()


if __name__ == "__main__":
    test()
