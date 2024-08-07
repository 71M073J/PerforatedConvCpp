import numpy as np
def getTotalNumCalc(net, perf_compare=(2,2)):
    calculationsBase = net._set_perforation((1, 1))._reset()._get_n_calc()
    calculationsOther = net._set_perforation(perf_compare)._reset()._get_n_calc()

    base_n = np.array([x[0] for x in calculationsBase])
    base_o = np.array([x[0] for x in calculationsOther])
    #n_diff = sum([int(calculationsBase[i][0] == calculationsOther[i][0]) for i in range(len(calculationsBase))])

    return f"{(base_n != base_o).sum()} out of {len(calculationsBase)} layers perforated, " \
           f"going from {base_n.sum()} to {base_o.sum()} operations ({(int(10000 * (1 - base_o.sum().astype(float)/base_n.sum().astype(float)))/100)}% improvement).\n" \
           f"(Counting only Conv layers)"
