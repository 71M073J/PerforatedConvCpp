
def perforateNet(net, perforation_mode):
    ...
    #TODO recursively iterate over net children, replacing all Conv2d with PerforatedConv2d, with same params, but with the perforation mode
    # and also add the get, set perf methods, and get_n_calc, and reset