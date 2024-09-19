import matplotlib.pyplot as plt
import torch
import numpy as np
h = torch.randn((10,10)) * 100
eval_modes = [None, (2,2), (3,3)]
for confs in [[h], [h,h,h]]:
    if len(confs) == 1:
        perf_type = None
    else:
        perf_type = False

    n_samp = 10000
    fig, ax = plt.subplots(len(confs), 1, figsize=(5, 15) if len(confs) != 1 else (6,5))
    if len(confs) == 1:
        ax = [ax]
    h=torch.cat(confs, dim=1)/n_samp
    mins, maxs = h.min().item(), h.max().item()
    #print(mins, maxs, n_samp, confs)
    for i, conf in enumerate(confs):
        imlast = ax[i].imshow((conf/n_samp)*100, vmin=mins*100, vmax=maxs*100)
        if perf_type is not None:
            ax[i].set_title(f"Perforation mode {eval_modes[i]}")
        else:
            ax[0].set_title("No perforation")

        ax[i].set_xticks([0], [""])
        ax[i].set_yticks(list(range(10)), ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    ax[-1].set_xticks(list(range(10)),
                     ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                      "horse", "ship", "truck"], rotation=90)

    # add space for colour bar
    if len(confs) == 1:
        fig.subplots_adjust(right=0.85, top=0.9, bottom=0.2)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.04, 0.7])

        c = plt.colorbar(imlast, cax=cbar_ax, ticks = [mins*100, maxs*100])
    else:
        fig.subplots_adjust(right=0.85, top=0.9, bottom=0.2)
        cbar_ax = fig.add_axes([0.82, 0.4, 0.02, 0.3])
        c = plt.colorbar(imlast, cax=cbar_ax, ticks = [mins*100, maxs*100])