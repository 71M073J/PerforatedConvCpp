import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
if not os.path.exists("./images"):
    os.mkdir("./images")
for name in [("resnet"), ("mobnetv3"), ("mobnetv2")]:
    for perf in [(1, 1), (2, 2), (3, 3), "random", "2by2_equivalent"]:
        losses = []
        accs = []
        perfmode = str(perf[0]) + "x" + str(perf[0]) if type(perf[-1]) == int else perf
        curr_file = f"{name}_{perfmode}"
        with open(f"./res/{curr_file}.txt", "r") as f:
            ff = f.read().replace("\n", ":").replace(" ", "")
            if ff.endswith(":"):
                ff = ff[:-1]
            eps = ff.split("TrainLoss:")[1:]
            for epoch_vals in eps:
                vals = [float(x) for x in epoch_vals.split(":") if x[1] == "."]
                losses.append(vals[::2])
                accs.append(vals[1::2])
        losses = np.array(losses)
        accs = np.array(accs)
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        fig.suptitle(curr_file)
        ax[0].plot(losses[:, 0], label="Train loss", c="forestgreen")
        ax[1].plot(accs[:, 0], label="Train acc", c="limegreen")
        ax[0].plot(losses[:, 1], label="SameAsTrain loss", c="orange")
        ax[1].plot(accs[:, 1], label="SameAsTrain acc", c="coral")
        ax[0].plot(losses[:, 2], label="1x1 loss", c="black")
        ax[1].plot(accs[:, 2], label="1x1 acc", c="silver")
        ax[0].plot(losses[:, 3], label="2x2 loss", c="cyan")
        ax[1].plot(accs[:, 3], label="2x2 acc", c="blue")
        ax[0].plot(losses[:, 4], label="3x3 loss", c="magenta")
        ax[1].plot(accs[:, 4], label="3x3 acc", c="violet")
        ax[0].set_title("Loss progression")
        ax[1].set_title("Accuracy progression")
        ax[0].legend()
        ax[0].set_ylim(-0.2, 3)
        ax[1].legend()
        plt.grid()
        plt.savefig(f"./images/{curr_file}.png")
        plt.show()

            #tests = best_ep.split("Average Epoch Test Loss: ")
            #print(f"Train loss: {tests[0].split()[0]}")