import numpy as np
import os
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def makeCapital(st):
    return st[0].upper() + st[1:]

network = "Agri"

if not os.path.exists(f"powerdrawdata/unet{network}Power.bin"):
    with open(f"powerdrawdata/unet{network}Power.csv", "r") as f:
        power = []
        for i, line in enumerate(f):
            if i == 0: continue
            power.append(float(line.split(",")[-1]))
            if i % 500000 == 0:
                print(i)
        powArr = np.array(power)
        powArr.tofile(f"powerdrawdata/unet{network}Power.bin")
        print(len(powArr))

if not os.path.exists(f"powerdrawdata/unet{network}Power_under20hz.bin"):
    arr = np.fromfile(f"powerdrawdata/unet{network}Power.bin")
    # plt.plot([i for i in range(len(arr[::10]))], arr[::10])
    # plt.show()

    arr2 = butter_lowpass_filter(arr, 10, 5000.0, 6)

    plt.plot([i / 50 for i in range(len(arr2[::100]))], arr2[::100])
    plt.show()

    arr2.tofile(f"powerdrawdata/unet{network}Power_under20hz.bin")

arr = np.fromfile(f"powerdrawdata/unet{network}Power.bin")

plt.plot([i / 50 for i in range(len(arr[::100]))], arr[::100])
plt.show()
def get_power_energy(fromsec, tosec, arr, name="Unnamed", fs=5000):
    mean = np.mean(arr[fromsec * fs:tosec * fs])
    time = tosec - fromsec
    print(mean, "Mean power draw (mA) for", name)
    print(int(mean * time) / 1000, "J of energy consumed for 1 epoch of training")
    print("Trained for", time, "seconds\n")
    return mean, mean*time / 1000, time

means = []
energies = []
times = []
for m, j, t in [get_power_energy(640, 1560, arr, "Unperforated"),
get_power_energy(1864, 2213, arr, "2 Perf"),
get_power_energy(2492, 2831, arr, "2 dau"),
get_power_energy(3110, 3378, arr, "3 perf"),
get_power_energy(3660, 3908, arr, "3 dau"),
get_power_energy(4192, 4641, arr, "random perf"),
get_power_energy(4914, 5372, arr, "random dau"),
get_power_energy(5656, 6064, arr, "2eq perf"),
get_power_energy(6340, 6736, arr, "2eq dau"),
 ] if network == "custom" else [
get_power_energy(65,1857, arr, "Unperforated"),
get_power_energy(2313, 2952, arr, "2 Perf"),
get_power_energy(3418, 3988, arr, "2 dau"),
get_power_energy(4457, 4943, arr, "3 perf"),
get_power_energy(5396, 5840, arr, "3 dau"),
get_power_energy(6291, 7123, arr, "random perf"),
get_power_energy(7587, 8409, arr, "random dau"),
get_power_energy(8864, 9626, arr, "2eq perf"),
get_power_energy(10076, 10786, arr, "2eq dau"),
 ]:
    means.append(m)
    energies.append(j)
    times.append(t)
fig, ax = plt.subplots(2,1, figsize=(8,8))
plt.suptitle(f"{makeCapital(network)} UNet\nPower draw on RPi4")
ax[0].bar(range(9),means, label="Avg power draw during training (mW)")
ax[1].bar(range(9),energies, label="Per-epoch energy consumption (J)")
ax[0].set_ylim(3000, 6100)
ax[0].set_ylabel("Avg power draw (mW)")
ax[1].set_ylabel("Total energy consumed (J)")
ax[0].legend(loc="lower right")
ax[0].set_xticks(range(9),[])
ax[1].set_xticks(range(9),["Unperforated", "2x2, standard", "2x2, DAU", "3x3, standard", "3x3, DAU",
                           "random, standard", "random, DAU", "2by2 eq, standard", "2by2 eq, DAU"], rotation=90)

ax[0].grid()
ax[1].grid()
ax[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"powerDrawUnet{makeCapital(network)}.png")
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
plt.suptitle(f"{makeCapital(network)} UNet\nPower draw on RPi4, baseline subtracted")
ax[0].bar(range(9), [x - 1600 for x in means], label="Avg power draw during training (mW)")
ax[1].bar(range(9), [x - times[i] * 1.6 for i, x in enumerate(energies)], label="Per-epoch energy consumption (J)")
ax[0].set_ylim(2500, 3500)
ax[0].set_ylabel("Avg power draw (mW)")
ax[1].set_ylabel("Total energy consumed (J)")
ax[0].legend(loc="lower right")
ax[0].set_xticks(range(9), [])
ax[1].set_xticks(range(9), ["Unperforated", "2x2, standard", "2x2, DAU", "3x3, standard", "3x3, DAU",
                            "random, standard", "random, DAU", "2by2 eq, standard", "2by2 eq, DAU"],
                 rotation=90)

ax[0].grid()
ax[1].grid()
ax[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"powerDrawUnet{makeCapital(network)}_nobase.png")
plt.show()
# 640: 1657(1582, 1560?): none - none
# 1864: 2213
# 2492: 2831
# 3110: 3378
# 3660: 3908
# 4192: 4641
# 4914: 5372
# 5656: 6064
# 6340: 6736

# 65,1857
# 2313, 2952
# 3418, 3988
# 4457, 4943
# 5396, 5840
# 6291, 7123
#7587, 8409
#8864, 9626
#10076, 10786
11240, 12276
12470, 13537
"""Power measurement was conducted using a monsoon high-voltage power meter (HVPM) power supply, 
accompanied by Monsoon Powertool utility, to save the measurements to a computer. 
The power supply was connected to a Raspberry Pi model 4, same as with speedup measurements. 
The power draw was calculated with a sampling rate of 5000 Hz."""