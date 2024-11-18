import os
def update(name):
    cwd = os.getcwd()
    other = ""
    if "tciglaric" in cwd:
        other = "/home/tciglaric/magistrska/PerforatedConvolution/PerforatedConvCpp/agriadapt"
    elif "username" in cwd:
        other = "/home/username/PerforatedConvCpp/agriadapt"
    else:
        print("device not recognised")
        quit()
    with open(name, "r") as f:
        text = f.read()
    with open(name, "w") as f:
        f.write(text.replace("/mnt/c/Users/timotej/pytorch/PyTorch-extension-Convolution/conv_cuda/agriadapt",
                             other))

if __name__ == "__main__":
    update("unet_train.py")
    update("agriadapt/segmentation/settings.py")