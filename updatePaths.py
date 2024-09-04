
def update(name):
    with open(name, "r") as f:
        text = f.read()
    with open(name, "w") as f:
        f.write(text.replace("/mnt/c/Users/timotej/pytorch/PyTorch-extension-Convolution/conv_cuda/agriadapt",
                             "/home/tciglaric/magistrska/PerforatedConvolution/PerforatedConvCpp/agriadapt"))

if __name__ == "__main__":
    update("unet_train.py")
    update("agriadapt/segmentation/settings.py")