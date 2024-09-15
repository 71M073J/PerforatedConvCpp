import os.path
import time
import copy
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models.resnet
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score
#from torchvision import transforms
import torchvision.transforms.v2 as transforms
from torch.distributions import Categorical
from contextlib import ExitStack
from torchinfo import summary
from pytorch_cinic.dataset import CINIC10
from ucihar import UciHAR
from Architectures.PerforatedConv2d import PerforatedConv2d
from Architectures.mobilenetv2 import MobileNetV2
from Architectures.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small, MobileNetV3
from Architectures.resnet import resnet152, resnet18, ResNet
from Architectures.UnetCustom import UNet
from torch import argmax, where, cat, stack
import agriadapt.segmentation.settings as settings
import agriadapt.segmentation.data.data as dd

metrics = {
    "iou": BinaryJaccardIndex,
    "precision": BinaryPrecision,
    "recall": BinaryRecall,
    "f1score": BinaryF1Score,
}
seed = 123
g = torch.Generator(seed)
num_workers = 1
class NormalizeImageOnly(torch.nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        self.norm = transforms.Normalize(means, stds)
    def forward(self, img, classes):
        # Do some transformations
        #print(img.shape,flush=True)
        #print(classes.shape,flush=True)
        #print("----", flush=True)
        return self.norm(img), classes
def get_datasets(data, batch_size, augment=True, image_resolution=None):
    test = None
    tf = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    if augment:
        tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
                                            transforms.RandomResizedCrop(size=32)]),
                   transforms.RandomHorizontalFlip()])
    if data == "cinic":
        tf.append(transforms.RandomRotation(45))
        tf.append(transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                       [0.24205776, 0.23828046, 0.25874835]))
        tf = transforms.Compose(tf)
        train = DataLoader(CINIC10(partition="train", download=True, transform=tf),  # collate_fn=col,
                              num_workers=num_workers, batch_size=batch_size, shuffle=True,
                              generator=g)
        valid = DataLoader(
            CINIC10(partition="valid", download=True, transform=tf), num_workers=num_workers,  # collate_fn=col,
            batch_size=batch_size, shuffle=True,
            generator=g, )
        test = DataLoader(
            CINIC10(partition="test", download=True, transform=tf), num_workers=num_workers,  # collate_fn=col,
            batch_size=batch_size, shuffle=True,
            generator=g, )
    elif data == "cifar":
        tf.extend([transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))])
            #(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) #old values<- supposedly miscalculated
        tf = transforms.Compose(tf)
        train = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=tf), batch_size=batch_size, shuffle=True, num_workers=num_workers)

        valid = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=tf), batch_size=batch_size, shuffle=False,
            num_workers=num_workers)
    elif "agri" in data:

        tf.append(transforms.RandomRotation(45))
        tf.append(NormalizeImageOnly([0.4858, 0.3100, 0.3815],
                                       [0.1342, 0.1193, 0.1214]))
        tf = transforms.Compose(tf)
        train, valid = dd.ImageImporter(
            "bigagriadapt",
            validation=True,
            sample=0,
            smaller=image_resolution, transform=tf
        ).get_dataset()
        _, test = dd.ImageImporter(
            "bigagriadapt",
            validation=False,
            sample=0,
            smaller=image_resolution, transform=tf
        ).get_dataset()
        train, valid, test = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),\
                             torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers),\
                             torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif data=="ucihar" or ("uci" in data.lower() and "har" in data.lower()):
        #UCIHAR is already normalised - kinda
        #"but we can still do it"
        #tf = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        tf = []
        tf.append(transforms.Normalize([-4.1705e-04, -9.0756e-05, 3.3419e-01],
                                       [0.1507, 0.3648, 0.5357]))

        tf = transforms.Compose(tf)

        train = torch.utils.data.DataLoader(UciHAR("train", transform=tf), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid = torch.utils.data.DataLoader(UciHAR("test", transform=tf), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        raise ValueError("Not supported dataset")

    return train, valid, test

def get_binary_masks_infest(mask, batch_dim=True, dim=3):
    result = []
    if batch_dim:
        # Arg max each mask separately
        arg_mask = argmax(mask, dim=1)
        for i in range(mask.shape[0]):
            # Create per-class binary masks
            b_mask = where(arg_mask[i] == 0, 1, 0)[None, :, :]
            w_mask = where(arg_mask[i] == 1, 1, 0)[None, :, :]
            # If we have a lettuce class as well
            if dim == 3:
                l_mask = where(arg_mask[i] == 2, 1, 0)[None, :, :]
                result.append(cat((b_mask, w_mask, l_mask)))
            elif dim == 2:
                result.append(cat((b_mask, w_mask)))
        return stack(result)


def calculate_segmentation_metrics(y_true, y_pred, name, metrics, device, results):
    y_pred = get_binary_masks_infest(y_pred, dim=2)
    assert y_true.shape == y_pred.shape
    for metric in metrics:
        for i, pred_class in enumerate(["back", "weeds"]):
            key = f"{name}/{metric}/{pred_class}"
            value = metrics[metric](validate_args=False).to(device)(
                y_true[:, i], y_pred[:, i]
            )
            if key in results:
                results[key].append(value.cpu())
            else:
                results[key] = [value.cpu()]


def get_perfs(perforation_mode, n_conv):
    if perforation_mode == "incremental":
        raise NotImplementedError()
    elif perforation_mode == "random":  # avg_proc 0.37 of non-perf
        perfs = np.random.randint(1, perforation_mode + 1, (perforation_mode, 2))
    elif perforation_mode == "2by2_equivalent":
        perfs = np.ones((n_conv, 2), dtype=int)
        rns = np.random.random(n_conv)
        for i in range(n_conv):
            rn = rns[i]
            if rn < 0.20029760897159576:
                perfs[i] = 3, 3
            elif rn < 0.32178425043821335:
                perfs[i] = 2, 3
            elif rn < 0.44327089190483093:
                perfs[i] = 3, 2
            # elif rn < 0.5378847792744637: #ultra bad memory layout, skipping - 3,1 is the same time complexity
            #    perfs[i] = 3,1
            elif rn < 0.6407210007309914:
                perfs[i] = 1, 3
            elif rn < 0.7435572221875191:
                perfs[i] = 2, 2
            elif rn < 0.8306062072515488:
                perfs[i] = 1, 2
            elif rn < 0.9176551923155785:
                perfs[i] = 2, 1
            else:
                perfs[i] = 1, 1
    else:
        raise ValueError("Supported vary modes are \"random\" and \"2by2_equivalent\"")
    return perfs


def train(net, op, data_loader, device, loss_fn, vary_perf, batch_size, run_name, grad_clip, perforation):
    n_conv = 0
    train_accs = []
    losses = []
    # entropies = 0
    # TODO class_accs = np.zeros((2, 15))
    weights = []
    if hasattr(net, "_set_perforation"):
        n_conv = len(net._get_n_calc())
        net._set_perforation(perforation)
        net._reset()
    for i, (batch, classes) in enumerate(data_loader):
        if vary_perf is not None and n_conv > 0:
            perfs = get_perfs(vary_perf, n_conv)
            net._set_perforation(perfs)
            # net._reset()

        batch = batch.to(device)
        classes = classes.to(device)
        pred = net(batch)
        loss = loss_fn(pred, classes)
        loss.backward()
        losses.append(loss.item())
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        op.step()
        op.zero_grad()
        if type(data_loader.dataset) in [torchvision.datasets.CIFAR10, CINIC10, UciHAR]:  # TODO UCIHAR
            #print("Should be here")
            acc = (F.softmax(pred.detach(), dim=1).argmax(dim=1) == classes).cpu()
        else:
            #print("should not be here")
            results = {}
            calculate_segmentation_metrics(classes, pred, run_name, metrics, device, results)
            acc = results[f"{run_name}/iou/weeds"][0]
            ...
            # Assuming segmentation
            # TODO TODO

        train_accs.append(torch.sum(acc) / batch_size)
        # entropy = Categorical(
        #    probs=torch.maximum(F.softmax(pred.detach().cpu(), dim=1), torch.tensor(1e-12)))  # F.softmax(pred.detach().cpu(), dim=1)
        # entropies += entropy.entropy().mean()
        # acc = (F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes)


    return losses, train_accs


def validate(net, valid_loader, device, loss_fn, file, eval_mode, batch_size, reporting):
    train_mode = None
    valid_losses = []
    valid_accs = []
    # ep_valid_losses = []
    net.eval()
    if hasattr(net, "_get_perforation"):
        train_mode = net._get_perforation()
    with torch.no_grad():
        if eval_mode is not None:
            net._set_perforation(eval_mode)
            # net._reset()
            # print(net._get_perforation())
        for i, (batch, classes) in enumerate(valid_loader):
            pred = net(batch.to(device))
            loss = loss_fn(pred, classes.to(device))
            valid_losses.append(loss.detach().cpu())
            acc = (F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes)
            valid_accs.append(torch.sum(acc) / batch_size)

        if reporting:
            if file is not None:
                print(f"Epoch mean acc: {np.mean(valid_accs).item()}, loss: {np.mean(valid_losses).item()}", file=file)
            print(f"Epoch mean acc: {np.mean(valid_accs).item()}, loss: {np.mean(valid_losses).item()}")
        # ep_valid_losses.append(l2.item() / (i + 1))

    if train_mode is not None:
        net._set_perforation(train_mode)
        net._reset()
    return valid_losses, valid_accs


def benchmark(net, op, scheduler=None, loss_function=torch.nn.CrossEntropyLoss(), run_name="test",
              perforation_mode=(2, 2),
              train_loader=None, valid_loader=None, test_loader=None, max_epochs=1, input_size=(1, 3, 32, 32),
              summarise=True,
              device="cpu", batch_size=64, reporting=True, file=None, grad_clip=None, eval_modes=(None,)):
    vary_perf = "dynamic" in run_name

    if summarise:
        summary(net, input_size=input_size)
    best_valid_losses = [999] * len(eval_modes)
    best_models = [None] * len(eval_modes)
    for epoch in range(max_epochs):
        losses, train_accs = train(net, op, train_loader, device, loss_fn=loss_function, perforation=perforation_mode,
                                   batch_size=batch_size,
                                   run_name=run_name, grad_clip=grad_clip, vary_perf=vary_perf)
        if reporting:
            if file is not None:
                print(f"Average Epoch {epoch} Train Loss:", np.mean(losses).item(), file=file)
                print(f"Epoch mean acc: {np.mean(train_accs).item()}", file=file)
            print(f"Average Epoch {epoch} Train Loss:", np.mean(losses).item())
            print(f"Epoch mean acc: {np.mean(train_accs).item()}")

        for ind, mode in enumerate(eval_modes):

            print("\ntesting eval mode", mode)
            if file is not None:
                print("\ntesting eval mode", mode, file=file)
            valid_losses, valid_accs = validate(net=net, valid_loader=valid_loader, device=device, loss_fn=loss_function,
                                          file=file, batch_size=batch_size, eval_mode=mode, reporting=reporting)
            # TODO net saving
            curr_loss = np.mean(valid_losses)
            if curr_loss < best_valid_losses[ind]:
                best_valid_losses[ind] = curr_loss
                best_models[ind] = copy.deepcopy(net.state_dict())
        if scheduler is not None:
            scheduler.step()
            print(", Current LR:", scheduler.get_last_lr()[0])

    if test_loader is not None:
        for ind, mode in enumerate(eval_modes):
            # TODO load best model
            net.eval()
            if best_models[ind] is not None:
                net.load_state_dict(best_models[ind])
            if mode is not None:
                net._set_perforation(mode)
                # net._reset()
            test_losses, test_accs = validate(net=net, valid_loader=test_loader, device=device, loss_fn=loss_function,
                                            file=file, batch_size=batch_size, eval_mode=mode, reporting=reporting)

            print("\nValidating eval mode", mode)
            print("Validation loss:", np.mean(test_losses))
            print("Validation acc:", np.mean(test_accs))

if __name__ == "__main__":
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    net = UNet(2)
    net.train()
    op = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(op, 0.99)
    train_loader, valid_loader, test_loader = get_datasets("agri", 1, True, image_resolution=(128,128))
    print("Datasets loaded, training...")
    benchmark(net, op, scheduler, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        max_epochs=5, device=device, perforation_mode=None, run_name="ASDA")