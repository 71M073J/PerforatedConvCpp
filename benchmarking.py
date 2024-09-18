import os.path
import random
import time
import copy
from typing import Union
import time
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
# from torchvision import transforms
from torchvision.models import resnet18, mobilenet_v2, mobilenet_v3_small
import torchvision.transforms.v2 as transforms
from agriadapt.segmentation.data.data import ImageDataset as AgriDataset
from torch.distributions import Categorical
from contextlib import ExitStack
from torchinfo import summary
from pytorch_cinic.dataset import CINIC10
from ucihar import UciHAR
# from Architectures.PerforatedConv2d import PerforatedConv2d
# from Architectures.mobilenetv2 import MobileNetV2
from perforateCustomNet import perforate_net_perfconv as perfPerf
from perforateCustomNet import perforate_net_downActivUp as perfDAU
# from Architectures.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small, MobileNetV3
# from Architectures.resnet import resnet152, resnet18, ResNet
from Architectures.UnetCustom import UNet as UNetCustom
from agriadapt.dl_scripts.UNet import UNet
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
        # print(img.shape,flush=True)
        # print(classes.shape,flush=True)
        # print("----", flush=True)
        return self.norm(img), classes


def get_datasets(data, batch_size, augment=True, image_resolution=None):
    test = None
    tf = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    if "agri" not in data:
        if augment:
            tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
                                                transforms.RandomResizedCrop(size=32)]),
                       transforms.RandomHorizontalFlip()])
    else:
        if augment:
            tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=image_resolution[0], padding=int(image_resolution[0]/8)),
                                                transforms.RandomResizedCrop(size=image_resolution[0])]),
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
        tf.extend([transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])
        # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) #old values<- supposedly miscalculated
        tf = transforms.Compose(tf)
        train = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=tf), batch_size=batch_size, shuffle=True,
            num_workers=num_workers)

        valid = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=tf), batch_size=batch_size, shuffle=False,
            num_workers=num_workers)
    elif "agri" in data:
        print(image_resolution)
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
        train, valid, test = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                         num_workers=num_workers), \
            torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers), \
            torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif data == "ucihar" or ("uci" in data.lower() and "har" in data.lower()):
        # UCIHAR is already normalised - kinda
        # "but we can still do it"
        # tf = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        tf = []
        tf.append(transforms.Normalize([-4.1705e-04, -9.0756e-05, 3.3419e-01],
                                       [0.1507, 0.3648, 0.5357]))

        tf = transforms.Compose(tf)

        train = torch.utils.data.DataLoader(UciHAR("train", transform=tf), batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers)
        valid = torch.utils.data.DataLoader(UciHAR("test", transform=tf), batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)
    else:
        raise ValueError("Not supported dataset")

    print("Datasets loaded...")
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
                results[key].append(value.cpu().item())
            else:
                results[key] = [value.cpu().item()]


def get_perfs(perforation_mode, n_conv):

    if type(perforation_mode) == tuple:
        perforation_mode = perforation_mode[0]
    if type(perforation_mode) == int:
        return perforation_mode, perforation_mode
    if type(perforation_mode[0]) == int:
        return perforation_mode
    if perforation_mode == "incremental":
        raise NotImplementedError()
    elif perforation_mode == "random":  # avg_proc 0.37 of non-perf
        perfs = np.random.randint(1, 3 + 1, (n_conv, 2))
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


def train(net, op, data_loader, device, loss_fn, vary_perf, batch_size, perforation_type, run_name, grad_clip,
          perforation_mode, pretrained, in_size):
    if perforation_type is None:
        perforation_type = "perf"

    results = {}
    n_conv = 0
    train_accs = []
    losses = []
    # entropies = 0
    # TODO class_accs = np.zeros((2, 15))
    weights = []
    if hasattr(net, "_set_perforation") and type(perforation_mode[0]) == int:
        n_conv = len(net._get_n_calc())
        net._set_perforation(perforation_mode)
        net._reset()
    elif perforation_mode[0] is not None:
        if type(perforation_mode[0]) == int:
            if "dau" in perforation_type.lower():
                perfDAU(net, in_size=in_size, perforation_mode=perforation_mode, pretrained=pretrained)
            elif "perf" in perforation_type.lower():
                perfPerf(net, in_size=in_size, perforation_mode=perforation_mode, pretrained=pretrained)
        else:
            if "dau" in perforation_type.lower():
                perfDAU(net, in_size=in_size, perforation_mode=(1, 1), pretrained=pretrained)
            elif "perf" in perforation_type.lower():
                perfPerf(net, in_size=in_size, perforation_mode=(1, 1), pretrained=pretrained)
        n_conv = len(net._get_n_calc())
        net._set_perforation(get_perfs(perforation_mode, n_conv))
        # net._reset()
    for i, (batch, classes) in enumerate(data_loader):
        if vary_perf is not None and n_conv > 0 and type(perforation_mode[0]) == str:
            perfs = get_perfs(perforation_mode[0], n_conv)
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
        if type(data_loader.dataset) in [torchvision.datasets.CIFAR10, CINIC10, UciHAR]:
            # print("Should be here")
            acc = (F.softmax(pred.detach(), dim=1).argmax(dim=1) == classes).cpu()
            train_accs.append(torch.sum(acc) / batch_size)
        else:
            calculate_segmentation_metrics(classes, pred, run_name, metrics, device, results)
            acc = torch.mean(torch.tensor(results[f"{run_name}/iou/weeds"]))
            train_accs.append(acc)
            ...

        # entropy = Categorical(
        #    probs=torch.maximum(F.softmax(pred.detach().cpu(), dim=1), torch.tensor(1e-12)))  # F.softmax(pred.detach().cpu(), dim=1)
        # entropies += entropy.entropy().mean()
        # acc = (F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes)
    for metric in results:
        results[metric] = torch.mean(torch.tensor(results[metric]))
    return losses, train_accs, results


def validate(net, valid_loader, device, loss_fn, file, eval_mode, batch_size, reporting, run_name, best_worst=False):
    train_mode = None
    valid_losses = []
    valid_accs = []
    if best_worst:
        max_im = (torch.zeros_like(valid_loader.dataset[0][0]), torch.zeros_like(valid_loader.dataset[0][0][0]))
        min_im = (torch.zeros_like(valid_loader.dataset[0][0]), torch.zeros_like(valid_loader.dataset[0][0][0]))
    else:
        max_im = 0
        min_im = 0
    results = {}
    # ep_valid_losses = []
    net.eval()
    if hasattr(net, "_get_perforation"):
        train_mode = net._get_perforation()
    max_dist = -99999
    min_dist = 99999
    max_ind, min_ind = -1, -1
    with torch.no_grad():
        if eval_mode is not None:
            net._set_perforation(eval_mode)
            # net._reset()
            # print(net._get_perforation())
        for i, (batch, classes) in enumerate(valid_loader):
            pred = net(batch.to(device))
            classes = classes.to(device)
            loss = loss_fn(pred, classes)
            valid_losses.append(loss.detach().cpu())
            softm = F.softmax(pred, dim=1)

            if type(valid_loader.dataset) == AgriDataset:
                if best_worst:
                    nearness = torch.abs(softm - classes)
                    suma = nearness.sum(dim=(1,2,3))
                    nearthest = suma.argmax()
                    nearthest_score = suma.max()
                    farthest = suma.argmin()
                    farthest_score = suma.min()
                    if nearthest_score > max_dist:
                        max_dist = nearthest_score
                        max_ind = nearthest + i * batch_size
                        max_im = (classes[nearthest][0].detach().cpu(), pred[nearthest][0].detach().cpu())
                    if farthest_score < min_dist:
                        min_dist = farthest_score
                        min_ind = farthest + i * batch_size
                        min_im = (classes[farthest][0].detach().cpu(), pred[farthest][0].detach().cpu())
                calculate_segmentation_metrics(classes, pred, run_name, metrics, device, results)
                acc = torch.mean(torch.tensor(results[f"{run_name}/iou/weeds"]))
                valid_accs.append(acc)

            else:
                if best_worst:
                    nearness = torch.abs(softm - torch.nn.functional.one_hot(classes.long(), num_classes=10))
                    suma = nearness.sum(dim=1)
                    nearthest = suma.argmax() + i * batch_size
                    nearthest_score = suma.max()
                    farthest = suma.argmin() + i * batch_size
                    farthest_score = suma.min()
                    if nearthest_score > max_dist:
                        max_dist = nearthest_score
                        max_ind = nearthest
                    if farthest_score < min_dist:
                        min_dist = farthest_score
                        min_ind = farthest
                acc = (softm.argmax(dim=1) == classes)
                valid_accs.append(torch.sum(acc) / batch_size)
        if reporting:
            if file is not None:
                print(f"Epoch mean acc: {np.mean(valid_accs).item()}, loss: {np.mean(valid_losses).item()}", file=file)
            print(f"Epoch mean acc: {np.mean(valid_accs).item()}, loss: {np.mean(valid_losses).item()}")
        # ep_valid_losses.append(l2.item() / (i + 1))
    ims = (max_im, min_im)
    if train_mode is not None:
        net._set_perforation(train_mode)
        # print(train_mode, flush=True)
        # net._reset()
    return valid_losses, valid_accs, (max_ind, min_ind), results, ims


def benchmark(net, op, scheduler=None, loss_function=torch.nn.CrossEntropyLoss(), run_name="test",
              perforation_mode=(2, 2), perforation_type="dau",
              train_loader=None, valid_loader=None, test_loader=None, max_epochs=1, in_size=(2, 3, 32, 32),
              summarise=True, pretrained=False,
              device="cpu", batch_size=64, reporting=True, file=None, grad_clip=None, eval_modes=(None,)):
    if type(perforation_mode[0]) == str:
        vary_perf = True
    else:
        vary_perf = None
    if perforation_mode[0] is None:
        eval_modes = (None,)
    if eval_modes is None:
        eval_modes = (None,)
    timeElapsed = 0
    if summarise:
        summary(net, input_size=in_size)
    best_valid_losses = [999] * len(eval_modes)
    best_models = [None] * len(eval_modes)
    imss = []
    for epoch in range(max_epochs):
        if reporting:
            if file is not None:
                print(f"\nEpoch {epoch} training:", file=file)
            print(f"\nEpoch {epoch} training:")
        torch.cuda.synchronize()
        t0 = time.time()
        losses, train_accs, results = train(net, op, train_loader, device, loss_fn=loss_function,
                                   perforation_mode=perforation_mode,
                                   batch_size=batch_size, perforation_type=perforation_type, pretrained=pretrained,
                                   run_name=run_name, grad_clip=grad_clip, vary_perf=vary_perf, in_size=in_size)
        torch.cuda.synchronize()
        t1 = time.time()
        timedelta = int((t1 - t0) * 1000) / 1000
        timeElapsed += (t1 - t0)
        if reporting:
            if file is not None:
                print(f"Average Epoch {epoch} Train Loss:", np.mean(losses).item(), file=file)
                print(f"Epoch mean acc: {np.mean(train_accs).item()}, Epoch time: {timedelta} s", file=file)
                print(results, file=file)
            print(f"Average Epoch {epoch} Train Loss:", np.mean(losses).item())
            print(f"Epoch mean acc: {np.mean(train_accs).item()}, Epoch time: {timedelta} s")
            print(results)

        for ind, mode in enumerate(eval_modes):

            print("\ntesting eval mode", mode)
            if file is not None:
                print("\ntesting eval mode", mode, file=file)
            valid_losses, valid_accs, (max_ind, min_ind), allMetrics, ims = validate(net=net, valid_loader=valid_loader, device=device,
                                                                    loss_fn=loss_function,
                                                                    file=file, batch_size=batch_size, eval_mode=mode,
                                                                    run_name=run_name, reporting=reporting)
            curr_loss = np.mean(valid_losses)
            if curr_loss < best_valid_losses[ind]:
                best_valid_losses[ind] = curr_loss
                best_models[ind] = copy.deepcopy(net.state_dict())
        if scheduler is not None:
            scheduler.step()
            print(", Current LR:", scheduler.get_last_lr()[0])

    if test_loader is None:
        test_loader = valid_loader
    best_outputs = []
    best_worst_ind = []
    if eval_modes is None:
        eval_modes = (None,)
    for ind, mode in enumerate(eval_modes):
        net.eval()
        if best_models[ind] is not None:
            net.load_state_dict(best_models[ind])
        if mode is not None:
            net._set_perforation(mode)
            # net._reset()
        net.eval()
        print("\nValidating eval mode", mode)
        test_losses, test_accs, indexes, allMetrics, ims = validate(net=net, valid_loader=test_loader, device=device,
                                                   loss_fn=loss_function,
                                                   file=file, batch_size=batch_size, eval_mode=mode,
                                                   reporting=reporting, run_name=run_name, best_worst=True)
        imss.append(ims)
        best_worst_ind.append(indexes)
        h = f"Validation loss ({mode}):" + str(np.mean(test_losses))
        print(h)
        best_outputs.append(h)
        h2 = f"Validation acc ({mode}):" + str(np.mean(test_accs))
        best_outputs.append(h2)
        print(h2)
    h3 = "Training time:" + str(timeElapsed) + " seconds"
    print("Training time:", timeElapsed, " seconds")
    best_outputs.append(h3)
    return best_outputs, best_worst_ind, imss


def runAllTests():
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    architectures = [
        [[(resnet18, "resnet18"), (mobilenet_v2, "mobnetv2"), (mobilenet_v3_small, "mobnetv3s")],
         ["cifar", "cinic", "ucihar"], [32]],
        [[(UNet, "unet_agri"), (UNetCustom, "unet_custom")], ["agri"], [128, 256, 512]]
    ]

    for version in architectures:  # classigication, segmetnationg
        for dataset in version[1]:

            loss_fn = torch.nn.CrossEntropyLoss()
            if dataset == "agri":
                loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9])).to(device)
                lr = 0.01
                max_epochs = 1  # 300
                batch_size = 32
            elif dataset == "ucihar":
                max_epochs = 1  # 100
                batch_size = 32
                lr = 0.01
            else:
                max_epochs = 1  # 200
                batch_size = 32
                lr = 0.1

            for model, modelname in version[0]:
                for img in version[2]:
                    #model = UNet
                    #img = 128
                    #dataset = "agri"
                    img_res = (img, img)
                    in_size = (2, 3, img, img)
                    if img == 128:
                        batch_size = 32
                    elif img == 256:
                        batch_size = 16
                    elif img == 512:
                        batch_size = 4

                    alreadyNoPerf = False
                    for perforation in (None, 2, 3, "random", "2by2_equivalent"):
                        perf = (perforation, perforation)
                        if perforation is None:
                            if alreadyNoPerf:
                                continue
                            else:
                                alreadyNoPerf = True

                        for perf_type in ["perf", "dau"]:
                            if perforation is None:
                                if perf_type == "dau":
                                    continue
                                else:
                                    perf_type = None
                            prefix = "allTests"
                            name = f"{modelname}_{dataset}_{img}_{perforation}_{perf_type}"
                            curr_file = f"{name}"
                            if os.path.exists(f"./{prefix}/{curr_file}_best.txt"):
                                print("file for", curr_file, "already exists, skipping...")
                                continue
                            if "agri" in dataset:
                                net = model(2).to(device)
                            else:
                                net = model(num_classes=10).to(device)

                            op = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0005)
                            train_loader, valid_loader, test_loader = get_datasets(dataset, batch_size, True,
                                                                                   image_resolution=img_res)

                            dims = [x for x in [0, 1, 2] if x != np.argmin(valid_loader.dataset[0][0].shape)]
                            #print(dims)
                            #h = torch.cat(
                            #         (valid_loader.dataset[0][1][0],
                            #          valid_loader.dataset[1][1][0]),
                            #         dim=1)
                            #lab = torch.tile(h, (3, 1, 1))#.movedim(0,-1)
                            #print(lab.shape)
                            #im = torch.cat((valid_loader.dataset[0][0], valid_loader.dataset[1][0]), dim=2)
                            #print(im.shape)
                            #im2 = torch.cat((im, lab), dim=1)
                            #print(im2.shape)
                            #quit()
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=max_epochs)

                            run_name = f"{curr_file}"
                            if not os.path.exists(f"./{prefix}/"):
                                os.mkdir(f"./{prefix}")
                            if not os.path.exists(f"./{prefix}/imgs"):
                                os.mkdir(f"./{prefix}/imgs")
                            print("starting run:", curr_file)

                            with open(f"./{prefix}/{curr_file}.txt", "w") as f:
                                eval_modes = [None, (2, 2), (3, 3)]
                                best_out, inds, ims = benchmark(net, op, scheduler, train_loader=train_loader,
                                                           valid_loader=valid_loader, test_loader=test_loader,
                                                           max_epochs=max_epochs, device=device, perforation_mode=perf,
                                                           run_name=run_name, batch_size=batch_size, loss_function=loss_fn,
                                                           eval_modes=eval_modes, in_size=in_size,
                                                           perforation_type=perf_type, file=f, summarise=False)

                                try:
                                    valid_loader.dataset.transform.transforms = [transforms.Resize(img)]
                                except:
                                    try:
                                        valid_loader.dataset.tf.transforms = [transforms.Resize(img)]
                                    except:
                                        try:
                                            valid_loader.dataset.trans.transforms = [transforms.Resize(img)]
                                        except:
                                            pass
                                if "agri" in dataset:
                                    imgs = torch.cat([torch.cat((torch.cat((valid_loader.dataset[mx][0], torch.tile(torch.cat((ims[i][0][0], ims[i][0][1]), dim=1), dims=(3,1,1))), dim=2),
                                                                 torch.cat((valid_loader.dataset[mn][0], torch.tile(torch.cat((ims[i][1][0], ims[i][1][1]), dim=1), dims=(3,1,1))), dim=2)), dim=1)
                                                        for i, (mx, mn) in enumerate(inds)], dim=2)
                                else:
                                    imgs = torch.cat(
                                        [torch.cat((valid_loader.dataset[mx][0], valid_loader.dataset[mn][0]), dim=dims[0])
                                         for
                                         (mx, mn) in inds], dim=dims[1])
                                if imgs.shape[0] == 3:
                                    # print("wtf?")
                                    imgs = imgs.movedim(0, -1)
                                plt.imshow(imgs)
                                if perf_type is not None:
                                    plt.title("Perforation mode\n" + str(" "*28).join(
                                        [str(x) if type(x) != tuple else (str(x[0]) + "x" + str(x[1])) for x in
                                         eval_modes]))
                                else:
                                    plt.title("No perforation")
                                plt.ylabel("Best img                Worst img")
                                plt.savefig(f"./{prefix}/imgs/{curr_file}.png")
                                plt.clf()
                            with open(f"./{prefix}/{curr_file}_best.txt", "w") as ff:
                                print(best_out, file=ff)
                                quit()


if __name__ == "__main__":
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    runAllTests()
    quit()
    net = torchvision.models.resnet18(num_classes=10)
    perfDAU(net, (2, 3, 32, 32))
    net._set_perforation((1, 3))
    net._reset()
    quit(123)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    architectures = [resnet18, mobilenet_v2, mobilenet_v3_small, UNet, UNetCustom]
    net = architectures[0](num_classes=10)
    net.train()
    # perfPerf(net)
    # perfDAU(net)
    max_epochs = 2
    dataset = "ucihar"
    batch_size = 32
    img_res = (128, 128)
    in_size = img_res if "agri" in dataset.lower() else (32, 32)
    in_size = (2, 3, in_size[0], in_size[1])
    op = torch.optim.Adam(net.parameters(), lr=0.001)
    op = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(op, 0.99)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=max_epochs)
    train_loader, valid_loader, test_loader = get_datasets(dataset, batch_size, True, image_resolution=img_res)

    run_name = "test"
    benchmark(net, op, scheduler, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
              max_epochs=max_epochs, device=device, perforation_mode=(3, 3), run_name=run_name, batch_size=batch_size,
              eval_modes=[None, (2, 2), (3, 3)], in_size=in_size, perforation_type="perf")
