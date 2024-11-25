import os.path
import random
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models.resnet
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchvision.models import resnet18, mobilenet_v2, mobilenet_v3_small
import torchvision.transforms.v2 as transforms
from agriadapt.segmentation.data.data import ImageDataset as AgriDataset
from torchinfo import summary
try:
    from pytorch_cinic.dataset import CINIC10
except:
    noCinic = True
from ucihar import UciHAR
from perforateCustomNet import perforate_net_perfconv as perfPerf
from perforateCustomNet import perforate_net_downActivUp as perfDAU
from Architectures.UnetCustom import UNet as UNetCustom
from agriadapt.dl_scripts.UNet import UNet
from torch import argmax, where, cat, stack
import agriadapt.segmentation.data.data as dd

metrics = {
    "iou": BinaryJaccardIndex,
    "precision": BinaryPrecision,
    "recall": BinaryRecall,
    "f1score": BinaryF1Score,
}
seed = 123
g = torch.Generator(device="cpu")
g.manual_seed(123)
torch.manual_seed(123)
np.random.seed(123)
num_workers = 1

try:
    a = CINIC10
except:
    CINIC10 = None

def profile_net(net, op, data_loader, vary_perf, batch_size, curr_file,
                perforation_mode, run_name, prefix, loss_fn):
    n_conv = 0
    if hasattr(net, "_get_perforation"):
        n_conv = len(net._get_perforation())
    for dev in ["cuda", "cpu"]:
        net.train()
        net.to(dev)
        results = {}
        train_accs = []
        losses = []
        loss_fn = loss_fn.to(dev)
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ], profile_memory=True,
        ) as p:
            for i, (batch, classes) in enumerate(data_loader):
                if vary_perf is not None and n_conv > 0 and type(perforation_mode[0]) == str:
                    perfs = get_perfs(perforation_mode[0], n_conv)
                    net._set_perforation(perfs)
                    # net._reset()

                batch = batch.to(dev)
                classes = classes.to(dev)
                pred = net(batch)
                if pred.device != classes.device:
                    loss = loss_fn(pred.cpu(), classes.cpu())
                else:
                    loss = loss_fn(pred, classes)
                loss.backward()
                losses.append(loss.item())

                op.step()
                op.zero_grad()
                if type(data_loader.dataset) in [torchvision.datasets.CIFAR10, CINIC10, UciHAR]:
                    # print("Should be here")
                    acc = (F.softmax(pred.detach(), dim=1).argmax(dim=1) == classes).cpu()
                    train_accs.append(torch.sum(acc) / batch_size)
                else:
                    calculate_segmentation_metrics(classes, pred, run_name, metrics, dev, results)
                    acc = torch.mean(torch.tensor(results[f"{run_name}/iou/weeds"]))
                    train_accs.append(acc)
                break

        with open(f"./{prefix}/{curr_file}_{dev}.txt", "w") as ff:
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1), file=ff)
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))


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
        if "ucihar" in data:
            tf.extend([transforms.RandomHorizontalFlip()])
        else:
            if augment:
                tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
                                                    transforms.RandomResizedCrop(size=32, scale=(0.85, 1.))]),
                           transforms.RandomHorizontalFlip()])
    else:
        if augment:
            tf.extend([transforms.RandomResizedCrop(size=image_resolution[0], scale=(0.5, 1.)),
                       transforms.RandomHorizontalFlip()])
    if data == "cinic" and not noCinic:
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
            num_workers=num_workers, generator=g, )

        valid = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=tf), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, generator=g, )
    elif "agri" in data:
        print(image_resolution)
        # tf.append(transforms.RandomRotation(45)) #BECAUSE FOR SEGMENTATION GIANT BLACK ANGLES KILL PERFORMANCE!!!!!!!
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
                                                         num_workers=num_workers, generator=g, ), \
            torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                        generator=g, ), \
            torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                        generator=g, )
    elif data == "ucihar" or ("uci" in data.lower() and "har" in data.lower()):
        # UCIHAR is already normalised - kinda
        # "but we can still do it"
        # tf = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        tf = []
        tf.append(transforms.Normalize([-4.1705e-04, -9.0756e-05, 3.3419e-01],
                                       [0.1507, 0.3648, 0.5357]))

        tf = transforms.Compose(tf)

        train = torch.utils.data.DataLoader(UciHAR("train", transform=tf), batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, generator=g, )
        valid = torch.utils.data.DataLoader(UciHAR("test", transform=tf), batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, generator=g, )
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

def get_first_layer_weights(net):
    for sub in net.children():
        if hasattr(sub, "weight"):
            return sub.weight.grad.detach().clone().cpu()
        elif len(list(sub.children())) != 0:
            return get_first_layer_weights(sub)
def train(net, op, data_loader, device, loss_fn, vary_perf, batch_size, perforation_type, run_name, grad_clip,
          perforation_mode, n_conv, grad_ep):
    net.train()
    results = {}
    train_accs = []
    losses = []
    # entropies = 0
    grads = []
    timet = 0
    haveCuda = torch.cuda.is_available()
    if haveCuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for i, (batch, classes) in enumerate(data_loader):
        if vary_perf is not None and n_conv > 0 and type(perforation_mode[0]) == str:
            perfs = get_perfs(perforation_mode[0], n_conv)
            net._set_perforation(perfs)
            # net._reset()

        batch = batch.to(device)
        classes = classes.to(device)
        pred = net(batch)
        loss = loss_fn(pred, classes)
        if torch.isnan(loss):
            print(f"NaN loss reached in batch {i}, quitting...")
            quit()
        loss.backward()
        if grad_ep:
            if haveCuda:
                torch.cuda.synchronize()
            t1 = time.time()
            timet += (t1 - t0)
            grads.append(get_first_layer_weights(net))
            if haveCuda:
                torch.cuda.synchronize()
            t0 = time.time()
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
    if haveCuda:
        torch.cuda.synchronize()
    t1 = time.time()
    timet += (t1 - t0)
    for metric in results:
        results[metric] = torch.mean(torch.tensor(results[metric]))
    return losses, train_accs, results, timet, grads


def validate(net, valid_loader, device, loss_fn, file, eval_mode, batch_size, reporting, run_name, dataset):

    valid_losses = []
    valid_accs = []
    results = {}
    sz = 6 if dataset == "ucihar" else 10
    conf = torch.zeros((sz, sz), device=device)
    # ep_valid_losses = []
    net.eval()
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
                calculate_segmentation_metrics(classes, pred, run_name, metrics, device, results)
                acc = torch.mean(torch.tensor(results[f"{run_name}/iou/weeds"]))
                valid_accs.append(acc.detach().cpu())

            else:
                for s, c in zip(softm.argmax(dim=1), classes):
                    conf[s, c] += 1

                acc = (softm.argmax(dim=1) == classes)
                valid_accs.append(torch.sum(acc).detach().cpu() / batch_size)
            if torch.isnan(valid_losses[-1]):
                print(f"NaN loss reached in batch {i}, quitting...")
                quit()
        if reporting:
            if file is not None:
                print(f"Epoch mean acc: {np.mean(valid_accs).item()}, loss: {np.mean(valid_losses).item()}", file=file)
            print(f"Epoch mean acc: {np.mean(valid_accs).item()}, loss: {np.mean(valid_losses).item()}")
        # ep_valid_losses.append(l2.item() / (i + 1))


    return valid_losses, valid_accs, results, conf.detach().cpu()



def benchmark(net, op, scheduler=None, loss_function=torch.nn.CrossEntropyLoss(), run_name="test",
              perforation_mode=(2, 2), perforation_type="perf",
              train_loader=None, valid_loader=None, test_loader=None, max_epochs=1, in_size=(2, 3, 32, 32),
              summarise=True, pretrained=True, dataset="idk", prefix="", do_grad=False, savemodels=False,
              device="cpu", batch_size=2, reporting=True, file=None, grad_clip=None, eval_modes=(None,)):
    if type(perforation_mode) not in [tuple, list]:
        perforation_mode = (perforation_mode, perforation_mode)
    if type(perforation_mode[0]) == str:
        vary_perf = True
    else:
        vary_perf = None

    if eval_modes is None:
        eval_modes = (None,)
    if hasattr(net, "_get_perforation"):
        n_conv = len(net._get_perforation())
    else:
        n_conv = 0
        eval_modes = [None,]
    # net._reset()
    print(f"starting run {run_name}...")
    timeElapsed = 0
    confs = []
    if summarise:
        summary(net, input_size=in_size)
    best_valid_losses = [999] * len(eval_modes)
    best_models = [None] * len(eval_modes)
    grad_ep = False
    for epoch in range(max_epochs):
        if do_grad:
            grad_ep = (epoch + 1) in [1,2,3,5,10,20,50,100,200, max_epochs]
        if reporting:
            if file is not None:
                print(f"\nEpoch {epoch} training:", file=file)
            print(f"\nEpoch {epoch} training:")
        losses, train_accs, results, timet, grads = train(net, op, train_loader, device, loss_fn=loss_function,
                                            perforation_mode=perforation_mode, grad_ep=grad_ep,
                                            batch_size=batch_size, perforation_type=perforation_type, run_name=run_name,
                                            grad_clip=grad_clip, vary_perf=vary_perf, n_conv=n_conv)
        if len(grads) != 0:
            grads = torch.stack(grads)
            n_bins = 500
            y, x = torch.histogram(grads, bins=n_bins)
            x = ((x + x.roll(-1)) * 0.5)[:-1] #calc bin centers
            plt.bar(x, y, label="Gradient magnitude distribution", width=(x[-1] - x[0]) / (n_bins-1))
            plt.xlabel("Bin limits")
            plt.ylabel("Num. of gradient values")
            plt.yscale("log")
            plt.title(f"Gradient distribution, Epoch {epoch}")
            plt.tight_layout()
            newpath = f"./{prefix}/{run_name}_grad_imgs/"
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            plt.savefig(os.path.join(newpath, f"grad_hist_e{epoch}.png"))
            plt.clf()
        timedelta = int(timet* 1000) / 1000
        timeElapsed += timet
        if reporting:
            if file is not None:
                print(f"Average Epoch {epoch} Train Loss:", np.mean(losses).item(), file=file)
                print(f"Epoch mean acc: {np.mean(train_accs).item()}, Epoch time: {timedelta} s", file=file)
                print(results, file=file)
            print(f"Average Epoch {epoch} Train Loss:", np.mean(losses).item())
            print(f"Epoch mean acc: {np.mean(train_accs).item()}, Epoch time: {timedelta} s")
            if results != {}:
                print(results)
        train_mode = None
        if hasattr(net, "_get_perforation"):
            train_mode = net._get_perforation()
        for ind, mode in enumerate(eval_modes):

            print("\ntesting eval mode", mode)
            if file is not None:
                print("\ntesting eval mode", mode, file=file)

            valid_losses, valid_accs, allMetrics, ims = validate(net=net, valid_loader=valid_loader, device=device,
                                                                 loss_fn=loss_function,
                                                                 file=file, batch_size=batch_size, eval_mode=mode,
                                                                 run_name=run_name, reporting=reporting,
                                                                 dataset=dataset)
            curr_loss = np.mean(valid_losses)
            if curr_loss < best_valid_losses[ind]:
                best_valid_losses[ind] = curr_loss
                best_models[ind] = copy.deepcopy(net.state_dict())
        if train_mode is not None:
            # print("returning to train mode", train_mode)
            net._set_perforation(train_mode)
            # print(train_mode, flush=True)
            # net._reset()
        if scheduler is not None:
            scheduler.step()
            print(", Current LR:", scheduler.get_last_lr()[0])

    if test_loader is None:
        test_loader = valid_loader
    best_outputs = []
    if eval_modes is None:
        eval_modes = (None,)
    metrics = []
    allMetrics = {}
    for ind, mode in enumerate(eval_modes):
        #net.eval()
        if best_models[ind] is not None:
            net.load_state_dict(best_models[ind])
        if not os.path.exists(f"./{prefix}/models"):
            os.makedirs(f"./{prefix}/models")
        #Save models for evaluation purposes
        if savemodels:
            torch.save(best_models[ind], f"./{prefix}/models/{run_name}.model")
        #net.eval()
        print("\nValidating eval mode", mode)
        test_losses, test_accs, allMetrics, conf = validate(net=net, valid_loader=test_loader, device=device,
                                                            loss_fn=loss_function,
                                                            file=file, batch_size=batch_size, eval_mode=mode,
                                                            reporting=reporting, run_name=run_name, dataset=dataset)
        metrics.append(allMetrics)
        confs.append(conf)
        h = f"Validation loss ({mode}):" + str(np.mean(test_losses))
        print(h)
        best_outputs.append(h)
        h2 = f"Validation acc ({mode}):" + str(np.mean(test_accs))
        best_outputs.append(h2)
        print(h2)
    h3 = "Training time:" + str(timeElapsed) + " seconds"
    print("Training time:", timeElapsed, " seconds")
    best_outputs.append(h3)
    return best_outputs, confs, allMetrics


def runAllTests():
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    architectures = [
        [[(resnet18, "resnet18"), (mobilenet_v2, "mobnetv2"), (mobilenet_v3_small, "mobnetv3s")], ["cifar", "ucihar"],
         [32]],
        [[(UNet, "unet_agri"), (UNetCustom, "unet_custom")], ["agri"], [128, 256, 512]],

        # "cinic" takes too long to run, ~45sec per epoch compared to ~9 for cifar ,so it would be about 2 hour training per config, maybe later

    ]

    for version in architectures:  # classigication, segmetnationg
        for dataset in version[1]:

            loss_fn = torch.nn.CrossEntropyLoss()
            if dataset == "agri":
                loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9])).to(device)

                lr = 0.5
                max_epochs = 300
                batch_size = 32
            elif dataset == "ucihar":
                max_epochs = 100
                batch_size = 32
                lr = 0.01
            else:
                max_epochs = 200
                batch_size = 32
                lr = 0.1

            for model, modelname in version[0]:
                for img in version[2]:
                    # model = UNet
                    # img = 128
                    # dataset = "agri"
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

                        for perf_type in ["perf", "dau",]:
                            if perforation is None:
                                if perf_type == "dau":
                                    continue
                                else:
                                    perf_type = None

                            prefix = "allTests_last"
                            prefix = "testFixPerf"

                            name = f"{modelname}_{dataset}_{img}_{perforation}_{perf_type}"
                            curr_file = f"{name}"

                            if "agri" in dataset:
                                net = model(2).to(device)
                            else:
                                sz = 6 if dataset == "ucihar" else 10
                                net = model(num_classes=sz).to(device)


                            pretrained = True #keep default network init
                            if perf[0] is not None:  # do we want to perforate? # Is the net not already perforated?
                                if type(perf[0]) != str:  # is perf mode not a string
                                    if "dau" in perf_type.lower():
                                        perfDAU(net, in_size=in_size, perforation_mode=perf,
                                                pretrained=pretrained)
                                    elif "perf" in perf_type.lower():
                                        perfPerf(net, in_size=in_size, perforation_mode=perf,
                                                 pretrained=pretrained)
                                else:  # it is a string
                                    if "dau" in perf_type.lower():
                                        perfDAU(net, in_size=in_size, perforation_mode=(2, 2),
                                                pretrained=pretrained)
                                    elif "perf" in perf_type.lower():
                                        perfPerf(net, in_size=in_size, perforation_mode=(2, 2),
                                                 pretrained=pretrained)
                            else:
                                print("Perforating base net for noperf training...")
                                perfPerf(net, in_size=in_size, perforation_mode=(2,2), pretrained=pretrained)
                                net._set_perforation((1,1))


                            if perforation == 2:
                                eval_modes = [(1, 1), (2, 2), (3, 3), (4, 4)]
                            elif perforation == 3:
                                eval_modes = [(1, 1), (2, 2), (3, 3), (4, 4)]
                            elif perforation is None:
                                eval_modes = [(1, 1), (2, 2), (3, 3), (4, 4)]
                            else:
                                eval_modes = [(1, 1), (2, 2), (3, 3), (4, 4)]

                            # continue
                            if hasattr(net, "_reset"):
                                net._reset()
                            net.to(device)



                            #TODO check why agriadapt had no gradient graphs
                            op = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0005)
                            train_loader, valid_loader, test_loader = get_datasets(dataset, batch_size, True,
                                                                                   image_resolution=img_res)
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=max_epochs)

                            run_name = f"{curr_file}"
                            if not os.path.exists(f"./{prefix}/"):
                                os.mkdir(f"./{prefix}")
                            if not os.path.exists(f"./{prefix}/imgs"):
                                os.mkdir(f"./{prefix}/imgs")
                            print("starting run:", curr_file)
                            cpuRun = False
                            if os.path.exists(f"./{prefix}/{curr_file}_best.txt") or cpuRun:
                                print("Cuda run (for accuracy performance) exists, running cpu speedtest...")
                                pref = prefix + "/cpu"
                                if not os.path.exists(f"./{pref}"):
                                    os.makedirs(f"./{pref}")
                                batch_size = 2
                                max_epochs = 10
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=max_epochs)
                                device = "cpu"
                                net.to(device)
                                loss_fn.to(device)
                                if os.path.exists(f"./{pref}/{curr_file}_best.txt") and not cpuRun:
                                    print("cpu speedtest exists, next")
                                    continue
                                with open(f"./{pref}/{curr_file}.txt", "w") as f:
                                    best_out, confs, metrics = benchmark(net, op, scheduler, train_loader=train_loader,
                                                                         valid_loader=valid_loader,
                                                                         test_loader=test_loader,
                                                                         max_epochs=max_epochs, device=device,
                                                                         perforation_mode=perf,
                                                                         run_name=run_name, batch_size=batch_size,
                                                                         loss_function=loss_fn, prefix=pref,
                                                                         eval_modes=eval_modes, in_size=in_size,
                                                                         dataset=dataset,
                                                                         perforation_type=perf_type, file=f,
                                                                         summarise=False)


                                    with open(f"./{pref}/{curr_file}_best.txt", "w") as ff:
                                        print(best_out, file=ff)
                                    if cpuRun:continue
                                    print("Now running profiling...")
                                    if type(perforation) == str:
                                        vary_perf = True
                                    else:
                                        vary_perf = None
                                    pref = prefix + "/profiling"
                                    if os.path.exists(f"./{pref}/{curr_file}_cpu.txt"):
                                        print("profiling also exists, next...")
                                        continue
                                    if not os.path.exists(pref):
                                        os.makedirs(pref)
                                    profile_net(net, op, data_loader=train_loader, vary_perf=vary_perf,
                                                batch_size=batch_size, curr_file=curr_file, perforation_mode=perf,
                                                run_name=run_name, prefix=pref, loss_fn=loss_fn)
                                continue #skip already processed configurations

                            print("net:", modelname)
                            print("Dataset:", dataset)
                            print(max_epochs, "epochs")
                            print("perforation mode", perf)
                            print("perforation type:", perf_type)
                            print("batch_size:", batch_size)
                            print("loss fn", loss_fn)
                            print("eval modes", eval_modes)
                            print("Learning rate:", lr)
                            print("run name:", curr_file)
                            with open(f"./{prefix}/{curr_file}.txt", "w") as f:
                                best_out, confs, metrics = benchmark(net, op, scheduler, train_loader=train_loader,
                                                            valid_loader=valid_loader, test_loader=test_loader,
                                                            max_epochs=max_epochs, device=device, perforation_mode=perf,
                                                            run_name=run_name, batch_size=batch_size, savemodels=True,
                                                            loss_function=loss_fn,prefix=prefix,do_grad=True,
                                                            eval_modes=eval_modes, in_size=in_size, dataset=dataset,
                                                            perforation_type=perf_type, file=f, summarise=False)

                                if not "agri" in dataset:
                                    n_samp = len((test_loader if test_loader is not None else valid_loader).dataset)
                                    fig, ax = plt.subplots(len(confs), 1,
                                                           figsize=(5, 20) if len(confs) != 1 else (6, 5))
                                    if len(confs) == 1:
                                        ax = [ax]
                                        mins, maxs = confs[0].min().item(), confs[0].max().item()
                                    # print(mins, maxs, n_samp, confs)
                                    for i, conf in enumerate(confs):
                                        mins, maxs = conf.min().item() / n_samp, conf.max().item() / n_samp
                                        imlast = ax[i].imshow((conf / n_samp) * 100, vmin=mins * 100, vmax=maxs * 100)

                                        ax[i].set_title(f"Perforation mode {eval_modes[i]}")
                                        fig.subplots_adjust(right=0.94, top=0.93 - (3 - i) * 0.05, bottom=0.2)
                                        cbar_ax = fig.add_axes([0.86, 0.2 * (i + 1) + i * -0.01, 0.02, 0.16])
                                        plt.colorbar(imlast, cax=cbar_ax, ticks=[mins * 100, maxs * 100])
                                        #ax[0].set_title("No perforation")

                                        if dataset != "ucihar":
                                            ax[i].set_yticks(list(range(10)),
                                                             ["airplane", "automobile", "bird", "cat", "deer", "dog",
                                                              "frog", "horse", "ship", "truck"])
                                        else:
                                            ax[i].set_yticks(list(range(6)),
                                                             ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING",
                                                              "STANDING", "LAYING"])
                                        ax[i].set_ylabel("Predicted")
                                        ax[i].set_xlabel("True")
                                    if dataset != "ucihar":
                                        ax[-1].set_xticks(list(range(10)),
                                                          ["airplane", "automobile", "bird", "cat", "deer", "dog",
                                                           "frog",
                                                           "horse", "ship", "truck"], rotation=90)
                                    else:
                                        ax[-1].set_yticks(list(range(6)),
                                                          ["WALKING", "UPSTAIRS", "DOWNSTAIRS",
                                                           "SITTING", "STANDING", "LAYING"])
                                    # add space for colour bar
                                    if len(confs) == 1:
                                        fig.subplots_adjust(right=0.85, top=0.9, bottom=0.24)
                                        cbar_ax = fig.add_axes([0.80, 0.20, 0.04, 0.7])
                                        plt.colorbar(imlast, cax=cbar_ax, ticks=[mins * 100, maxs * 100])

                                    plt.savefig(f"./{prefix}/imgs/{curr_file}.png")
                                    plt.clf()

                                    with open(f"./{prefix}/{curr_file}_confs.txt", "w") as fc:
                                        print(confs, eval_modes, file=fc)
                                        print(confs, eval_modes)
                                        print(n_samp)
                                else:
                                    with open(f"./{prefix}/{curr_file}_confs.txt", "w") as fc:
                                        print(metrics, file=fc)
                            with open(f"./{prefix}/{curr_file}_best.txt", "w") as ff:
                                print(best_out, file=ff)


if __name__ == "__main__":
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    runAllTests()
    quit()
