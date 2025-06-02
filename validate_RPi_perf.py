import os

import numpy as np
import torch
import copy
from torch import argmax, where, cat, stack
import matplotlib.pyplot as plt
perf = True
if perf:
    from perforateCustomNet import perforate_net_perfconv, perforate_net_downActivUp


seed = 123
g = torch.Generator(device="cpu")
g.manual_seed(123)
torch.manual_seed(123)
np.random.seed(123)
num_workers = 1
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):

        return self.conv1(x)

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



if __name__ == "__main__":
    compareGrads = False
    if compareGrads:
        #If this doesnt work, we run the entire run for one network on RPi
        for perforation in [None, "DAU", "perf"]:
            net = SimpleNet()
            if perforation == "DAU":
                perforate_net_downActivUp(net, in_size=(2,3,9,9))
            elif perforation == "perf":
                perforate_net_perfconv(net, in_size=(2,3,9,9))
            net.conv1.weight = torch.nn.Parameter(torch.ones_like(net.conv1.weight, dtype=torch.float) / (3 * 2 * 3 * 3))
            net.conv1.bias = torch.nn.Parameter(torch.ones_like(net.conv1.bias, dtype=torch.float) / (3*3))
            data = torch.ones((2, 3, 9, 9), dtype=torch.float)
            data[:, 0, ::2, :] = 1
            data[:, 0, 1::2, :] = 2
            data[:, 0, :, ::2] = 2
            data[:, 0, :, 1::2] = 3

            data[:, 1, ::2, :] = 0
            data[:, 1, 1::2, :] = 3
            data[:, 1, :, ::2] = 1
            data[:, 1, :, 1::2] = 2

            data[:, 2, ::2, :] = -1
            data[:, 2, 1::2, :] = -2
            data[:, 2, :, ::2] = -2
            data[:, 2, :, 1::2] = -3

            data.requires_grad = True

            dataCompare = torch.ones((2, 3, 9, 9), dtype=torch.float)
            dataCompare[:, 0, ::2, :] = 0.8
            dataCompare[:, 0, 1::2, :] = 0.2
            dataCompare[:, 0, :, ::2] = -0.4
            dataCompare[:, 0, :, 1::2] = 0.3


            dataCompare[:, 1, ::2, :] = 1
            dataCompare[:, 1, 1::2, :] = -0.2
            dataCompare[:, 1, :, ::2] = -0.4
            dataCompare[:, 1, :, 1::2] = 0.5

            dataCompare[:, 2, ::2, :] = 1
            dataCompare[:, 2, 1::2, :] = 0.2
            dataCompare[:, 2, :, ::2] = 0.4
            dataCompare[:, 2, :, 1::2] = 0.5

            dataCompare[:, 1, :6, :5] += 0.5
            #loss_fn = torch.nn.MSELoss()
            loss_fn = torch.nn.CrossEntropyLoss()

            res = net(data)
            suma = loss_fn(res, dataCompare)
            suma.backward()
            verbose = False
            if verbose:
                print(res)
                print("\n---\n")
                print(suma)
                print(net.conv1.weight.grad)
                print(data.grad * 1000)
                print(net.conv1.weight.shape)

            base = f"./RpiComparison/{perforation}/"
            if not os.path.exists(base):
                os.makedirs(base)

            if os.path.exists(f"{base}weightGrad.bin"):
                weightGradCheck = torch.tensor(np.fromfile(f"{base}weightGrad.bin", dtype=np.float32)).view(net.conv1.weight.grad.shape)
                dataGradCheck = torch.tensor(np.fromfile(f"{base}dataGrad.bin", dtype=np.float32)).view(data.grad.shape)
                resCheck = torch.tensor(np.fromfile(f"{base}res.bin", dtype=np.float32)).view(res.shape)

                print("Checking weightgrad similarity...")
                print(weightGradCheck - net.conv1.weight.grad)
                print((weightGradCheck - net.conv1.weight.grad).abs().mean(), "abs mean weightgrad diff, Checking datagrad similarity...")
                print(dataGradCheck - data.grad)
                print((dataGradCheck - data.grad).abs().mean(), "abs mean datagrad diff, Checking result similarity...")
                print(resCheck - res)
                print((resCheck - res).abs().mean(), "abs mean res diff")
            else:
                print("No existing compare files found, skipping...")

            if not os.path.exists(f"{base}weightGrad.bin"):
                net.conv1.weight.grad.detach().numpy().tofile(f"{base}weightGrad.bin")
                data.grad.detach().numpy().tofile(f"{base}dataGrad.bin")
                res.detach().numpy().tofile(f"{base}res.bin")





            fig, ax = plt.subplots(1, 5)
            ax[0].imshow(res.detach()[0].transpose(0, 1).transpose(1, 2))
            #ax[1].imshow(res.grad.detach()[0].transpose(0, 1).transpose(1, 2))
            ax[1].imshow((1 + net.conv1.weight.grad.detach()[0])/2)
            ax[2].imshow((1 + net.conv1.weight.grad.detach()[1])/2)
            ax[3].imshow((1 + net.conv1.weight.grad.detach()[2])/2)
            ax[-1].imshow(data.grad.detach()[0].transpose(0, 1).transpose(1, 2)*1000 + 0.2)
            plt.savefig(f"{base}CompareGrads.png")
            plt.show()

    doEntireNetworkRun = True
    if doEntireNetworkRun:
        from agriadapt.segmentation.models.UNet import UNet as UNetAgri
        import agriadapt.segmentation.data.data as dd
        net = UNetAgri(2)
        if perf:
            from Architectures.UnetCustom import UNet
            net = UNet(2)
            perforate_net_downActivUp(net, in_size=(2, 3, 128, 128))
            net._set_perforation((3,3))
            net._reset()

        def get_datasets(data, batch_size, augment=True, image_resolution=None):
            try:from torchvision.transforms import v2 as transforms
            except:from torchvision import transforms

            class NormalizeImageOnly(torch.nn.Module):
                def __init__(self, means, stds):
                    super().__init__()
                    self.norm = transforms.Normalize(means, stds)

                def forward(self, img, classes=None):
                    if classes is None:
                        classes = img[3:]
                        return self.norm(img[:3]), classes
                    # Do some transformations
                    # print(img.shape,flush=True)
                    # print(classes.shape,flush=True)
                    # print("----", flush=True)
                    return self.norm(img), classes

            test = None
            try:
                tf = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
            except:
                tf = []
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
            return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                             num_workers=num_workers, generator=g, ), \
                torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            generator=g, ), \
                torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            generator=g, )
        #from benchmarking import get_datasets, calculate_segmentation_metrics

        from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score
        metrics = {
            "iou": BinaryJaccardIndex,
            "precision": BinaryPrecision,
            "recall": BinaryRecall,
            "f1score": BinaryF1Score,
        }
        for op in [torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=0.1),torch.optim.SGD(net.parameters(), lr=0.5, weight_decay=0.0005)]:
            bs = 16
            device = "cuda" if torch.cuda.is_available() else "cpu"
            epochs = 100
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9])).to(device)
            net.to(device)
            #op =torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.1)
            #op = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0005)
            #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op,T_max=epochs)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(op,0.99)
            train_loader, valid_loader, test_loader = get_datasets("agri", bs, True, (128,128))
            import time
            min_loss = 99999
            saveModel = None
            for ep in range(epochs):
                losses = []
                results = {}
                t1 = time.time()
                for i, (batch, classes) in enumerate(train_loader):
                    batch = batch.to(device)
                    classes = classes.to(device)
                    pred = net(batch)
                    loss = loss_fn(pred, classes)
                    loss.backward()
                    op.step()
                    op.zero_grad()
                    #lr_scheduler.step()
                    losses.append(loss.item())
                    calculate_segmentation_metrics(classes, pred, "test", metrics, device, results)
                t2 = time.time()
                print(torch.mean(torch.tensor(results[f"test/iou/weeds"])), "avg IoU", torch.mean(torch.tensor(losses)), "Avg loss")
                print((t2 - t1), "seconds elapsed for epoch", ep)
                #quit()
                if ep % 10 == 9 or ep > epochs*0.5:
                    net.eval()
                    losses = []
                    results = {}
                    for i, (batch, classes) in enumerate(valid_loader):
                        batch = batch.to(device)
                        classes = classes.to(device)
                        pred = net(batch)
                        loss = loss_fn(pred, classes)
                        lr_scheduler.step()
                        losses.append(loss.item())
                        calculate_segmentation_metrics(classes, pred, "test", metrics, device, results)
                    mean_loss = torch.mean(torch.tensor(losses))
                    if mean_loss < min_loss:
                        min_loss = mean_loss
                        saveModel = copy.deepcopy(net.state_dict())
                    print(torch.mean(torch.tensor(results[f"test/iou/weeds"])), "avg TEST IoU", torch.mean(torch.tensor(losses)), "Avg loss")
                    print("Current LR:", lr_scheduler.get_last_lr())
                    net.train()

            net.load_state_dict(saveModel)
            net.eval()
            losses = []
            results = {}
            for i, (batch, classes) in enumerate(test_loader):
                batch = batch.to(device)
                classes = classes.to(device)
                pred = net(batch)
                loss = loss_fn(pred, classes)
                lr_scheduler.step()
                losses.append(loss.item())
                calculate_segmentation_metrics(classes, pred, "test", metrics, device, results)
            mean_loss = torch.mean(torch.tensor(losses))

            print(torch.mean(torch.tensor(results[f"test/iou/weeds"])), "avg IoU", torch.mean(torch.tensor(losses)), "Avg loss")