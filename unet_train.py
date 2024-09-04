import os
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
from torchinfo import summary
#import wandb as wandb
#from fvcore.nn import FlopCountAnalysis, parameter_count
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
#from ptflops import get_model_complexity_info
#import thop
#import pthflops
import sys
import matplotlib
import cv2
import random
import numpy as np
extrapath = '/mnt/c/Users/timotej/pytorch/PyTorch-extension-Convolution/conv_cuda/agriadapt'
sys.path.append(extrapath)
import segmentation.settings as settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise
from segmentation.models.slim_squeeze_unet import (
    SlimSqueezeUNet,
    SlimSqueezeUNetCofly,
)
from segmentation.models.slim_unet import SlimUNet
from agriadapt.dl_scripts.UNet import UNet
from Architectures.UNetPerf import UNet as UNetPerf
from Architectures.UNetDAU import UNet as UNetDAU
from UnetCustom import UNet as UNetCustom
from perforateCustomNet import perforate_net_perfconv, perforate_net_downActivUp


class Training:
    def __init__(
        self,
        device,
        architecture=settings.MODEL,
        epochs=settings.EPOCHS,
        learning_rate=settings.LEARNING_RATE,
        learning_rate_scheduler=settings.LEARNING_RATE_SCHEDULER,
        batch_size=settings.BATCH_SIZE,
        regularisation_l2=settings.REGULARISATION_L2,
        image_resolution=settings.IMAGE_RESOLUTION,
        widths=settings.WIDTHS,
        dropout=settings.DROPOUT,
        verbose=1,
        wandb_group=None,
        dataset="infest",
        continue_model="",  # This is set to model name that we want to continue training with (fresh training if "")
        sample=0,
            save_model=False, #do we want to save checkpoints
    ):
        self.architecture = architecture
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.batch_size = batch_size
        self.regularisation_l2 = regularisation_l2
        self.image_resolution = image_resolution
        self.widths = widths
        self.dropout = dropout
        self.verbose = verbose
        self.wandb_group = wandb_group
        self.dataset = dataset
        self.continue_model = continue_model
        self.sample = sample
        self.save_model = save_model
        self.allVals = None

        self.best_fitting = [0, 0, 0, 0]

    def _report_settings(self):
        print("=======================================")
        print("Training with the following parameters:")
        print("Dataset: {}".format(self.dataset))
        print("Model architecture: {}".format(self.architecture))
        print("Epochs: {}".format(self.epochs))
        print("Learning rate: {}".format(self.learning_rate))
        print("Learning rate scheduler: {}".format(self.learning_rate_scheduler))
        print("Batch size: {}".format(self.batch_size))
        print("L2 regularisation: {}".format(self.regularisation_l2))
        print("Image resolution: {}".format(self.image_resolution))
        print("Dropout: {}".format(self.dropout))
        print("Network widths: {}".format(self.widths))
        print("Loss function weights: {}".format(settings.LOSS_WEIGHTS))
        print(
            "Transfer learning model: {}".format(
                self.continue_model if self.continue_model else "None"
            )
        )
        print("=======================================")

#    def _report_model(self, model, input, loader):
#        print("=======================================")
#        for width in self.widths:
#            model.set_width(width)
#            flops = FlopCountAnalysis(model, input)
#            # Flops
#            # Facebook Research
#            # Parameters
#            # Facebook Research
#
#            # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
#            print(sum(p.numel() for p in model.parameters() if p.requires_grad))
#            # https://pypi.org/project/ptflops/
#            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#            print(flops.total(), sum([x for x in parameter_count(model).values()]))
#            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#            print("-----------------------------")
#            print(
#                get_model_complexity_info(
#                    model, (3, 128, 128), print_per_layer_stat=False
#                )
#            )
#            print("-----------------------------")
#            print("*****************************")
#            print(thop.profile(model, (input,)))
#            print("*****************************")
#            print("?????????????????????????????")
#            print(pthflops.count_ops(model, input))
#            print("?????????????????????????????")
#            # print(flops.by_operator())
#            # print(flops.by_module())
#            # print(flops.by_module_and_operator())
#            # print(flop_count_table(flops))
#        print("=======================================")

    def _find_best_fitting(self, metrics):
        """
        Could you perhaps try training it by monitoring the validation scores for each
        width and then stopping the training at the epoch which maximises the difference
        between the widths when they are in the right order?

        Compare current metrics to best fitting and overwrite them if new best
        fitting were found given to a heuristic we have to come up with.

        Return True if best fitting was found, otherwise false.
        """
        allVals = metrics
        metrics = [
            #metrics["iou/valid/25/weeds"],
            #metrics["iou/valid/50/weeds"],
            #metrics["iou/valid/75/weeds"],
            #metrics["iou/valid/100/weeds"],
            metrics["valid/100/iou/weeds"],
        ]

        # First check if the widths are in order.
        for i, m in enumerate(metrics):
            if i == 0:
                continue
            if metrics[i - 1] > m:
                # print("Metrics not in order, returning false.")
                return False

        # Then check if the differences between neighbours are higher than current best
        if sum(
            [self.best_fitting[i] - self.best_fitting[i - 1] for i in range(1, 1)]
        ) > sum([metrics[i] - metrics[i - 1] for i in range(1, 1)]):
            return False

        print()
        print("New best scores:")
        print("All scores:", allVals)
        print(f"Comparing metrics: {metrics}")
        print(f"Current best:      {self.best_fitting}")
        print()
        self.best_fitting = metrics
        self.allVals = allVals
        return True

    def _learning_rate_scheduler(self, optimizer):
        if self.learning_rate_scheduler == "no scheduler":
            return None
        elif self.learning_rate_scheduler == "linear":
            return LinearLR(
                optimizer,
                start_factor=1,
                end_factor=0,
                total_iters=self.epochs,
            )
        elif self.learning_rate_scheduler == "exponential":
            return ExponentialLR(
                optimizer,
                0.99,
            )
        elif self.learning_rate_scheduler == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=self.epochs
            )

    def train(self):
        if self.verbose:
            print("Training process starting...")
            self._report_settings()
        # Prepare the data for training and validation

        from torchvision.transforms import transforms
        tf = []
        #tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
        #                                    transforms.RandomResizedCrop(size=32)
        #                                    ]),
        #           transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=45)])
        tf.append(transforms.Normalize([0.4837999, 0.3109686, 0.38634193, 0, 0],
                                       [0.11667825, 0.08259449, 0.09035885, 1, 1]))
        # tf.append(transforms.Normalize([0.50348324, 0.31018394, 0.39796165],
        #                               [0.11345574, 0.07511874, 0.08323097]))
        tf = transforms.Compose(tf)
        tf = None
        ii = ImageImporter(
            self.dataset,
            validation=True,
            sample=self.sample,
            smaller=self.image_resolution,
        )

        train, validation = ii.get_dataset(tf)
        if self.verbose:
            print("Number of training instances: {}".format(len(train)))
            print("Number of validation instances: {}".format(len(validation)))

        # Wandb report startup
        garage_path = ""

        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validation, batch_size=self.batch_size, shuffle=False)

        # Prepare a weighted loss function
        loss_function = torch.nn.CrossEntropyLoss(
            weight=tensor(settings.LOSS_WEIGHTS).to(self.device)
        )

        # Prepare the model
        out_channels = len(settings.LOSS_WEIGHTS)
        if not self.continue_model:
            if self.architecture == "slim":
                model = SlimUNet(out_channels)
            elif self.architecture == "squeeze":
                model = SlimSqueezeUNet(out_channels)
                if self.dataset == "cofly":
                    model = SlimSqueezeUNetCofly(out_channels)
                # model = SlimPrunedSqueezeUNet(in_channels, dropout=self.dropout)
            else:
                if self.architecture == "unet_perf":
                    model = UNet(out_channels)
                    perforate_net_perfconv(model, perforation_mode=(2,2))
                elif self.architecture == "unet":
                    model = UNet(out_channels)
                elif self.architecture == "unet2":
                    model = UNetPerf(out_channels, perforation_mode=(2,2))
                    #perforate_net_perfconv(model, perforation_mode=(1,1))
                elif self.architecture == "unet_dau":
                    model = UNet(out_channels)#UNetDAU(out_channels)
                    perforate_net_downActivUp(model, perforation_mode=(2,2), in_size=(1,3,self.image_resolution[0], self.image_resolution[1]))
                elif self.architecture == "unet_custom":
                    model = UNetCustom(out_channels)
                    #perforate_net_downActivUp(model, perforation_mode=(2,2), in_size=(1,3,self.image_resolution[0], self.image_resolution[1]))
                    #perforate_net_perfconv(model, perforation_mode=(2,2), in_size=(1,3,self.image_resolution[0], self.image_resolution[1]))
                elif self.architecture == "unet_custom_dau":
                    model = UNetCustom(out_channels)
                    perforate_net_downActivUp(model, perforation_mode=(2,2), in_size=(1,3,self.image_resolution[0], self.image_resolution[1]))
                    #perforate_net_perfconv(model, perforation_mode=(2,2), in_size=(1,3,self.image_resolution[0], self.image_resolution[1]))
                elif self.architecture == "unet_custom_perf":
                    model = UNetCustom(out_channels)
                    # perforate_net_downActivUp(model, perforation_mode=(2,2), in_size=(1,3,self.image_resolution[0], self.image_resolution[1]))
                    perforate_net_perfconv(model, perforation_mode=(2,2), in_size=(1,3,self.image_resolution[0], self.image_resolution[1]))
                else:
                    raise ValueError("Unknown model architecture.")
        else:
            model = torch.load(
                Path(settings.PROJECT_DIR)
                / "segmentation/training/garage/"
                / self.continue_model
            )
        summary(model, input_size=(self.batch_size, 3, self.image_resolution[0], self.image_resolution[1]))
        # summary(model, input_size=(in_channels, 128, 128))
        model.to(self.device)

        # Prepare the optimiser
        optimizer = Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularisation_l2,
        )
        scheduler = self._learning_rate_scheduler(optimizer)

        for epoch in range(self.epochs):
            s = datetime.now()

            model.train()
            for X, y in train_loader:
                # Move to GPU
                X, y = X.to(self.device), y.to(self.device)
                # Reset optimiser
                optimizer.zero_grad()
                outputs = model.forward(X)
                # Calculate loss function
                loss = loss_function(outputs, y)
                # Backward pass
                loss.backward()
                # Update weights
                optimizer.step()

            model.eval()
            with torch.no_grad():
                metrics = Metricise(device=self.device)
                ind_worst, img_worst, ind_best, img_best = metrics.evaluate(
                    model, train_loader, "train", epoch, loss_function=loss_function
                )
                #cv2.destroyAllWindows()
                #print(
                #    torch.nn.functional.upsample(train_loader.dataset[ind_worst][0].unsqueeze(0), scale_factor=(5,5)).squeeze().movedim(0,-1)[:, :, [1,2,0]].shape,
                #    torch.tile(torch.nn.functional.upsample(train_loader.dataset[ind_worst][1].unsqueeze(0), scale_factor=(5,5)).squeeze().movedim(0,-1)[:, :, 0].unsqueeze(-1), (1, 1, 3)).shape,
                #    img_worst[0].shape,
                #    img_worst.cpu().detach().numpy().shape)
                load = train_loader
                sz = load.dataset[ind_best][0].shape[-1]
                scale = 512/sz
                scale = (scale, scale)
                img = torch.cat((
                                        torch.cat((
                    torch.nn.functional.upsample(load.dataset[ind_worst][0].unsqueeze(0), scale_factor=scale).squeeze().movedim(0,-1)[:, :, [1,2,0]],
                    torch.tile(torch.nn.functional.upsample(load.dataset[ind_worst][1].unsqueeze(0), scale_factor=scale).squeeze().movedim(0,-1)[:, :, 0].unsqueeze(-1), (1, 1, 3)),
                    torch.tile(torch.nn.functional.upsample(img_worst[0].unsqueeze(0).unsqueeze(0), scale_factor=scale).squeeze(), (3, 1, 1)).movedim(0,-1),

                                        ), dim=1),
                                        torch.cat((
                    torch.nn.functional.upsample(load.dataset[ind_best][0].unsqueeze(0), scale_factor=scale).squeeze().movedim(0, -1)[:, :, [1, 2, 0]],
                    torch.tile(torch.nn.functional.upsample(load.dataset[ind_best][1].unsqueeze(0), scale_factor=scale).squeeze().movedim(0, -1)[:, :, 0].unsqueeze(-1), (1, 1, 3)),
                    torch.tile(torch.nn.functional.upsample(img_best[0].unsqueeze(0).unsqueeze(0), scale_factor=scale).squeeze(), (3, 1, 1)).movedim(0, -1),

                                        ), dim=1),

                                    #TODO torch.nn.functional.upsample(...)#TODO
                                    ),dim=0).numpy()
                if extrapath.startswith("/mnt"):
                    cv2.imshow("Window", img)
                    cv2.waitKey(10)
                metrics.evaluate(
                    model,
                    valid_loader,
                    "valid",
                    epoch,
                    loss_function=loss_function,
                    image_pred=epoch % 50 == 0,
                )
                if epoch == settings.EPOCHS - 1:
                    dirname = "unet_imgs"
                    if not os.path.isdir(f"./{dirname}"):
                        os.mkdir(f"./{dirname}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.savefig(f"./{dirname}/{self.architecture}{self.image_resolution[0]}.png")
                    plt.clf()
                    plt.cla()
            if self.learning_rate_scheduler == "no scheduler":
                metrics.add_static_value(self.learning_rate, "learning_rate")
            else:
                metrics.add_static_value(scheduler.get_last_lr(), "learning_rate")
                scheduler.step()

            res = metrics.report()
            #print(res)
            #print(res["valid/100/iou/weeds"])
            # Only save the model if it is best fitting so far
            # The beginning of the training is quite erratic, therefore, we only consider models from epoch 50 onwards
            if epoch > 50 and epoch % 20 == 0:
                with open("./outputs_unet.txt", "a") as gg:
                    print(self.architecture, file=gg)
                    print("input size:", self.image_resolution)
                    print(self.allVals, file=gg)
                    print("\n--------------------------------\n")
            #if epoch > 50:
            if self._find_best_fitting(res) and self.save_model:
                torch.save(
                    model.state_dict(),
                    garage_path + "model_{}.pt".format(str(epoch).zfill(4)),
                )
            if self.verbose and epoch % 10 == 0:
                print(
                    "Epoch {} completed. Running time: {}".format(
                        epoch + 1, datetime.now() - s
                    )
                )
        torch.save(model.state_dict(), garage_path + "model_final.pt".format(epoch))


if __name__ == "__main__":
    # Train on GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    import torch

    torch.manual_seed(settings.SEED)
    random.seed(settings.SEED)

    np.random.seed(settings.SEED)
    # for sample_size in [10, 25, 50, 100]:
    # We need to train the new geok models of different sizes with and without transfer learning from cofly dataset
    # We do this for both sunet and ssunet
    # for architecture in ["slim", "squeeze"]:
    #architecture = "unet_perf"
    architectures = ["unet_custom","unet_custom_dau","unet_custom_perf", "unet_dau", "unet_perf", "unet2", "unet",]
    #architectures = ["unet_custom_dau"]
    for architecture in architectures:
        print("--------------------------\n\n")
        print(architecture)
        print("\n\n--------------------------")

        for i, (image_resolution, batch_size) in enumerate(zip(
            [(128, 128),
             (256, 256),
             (512, 512)
             ],
            [2**5,
             2**3,
             2**1
             ]
        )):
            lr = 0
            if i > 0 and ("dau" in architecture) and False:
                lr = 0.005
            else:
                lr = 0.01
            # tr = Training(
            #     device,
            #     dataset="geok",
            #     image_resolution=image_resolution,
            #     architecture=architecture,
            #     batch_size=batch_size,
            # )
            # tr.train()
            t0 = time.time()
            print(image_resolution, batch_size)
            tr = Training(
                device,
                dataset="geok",
                image_resolution=image_resolution,
                architecture=architecture,
                batch_size=batch_size,
                continue_model="",
                learning_rate=lr,
            )
            tr.train()
            t1 = time.time()
            print("perforated training completed in", t1 - t0, "seconds")
