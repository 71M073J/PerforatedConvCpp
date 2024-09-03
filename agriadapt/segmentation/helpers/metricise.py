import numpy as np
import torch
import wandb

from segmentation import settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.masking import get_binary_masks_infest
from adaptation.inference import AdaptiveWidth
import matplotlib.pyplot as plt

class Metricise:
    """
    How to use this class?
    This is meant to be used on per-epoch basis, meaning a typical usage looks like this:
    for __ in epochs:
        # Instantiate a fresh object to get empy metrics dict
        metrics = Metricise()
        # Probably train and valid set (or only test when testing)
        metrics.evaluate(model, loader, dataset_name, epoch, ...)
        metrics.evaluate(model, loader, dataset_name, epoch, ...)
        results = metrics.report()
    """

    def __init__(
        self,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        classes=("back", "weeds"),
        widths=settings.WIDTHS,
        metrics=settings.METRICS,
        # Calculate also the metrics for our adaptive algorithm
        use_adaptive=False,
        # Calculate also the metrics of the oracle predictor
        use_oracle=False,
    ):
        self.results = {}
        self.device = device
        self.classes = classes
        self.widths = widths
        self.metrics = metrics
        self.use_adaptive = use_adaptive
        self.use_oracle = use_oracle

    def evaluate(
        self,
        model,
        loader,
        name,
        epoch,
        loss_function=None,
        image_pred=False,
        adaptive_knns=None,
    ):
        losses = []
        imgs = []
        for width in settings.WIDTHS:
            key = f"{name}/{int(width * 100)}"
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                #model.set_width(width)
                y_pred = model.forward(X)
                for i in range(len(y_pred)):
                    losses.append(loss_function(y_pred[i][None, :, :, :], y[i][None, :, :, :]).detach().cpu())
                    imgs.append(y_pred[i])
                    #TODO: find good and bad performing images for every network for each network, make a folder that contains 3 best and 3 worst images
                self.calculate_metrics(y, y_pred, key)
                if loss_function:
                    self.add_static_value(loss_function(y, y_pred).cpu(), key + "/loss")

                if width == self.widths[-1] and image_pred:
                    self._add_image(X, y, y_pred, epoch)


        if self.use_adaptive:
            self._calculate_adpative(adaptive_knns, name, loader, model)
        ind_worst = np.argmax(losses)
        ind_best = np.argmin(losses)
        return ind_worst, imgs[ind_worst].cpu().detach().round(), ind_best, imgs[ind_best].cpu().detach().round()

    def calculate_metrics(self, y_true, y_pred, name):
        y_pred = get_binary_masks_infest(y_pred, dim=2)
        assert y_true.shape == y_pred.shape
        for metric in self.metrics:
            for i, pred_class in enumerate(self.classes):
                key = f"{name}/{metric}/{pred_class}"
                value = self.metrics[metric](validate_args=False).to(self.device)(
                    y_true[:, i], y_pred[:, i]
                )
                if key in self.results:
                    self.results[key].append(value.cpu())
                else:
                    self.results[key] = [value.cpu()]

    def _calculate_adpative(self, adaptive_knns, name, loader, model):
        if not adaptive_knns:
            raise ValueError(
                "Please set adaptive_knns if you want to run the adaptation algorithm."
            )
        width_selector = AdaptiveWidth(adaptive_knns)
        key = f"{name}/adapt"
        for X, y in loader:
            image = ImageImporter("geok").tensor_to_image(X)[0]
            X, y = X.to(self.device), y.to(self.device)
            width = width_selector.get_image_width(image)
            model.set_width(width)
            y_pred = model.forward(X)
            self.calculate_metrics(y, y_pred, key)
            self.add_static_value(width, f"{key}/width")

    def _calculate_oracle(self):
        for metric in self.metrics:
            for pred_class in self.classes:
                values, widths = [], []
                for i in range(len(self.results[f"test/25/{metric}/{pred_class}"])):
                    max_val, max_width = 0, 0
                    for width in self.widths:
                        value = self.results[
                            f"test/{str(int(width*100))}/{metric}/{pred_class}"
                        ][i]
                        if value > max_val:
                            max_val = value
                            max_width = width
                    values.append(max_val)
                    widths.append(max_width)
                self.results[f"test/oracle/{metric}/{pred_class}"] = np.mean(values)
                self.results[f"test/oracle/{metric}/{pred_class}/width"] = np.mean(
                    widths
                )

    def add_static_value(self, value, key):
        if key in self.results:
            self.results[key].append(value)
        else:
            self.results[key] = [value]

    def _add_image(self, X, y, y_pred, epoch):
        y = torch.argmax(y, dim=1)
        y = y + 1

        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred + 1

        table = wandb.Table(columns=["ID", "Epoch", "Image"])
        class_labels = {
            i + 1: class_label for i, class_label in enumerate(self.classes)
        }
        for i in range(10):
            try:
                original_image = X[i].cpu().permute(1, 2, 0)

                mask_org = y[i].cpu()
                mask_pred = y_pred[i].cpu()
                masks = wandb.Image(
                    original_image.numpy(),
                    masks={
                        "labels": {
                            "mask_data": mask_org.numpy(),
                            "class_labels": class_labels,
                        },
                        "predictions": {
                            "mask_data": mask_pred.numpy(),
                            "class_labels": class_labels,
                        },
                    },
                )

                table.add_data(i, epoch, masks)
            except IndexError:
                pass
        self.results[f"images"] = table

    def report(self, wandb):
        self._aggregate_metrics()
        if wandb and settings.WANDB:
            wandb.log(self.results)
        return self.results

    def _aggregate_metrics(self):
        remove_list = []
        if self.use_oracle:
            self._calculate_oracle()
        for x in self.results:
            if x == "images":
                if self.results[x]:
                    self.results[x] = self.results[x]
                else:
                    remove_list.append(x)
            elif x == "learning_rate":
                self.results[x] = self.results[x][0]
            else:
                self.results[x] = np.mean(self.results[x])
        for x in remove_list:
            self.results.pop(x)
