import copy

import numpy as np
import torch
from torch.utils.data import Dataset


def tensorise(x, channels=3):
    return torch.tensor([float(x) for x in x.split(" ") if len(x) > 0]).view(-1, channels).movedim(0,1)

class UciHAR(Dataset):
    def __init__(self, mode=None, transform=None):
        if mode not in ["train", "test"]:
            raise ValueError("Mode not supported")
        #tensor = None
        #with open(f"./data/UCI HAR Dataset/{mode}/X_{mode}.txt") as f:
        #    tensor = torch.stack(list(map(tensorise, f.readlines())))
        ts = []
        for file in ["body_acc", "body_gyro", "total_acc"]:
            t = []
            for dim in ["x", "y", "z"]:
                with open(f"./data/UCI HAR Dataset/{mode}/Inertial Signals/{file}_{dim}_{mode}.txt") as f:
                    t.append(torch.stack(list(map(lambda p: tensorise(p, 1), f.readlines()))))
            t = torch.cat(t, dim=1)
            t = torch.cat((t, t, t[:, :2, :]), dim=1)
            #print(t.shape)
            ts.append(t)
        ts = torch.stack(ts, dim=1)
        self.images = torch.cat((ts[:, :, :, :32], ts[:, :, :, 32:64], ts[:, :, :, 64:3*32], ts[:, :, :, 3*32:]), dim=2)
        with open(f"./data/UCI HAR Dataset/{mode}/y_{mode}.txt") as f:
            self.classes = torch.stack(list(map(lambda p: tensorise(p, 1), f.readlines()))).long().squeeze()
            if torch.min(self.classes) == 1 and torch.max(self.classes) == 6:
                self.classes = self.classes - 1
        self.transform = transform
        #    self.classes = torch.nn.functional.one_hot(torch.stack(list(map(lambda p: tensorise(p, 1), f.readlines()))).long().squeeze(), num_classes=10)
        #    print(self.classes.shape)
        #tensor = torch.cat((tensor, ts), dim=-1)
    #def __len__(self):
    #    return len(self.X)

    def __getitem__(self, item):
        if self.transform is not None:
            return self.transform(self.images[item]), self.classes[item]
        else:
            return self.images[item], self.classes[item]

    def __len__(self):
        return len(self.classes)


if __name__ == "__main__":
    test = UciHAR("train")

    print(test[97])
    print()