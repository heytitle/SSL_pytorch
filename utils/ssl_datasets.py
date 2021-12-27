import torch

from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from augmentation import *
import lightly


class SSL_CIFAR10(object):
    def __init__(self, data_root, augmentation, dl_kwargs):
        # Cifar10 Mean and Std
        CIFAR10_NORM = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]

        collate_fn = lightly.data.SimCLRCollateFunction(
            input_size=32,
            gaussian_blur=0.0,
        )

        test_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_NORM[0],
                    std=IMAGENET_NORM[1],
                ),
            ]
        )

        tv_ds_train_ssl = CIFAR10(data_root)
        tv_ds_train_knn = CIFAR10(data_root, download=True)
        tv_ds_test = CIFAR10(data_root, download=True, train=False)

        dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
            tv_ds_train_ssl
        )
        dataset_train_eval = lightly.data.LightlyDataset.from_torch_dataset(
            tv_ds_train_knn, transform=test_transforms
        )
        dataset_test = lightly.data.LightlyDataset.from_torch_dataset(
            tv_ds_test, transform=test_transforms
        )

        batch_size = 512
        num_workers = 2

        self.train_dl = torch.utils.data.DataLoader(
            dataset_train_ssl,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
        )

        self.train_eval_dl = torch.utils.data.DataLoader(
            dataset_train_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

        self.test_dl = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
