import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.random import manual_seed

import torchvision
from torchvision.transforms import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from compressai.models import MeanScaleHyperprior

import models
from transforms import make_collate_fn


IMAGENET_ROOT = "../data/imagenet_multitask/"

BATCH_SIZE = 16
LATENT_CHANNELS = 90

SINGLE_TASK = "mono"

MNIST = "mnist"
FASHION_MNIST = "fashion-mnist"

DATASET_ROOTS = {FASHION_MNIST: "../data/fashion-mnist",
                 MNIST: "../data/mnist"}
def_t = transforms.Compose([transforms.Resize((256, 256)),
                            transforms.ToTensor()])

def_c = make_collate_fn("mono")

DATASET_TRANSFORMS = {FASHION_MNIST: def_t,
                      MNIST: def_t}

DATASET_COLLATE = {FASHION_MNIST: def_c,
                   MNIST: def_c}

DATASET = FASHION_MNIST

WANDB_PROJECT_NAME = "vilab-compression"
WANDB_RUN_NAME = f"S-{DATASET}-{SINGLE_TASK}"


def set_wandb_logger():
    return WandbLogger(name=WANDB_RUN_NAME,
                       project=WANDB_PROJECT_NAME,
                       log_model="all")


def get_dataloader(dataset_name: str, batch_size: int, num_workers: int, is_train=False) -> Tuple[Dataset, DataLoader]:

    root = DATASET_ROOTS[dataset_name]
    trans = DATASET_TRANSFORMS[dataset_name]

    if dataset_name == FASHION_MNIST:
        dataset = torchvision.datasets.FashionMNIST(root,
                                                    download=True,
                                                    transform=trans,
                                                    train=is_train)
    elif dataset_name == MNIST:
        dataset = torchvision.datasets.MNIST(root,
                                             download=True,
                                             transform=trans,
                                             train=is_train)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")

    dataset = torch.utils.data.Subset(dataset, range(16))
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=DATASET_COLLATE[DATASET])

    return dataset, dataloader


def main():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    manual_seed(21)

    task_input_channels = {"rgb": 3,
                           "mono": 1,
                           "depth": 1,
                           "semseg": 1}

    dataset_train, dataloader_train = get_dataloader(dataset_name=DATASET,
                                                     batch_size=BATCH_SIZE,
                                                     num_workers=4, is_train=True)
    dataset_val, dataloader_val = get_dataloader(dataset_name=DATASET,
                                                 batch_size=BATCH_SIZE,
                                                 num_workers=4, is_train=False)

    single_task_compressor = models.SingleTaskCompressor(MeanScaleHyperprior,
                                                         task=SINGLE_TASK,
                                                         input_channels=task_input_channels[SINGLE_TASK],
                                                         latent_channels=LATENT_CHANNELS)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=100,
        # logger=set_wandb_logger()
    )

    trainer.fit(model=single_task_compressor,
                train_dataloaders=dataloader_train,
                val_dataloaders=dataloader_val)


if __name__ == "__main__":
    main()
