from typing import Tuple

import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from torch.random import manual_seed

import torchvision
from torchvision.transforms import transforms

import pytorch_lightning as pl

from compressai.models import MeanScaleHyperprior, ScaleHyperprior

import utils
import models
from callbacks import LogPredictionSamplesCallback
from transforms import make_collate_fn

from constants import WANDB_PROJECT_NAME, MNIST, FASHION_MNIST, CLEVR, DATASET, WANDB_RUN_NAME, SINGLE_TASK

IMAGENET_ROOT = "../data/imagenet_multitask/"

BATCH_SIZE = 16
LATENT_CHANNELS = 90

DATASET_ROOTS = {FASHION_MNIST: "../data/fashion-mnist",
                 MNIST: "../data/mnist",
                 CLEVR: "../data/clevr"}

def_t = transforms.Compose([transforms.Resize((256, 256)),
                            transforms.ToTensor()])

def_c = make_collate_fn("mono")

DATASET_TRANSFORMS = {FASHION_MNIST: def_t,
                      MNIST: def_t,
                      CLEVR: def_t}

DATASET_COLLATE = {FASHION_MNIST: def_c,
                   MNIST: def_c}


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

    # dataset = torch.utils.data.Subset(dataset, range(16))
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=DATASET_COLLATE[DATASET])

    return dataset, dataloader


def main():
    manual_seed(21)

    utils.set_wandb_logger(WANDB_RUN_NAME, WANDB_PROJECT_NAME)

    task_input_channels = {"rgb": 3,
                           "mono": 1,
                           "depth": 1,
                           "semseg": 1}

    dataset_train, dataloader_train = get_dataloader(dataset_name=DATASET,
                                                     batch_size=BATCH_SIZE,
                                                     num_workers=4,
                                                     is_train=True)
    dataset_val, dataloader_val = get_dataloader(dataset_name=DATASET,
                                                 batch_size=BATCH_SIZE,
                                                 num_workers=4,
                                                 is_train=False)

    single_task_compressor = models.SingleTaskCompressor(ScaleHyperprior,
                                                         task=SINGLE_TASK,
                                                         input_channels=task_input_channels[SINGLE_TASK],
                                                         latent_channels=LATENT_CHANNELS,
                                                         n_epoch_log=1,
                                                         pretrained=True)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=4,
        max_epochs=1000,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=WandbLogger(name=WANDB_RUN_NAME,
                           project=WANDB_PROJECT_NAME,
                           log_model="all"),
        callbacks=[LogPredictionSamplesCallback()]
    )

    trainer.fit(model=single_task_compressor,
                train_dataloaders=dataloader_train,
                val_dataloaders=dataloader_val)


if __name__ == "__main__":
    main()
