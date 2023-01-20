
import sys
import argparse

from typing import Tuple, Callable

from datasets.transforms import task_configs

from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
from torchvision.transforms import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from compressai.models import ScaleHyperprior

import models
import datasets
from datasets.transforms import make_collate_fn
import callbacks

from constants import (WANDB_PROJECT_NAME, MNIST, FASHION_MNIST, CLEVR)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-p",
        "--pretrained",
        type=bool,
        default=True,
        help="Whether to use pretrained backbone or not",
    )
    parser.add_argument("-q",
                        "--quality",
                        default=4,
                        type=int,
                        choices=range(1, 9),
                        help="Quality of the pretrained model (bigger models have bigger latent size 192 vs 320")

    parser.add_argument("-t",
                        "--tasks",
                        required=True,
                        nargs='+',
                        type=str,
                        help="Task(s) that will be used")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Max number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )

    parser.add_argument("-w",
                        "--wandb-run-name",
                        default="",
                        help="additional name for the run")

    parser.add_argument("-m",
                        "--model",
                        required=True,
                        type=int,
                        choices=range(1, 5),
                        help="Which type of the model to choose:"
                             "1 - SingleTask, 2 - MixedLatentMultitask, "
                             "3 - SeparateLatentMultitask, 4 - SharedSeparateLatentMultitask")

    parser.add_argument("-g",
                        "--devices",
                        default=1,
                        type=int,
                        help="Number of devices to use")

    parser.add_argument("-l",
                        "--latent-size",
                        default=190,
                        type=int,
                        help="Number of channels in the latent space")

    parser.add_argument("-a",
                        "--accelerator",
                        default="gpu",
                        choices=("mps", "cpu", "gpu"),
                        help="Which accelerator to use")

    args = parser.parse_args(argv)
    return args


BATCH_SIZE = 16
LATENT_CHANNELS = 90

DATASET_ROOTS = {FASHION_MNIST: "../data/fashion-mnist",
                 MNIST: "../data/mnist",
                 CLEVR: "../../vilabdatasets/clevr/clevr-taskonomy-complex/"}


def get_dataloader(dataset_name: str, batch_size: int, num_workers: int, collate: Callable, is_train=False) -> Tuple[Dataset, DataLoader]:

    root = DATASET_ROOTS[dataset_name]

    default_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ToTensor()])

    if dataset_name == FASHION_MNIST:
        dataset = torchvision.datasets.FashionMNIST(root,
                                                    download=True,
                                                    transform=default_transform,
                                                    train=is_train)
    elif dataset_name == MNIST:
        dataset = torchvision.datasets.MNIST(root,
                                             download=True,
                                             transform=default_transform,
                                             train=is_train)
    elif dataset_name == CLEVR:
        dataset = datasets.CLEVRDataset(root,
                                        tasks=args.tasks,
                                        split="train" if is_train else "val")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")

    dataset = Subset(dataset, range(16))
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate)

    return dataset, dataloader


def main(args):
    pl.seed_everything(21)

    t_string = "_".join(list(args.tasks))
    wandb_run_name = f"{args.dataset}-{t_string}-{args.wandb_run_name}"
    wandb_logger = WandbLogger(name=wandb_run_name, project=WANDB_PROJECT_NAME, log_model="all")

    default_collate = make_collate_fn(args.tasks)

    dataset_train, dataloader_train = get_dataloader(dataset_name=args.dataset,
                                                     batch_size=args.batch_size,
                                                     num_workers=args.num_workers,
                                                     is_train=True,
                                                     collate=default_collate)

    dataset_val, dataloader_val = get_dataloader(dataset_name=args.dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 is_train=False,
                                                 collate=default_collate)

    if args.model == 1:
        single_task_compressor = models.SingleTaskCompressor(ScaleHyperprior,
                                                             task=args.tasks[0],
                                                             input_channels=task_configs.task_parameters[args.tasks[0]]["out_channels"],
                                                             latent_channels=10,  # TODO: this doesn't matter - for pretrained networks it's fixed
                                                             pretrained=args.pretrained,
                                                             quality=args.quality,
                                                             lmbd=args.lmbda)
    else:
        raise NotImplementedError(f"Architecture number {args.model} is not available")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[callbacks.LogPredictionSamplesCallback(wandb_logger=wandb_logger),
                   LearningRateMonitor()]
    )

    trainer.fit(model=single_task_compressor,
                train_dataloaders=dataloader_train,
                val_dataloaders=dataloader_val)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

