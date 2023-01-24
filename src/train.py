
import sys
import argparse

from typing import Tuple, Callable

from datasets.transforms import task_configs

from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
from torchvision.transforms import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from compressai.models import ScaleHyperprior

import models
import datasets
from datasets.transforms import make_collate_fn
import callbacks

from constants import (WANDB_PROJECT_NAME, MNIST, FASHION_MNIST, CLEVR)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Training dataset"
    )

    parser.add_argument("-t",
                        "--tasks",
                        required=True,
                        nargs='+',
                        type=str,
                        help="Task(s) that will be used")

    parser.add_argument("-m",
                        "--model",
                        required=True,
                        type=int,
                        choices=range(1, 5),
                        help="Which type of the model to choose:"
                             "1 - SingleTask, 2 - MixedLatentMultitask, "
                             "3 - SeparateLatentMultitask, 4 - SharedSeparateLatentMultitask")

    parser.add_argument("-l",
                        "--latent-channels",
                        required=True,
                        type=int,
                        help="Number of channels in the latent code (information bottleneck) of the network")

    parser.add_argument("-c",
                        "--conv-channels",
                        default=100,
                        type=int,
                        required=True,
                        help="Number of channels in all convolutions of the network (except the layers right "
                             "before and after the bottleneck)")

    parser.add_argument("-w",
                        "--wandb-run-name",
                        required=True,
                        help="additional name for the run")

    parser.add_argument(
        "-p",
        "--pretrained",
        type=bool,
        default=False,
        help="Whether to use pretrained backbone or not",
    )
    parser.add_argument("-q",
                        "--quality",
                        default=4,
                        type=int,
                        choices=range(1, 9),
                        help="Quality of the pretrained model (bigger models have bigger latent size 192 vs 320")

    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Max number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lrm",
        "--learning-rate-main",
        default=1e-5,
        type=float,
        help="Learning rate for the main loss (default: %(default)s)",
    )

    parser.add_argument(
        "-lra",
        "--learning-rate-aux",
        default=1e-3,
        type=float,
        help="Learning rate for the auxilary loss(default: %(default)s)",
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

    parser.add_argument("-g",
                        "--devices",
                        default=1,
                        type=int,
                        help="Number of devices to use")

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

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate)

    return dataset, dataloader


def main(args):
    pl.seed_everything(21)

    wandb_logger = WandbLogger(name=args.wandb_run_name, project=WANDB_PROJECT_NAME, log_model="all")

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
        model_type = models.SingleTaskCompressor
    elif args.model == 2:
        model_type = models.MultiTaskMixedLatentCompressor
    else:
        raise NotImplementedError(f"Architecture number {args.model} is not available")

    input_channels = tuple(task_configs.task_parameters[t]["out_channels"] for t in args.tasks)

    compressor = model_type(compression_model_class=ScaleHyperprior,
                            tasks=args.tasks,
                            input_channels=input_channels,
                            latent_channels=args.latent_channels,
                            conv_channels=args.conv_channels,
                            quality=args.quality,
                            lmbda=args.lmbda,
                            learning_rate_main=args.learning_rate_main,
                            learning_rate_aux=args.learning_rate_aux)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[callbacks.LogPredictionSamplesCallback(wandb_logger=wandb_logger),
                   ModelCheckpoint(every_n_epochs=50, filename=args.wandb_run_name),
                   LearningRateMonitor()]
    )

    trainer.fit(model=compressor,
                train_dataloaders=dataloader_train,
                val_dataloaders=dataloader_val)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

