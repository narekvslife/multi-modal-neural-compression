import sys
import argparse

from typing import List, Tuple, Callable

from datasets.transforms import task_configs

import wandb

from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
from torchvision.transforms import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from compressai.models import ScaleHyperprior

import utils
import models
import datasets
import callbacks
from datasets.transforms import make_collate_fn

from constants import WANDB_PROJECT_NAME, MNIST, FASHION_MNIST, CLEVR


MODEL_NUMBER = {1: models.SingleTaskCompressor,
                2: models.MultiTaskMixedLatentCompressor,
                3: models.MultiTaskDisjointLatentCompressor,
                4: models.MultiTaskSharedLatentCompressor}

MODEL_NAME = {"SingleTaskCompressor": models.SingleTaskCompressor,
              "MultiTaskMixedLatentCompressor": models.MultiTaskMixedLatentCompressor,
              "MultiTaskDisjointLatentCompressor": models.MultiTaskDisjointLatentCompressor,

              "MultiTaskSharedLatentCompressor": models.MultiTaskSharedLatentCompressor}

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-t",
        "--tasks",
        required=True,
        nargs="+",
        type=str,
        help="Task(s) that will be used",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=int,
        choices=range(1, 5),
        help="Which type of the model to choose:"
        "1 - SingleTask, "
        "2 - MixedLatentMultitask, "
        "3 - SeparateLatentMultitask, "
        "4 - SharedSeparateLatentMultitask",
    )
    parser.add_argument(
        "-l",
        "--latent-channels",
        required=True,
        type=int,
        help="Number of channels in the latent code (information bottleneck) of the network",
    )
    parser.add_argument(
        "-c",
        "--conv-channels",
        default=100,
        type=int,
        required=True,
        help="Number of channels in all convolutions of the network (except the layers right "
        "before and after the bottleneck)",
    )
    parser.add_argument(
        "-w", "--wandb-run-name", required=True, help="additional name for the run"
    )
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
        default=1e-4,
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
        help="Bit-rate distortion tradeoff parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "-g", "--devices", default=1, type=int, help="Number of devices to use"
    )
    parser.add_argument(
        "-a",
        "--accelerator",
        default="gpu",
        choices=("mps", "cpu", "gpu"),
        help="Which accelerator to use",
    )
    parser.add_argument(
        "-p",
        "--precision",
        default="32",
        help="Precision arithmetic used for trainning",
    )
    parser.add_argument(
        "--wandb_checkpoint_path",
        default=None,
        help="Name of model checkpoint from a particular run. Looks like: <entity>/<project>/model-<run>:v#",
    )
    parser.add_argument(
        "--continue-run-id",
        default=None,
        help="id of a run to continue from last checkpoint",
    )
    args = parser.parse_args(argv)
    return args

# TODO: move this paths to configs
DATASET_ROOTS = {
    FASHION_MNIST: "data/fashion-mnist",
    MNIST: "data/mnist",
    # CLEVR: "../data/clevr",
    CLEVR: "../../vilabdatasets/clevr/clevr-taskonomy-complex/",
}


def get_dataloader(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    tasks: List[str],
    collate: Callable,
    is_train=False,
) -> Tuple[Dataset, DataLoader]:
    root = DATASET_ROOTS[dataset_name]

    # TODO: add image resize params to config
    default_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    if dataset_name == FASHION_MNIST:
        dataset = torchvision.datasets.FashionMNIST(
            root, download=True, transform=default_transform, train=is_train
        )
    elif dataset_name == MNIST:
        dataset = torchvision.datasets.MNIST(
            root, download=True, transform=default_transform, train=is_train
        )
    elif dataset_name == CLEVR:
        dataset = datasets.CLEVRDataset(
            root, tasks=tasks, split="train" if is_train else "val"
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")

    # dataset = Subset(dataset, range(batch_size))  # Use this only for local checking

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
    )

    return dataset, dataloader


def main(args):
    pl.seed_everything(21)


    # this is the case where we want to continue a run and report to the same experiment
    wandb_run_id = None
    if args.continue_run_id.lower() != "none":
        wandb_run_id = args.continue_run_id

    wandb_logger = WandbLogger(
        name=args.wandb_run_name,
        project=WANDB_PROJECT_NAME,
        id=wandb_run_id,
        log_model="all",
        resume="allow",
    )

    checkpoint_path = None
    # this is the case where we want to take a checkpoint and start a new experiment from that
    if args.wandb_checkpoint_path.lower() != "none":
        raise NotImplemented()
        compressor.learning_rate_main = args.learning_rate_main
        compressor.learning_rate_aux = args.learning_rate_aux

    # this is the case where we want to continue a run and report to the same experiment
    elif args.continue_run_id.lower() != "none":
        checkpoint_path, model_name, tasks = utils.find_last_wandb_checkpoint(wandb.run)
        model_type = MODEL_NAME[model_name]
        compressor = utils.load_from_checkpoint(checkpoint_path, model_type)
    else:
        model_type = MODEL_NUMBER[args.model]

        input_channels = tuple(
            task_configs.task_parameters[t]["in_channels"] for t in args.tasks
        )
        output_channels = tuple(
            task_configs.task_parameters[t]["out_channels"] for t in args.tasks
        )

        compressor = model_type(
            compressor_backbone_class=ScaleHyperprior,
            tasks=args.tasks,
            input_channels=input_channels,
            output_channels=output_channels,
            latent_channels=args.latent_channels,
            conv_channels=args.conv_channels,
            lmbda=args.lmbda,
            learning_rate_main=args.learning_rate_main,
            learning_rate_aux=args.learning_rate_aux,
        )

        tasks = args.tasks


    default_collate = make_collate_fn(tasks)

    _, dataloader_train = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tasks=tasks,
        is_train=True,
        collate=default_collate,
    )

    _, dataloader_val = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tasks=tasks,
        is_train=False,
        collate=default_collate,
    )

    wandb_logger.experiment.config.update(
        {"architecture_type": compressor.get_model_name()}, allow_val_change=True
    )

    # dont even ask me

    # TODO: this is a crutch because pytorch ligning handles the optimizers terribly!!!!!
    # it seems to load/save/store (?) the oprimizers parameters such as adam per-parameter-weights
    # in random order, that's why during first train iteration when trying to update those weights
    # we get a shape mismatch :)))))))))
    class kek_strategy(pl.strategies.single_device.SingleDeviceStrategy):
        @property
        def lightning_restore_optimizer(self) -> bool:
            return False

    trainer = pl.Trainer(
            strategy=kek_strategy(device="cuda:0"),
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        precision=args.precision if not args.precision.isnumeric() else int(args.precision),
        callbacks=[
            callbacks.LogPredictionSamplesCallback(wandb_logger=wandb_logger),
            ModelCheckpoint(every_n_epochs=100, filename=args.wandb_run_name),
            LearningRateMonitor(),
        ],
    )

    trainer.fit(
        model=compressor,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
        ckpt_path=checkpoint_path
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)