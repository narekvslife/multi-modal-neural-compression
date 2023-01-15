from typing import Tuple

import torch
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torch.random import manual_seed

import torchvision
from torchvision.transforms import transforms

from compressai.models import MeanScaleHyperprior

import wandb

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


def show_images(images: list):
    fig, axs = plt.subplots(1, len(images))

    for i in range(len(images)):
        axs[i].imshow(images[i])

    plt.show()


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
    manual_seed(21)

    task_input_channels = {"rgb": 3,
                           "mono": 1,
                           "depth": 1,
                           "semseg": 1}

    dataset_train, dataloader_train = get_dataloader(dataset_name=DATASET,
                                                     batch_size=BATCH_SIZE,
                                                     num_workers=1,
                                                     is_train=True)
    # dataset_val, dataloader_val = get_dataloader(dataset_name=DATASET,
    #                                              batch_size=BATCH_SIZE,
    #                                              num_workers=1,
    #                                              is_train=False)

    # run = wandb.init()
    # artifact = run.use_artifact('narekvslife/vilab-compression/model-w99zdk8g:v19', type='model')
    # artifact_dir = artifact.download()
    single_task_compressor = models.SingleTaskCompressor.load_from_checkpoint("artifacts/model-w99zdk8g:v19/model.ckpt",
                                                                              compression_model_class=MeanScaleHyperprior,
                                                                              task=SINGLE_TASK,
                                                                              input_channels=task_input_channels[SINGLE_TASK],
                                                                              latent_channels=LATENT_CHANNELS)

    for batch in dataloader_train:
        x_hats, likelihoods = single_task_compressor(batch)
        x_hats = x_hats[SINGLE_TASK]

        show_images([b[0].detach().numpy() for b in batch] + [b[0].detach().numpy() for b in x_hats])


if __name__ == "__main__":
    main()
