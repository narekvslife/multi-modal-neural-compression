import os

from torch.utils.data import DataLoader

from compressai.models import MeanScaleHyperprior

import models
import transforms
from dataset_folder import MultiTaskImageFolder

IMAGENET_ROOT = "../data/imagenet_multitask/"

BATCH_SIZE = 4

LATENT_CHANNELS = 90


def main():
    tasks = ("rgb", "depth", "semseg")
    task_input_channels = {"rgb": 3,
                           "depth": 1,
                           "semseg": 1}
    per_task_input_channels = tuple(task_input_channels.values())

    transform: dict = transforms.make_transforms(tasks)

    # One item of this dataset looks like:
    # ({"task1": image1,  ..., "taskN":imageN}, image_class_index)
    dataset = MultiTaskImageFolder(root=os.path.join(IMAGENET_ROOT, "train"),
                                   tasks=tasks,
                                   transform=transform,
                                   max_images=100)

    # multitask_dataloader = DataLoader(dataset,
    #                                   batch_size=BATCH_SIZE,
    #                                   collate_fn=transforms.make_collate_fn_for_tasks(tasks))
    #
    # multitask_compressor = models.MultiTaskMixedLatentCompressor(MeanScaleHyperprior,
    #                                                              tasks=tasks,
    #                                                              input_channels=per_task_input_channels,
    #                                                              latent_channels=LATENT_CHANNELS)
    #
    # for batch in multitask_dataloader:
    #     x_hats, likelihoods = multitask_compressor(batch)
    #     print(len(x_hats))
    #     print(multitask_compressor.get_loss(batch))

    single_task = "depth"
    depth_dataloader = DataLoader(dataset,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=transforms.make_collate_fn_for_tasks(single_task))

    single_task_compressor = models.SingleTaskCompressor(MeanScaleHyperprior,
                                                         task=single_task,
                                                         input_channels=task_input_channels[single_task],
                                                         latent_channels=LATENT_CHANNELS)
    for batch in depth_dataloader:
        x_hats, likelihoods = single_task_compressor(batch)
        print(len(x_hats))
        loss = single_task_compressor.get_loss(batch)
        print(loss)


if __name__ == "__main__":
    main()
