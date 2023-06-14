import sys
import argparse

import pytorch_lightning as pl
import torch
from tqdm import tqdm

import models
import datasets
import train
import utils

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-p", "--model-path", type=str, required=True, help="Path to the saved model"
    )


    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
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
        "-a",
        "--accelerator",
        default="gpu",
        choices=("mps", "cpu", "gpu"),
        help="Which accelerator to use",
    )

    args = parser.parse_args(argv)
    return args

def main(args):
    pl.seed_everything(21)

    default_collate = datasets.transforms.make_collate_fn(args.tasks)

    _, dataloader = train.get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tasks=args.tasks,
        is_train=True,
        collate=default_collate
    )

    # TODO: move this to config as dictionary
    if args.model == 1:
        model_type = models.SingleTaskCompressor
    elif args.model == 2:
        model_type = models.MultiTaskMixedLatentCompressor
    elif args.model == 3:
        model_type = models.MultiTaskDisjointLatentCompressor
    elif args.model == 4:
        model_type = models.MultiTaskSharedLatentCompressor
    else:
        raise NotImplementedError(
            f"Architecture number {args.model} is not a valid choice"
        )

    ckpt_params = torch.load(args.model_path, map_location=args.accelerator)
    
    compressor = model_type(**ckpt_params["hyper_parameters"]).eval()
    compressor.load_state_dict(ckpt_params["state_dict"])
    compressor.update_bottleneck_values()

    compressor = compressor.to(args.accelerator)

    total_bytes = 0
    for batch in tqdm(dataloader):
        _, batch_bytes = compressor.compress(batch)
        total_bytes += batch_bytes

        # strings, shape = compressed_data["strings"], compressed_data["shape"]

        # _, batch_bytes  = compressor.decompress(strings, shape)
        # decompressed_image = decompressed_data["depth_euclidean"][3].detach().permute(1, 2, 0)

        # forwarded_image, _ = compressor(batch)
        # forwarded_image = forwarded_image["depth_euclidean"][3].detach().permute(1, 2, 0) 
        
        # original_image = batch["depth_euclidean"][3].detach().permute(1, 2, 0)

        # utils.show_images([decompressed_image, forwarded_image, original_image])

    print(f"Compressed train dataset takes up {(total_bytes / 1024):.2f} KB")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
