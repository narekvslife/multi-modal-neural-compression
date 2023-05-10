import sys
import argparse

import pytorch_lightning as pl

import models
import datasets
import train

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-p", "--model-path", type=str, required=True, help="Path to the saved model"
    )


    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
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

    args = parser.parse_args(argv)
    return args

def main(args):
    pl.seed_everything(21)


    default_collate = datasets.transforms.make_collate_fn(args.tasks)

    _, dataloader = train.get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
        collate=default_collate,
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

    compressor = torch.load



if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
