from typing import Any

import torch

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger


class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger: WandbLogger):
        super(LogPredictionSamplesCallback, self).__init__()

        self.wandb_logger = wandb_logger

    def log_predicted_images(
        self, batch, trainer: pl.Trainer, pl_module: pl.LightningModule, directory: str
    ) -> None:
        # unfortunately this has to be here :(
        for task in pl_module.tasks:
            batch[task] = batch[task].to(pl_module.device)

        x_hats, _ = pl_module(batch)

        for task in pl_module.tasks:
            x_hats_task = x_hats[task]

            pred_key = f"{directory}/{task}/predicted"
            target_key = f"{directory}/{task}/target"

            kwargs = {}

            pred_images = [xh for xh in x_hats_task]
            target_images = [x for x in batch[task]]

            self.wandb_logger.log_image(key=pred_key, images=pred_images, **kwargs)

            # log target images only once TODO: move target image logging to a separate function
            if trainer.current_epoch == trainer.check_val_every_n_epoch - 1:
                kwargs = {}

                self.wandb_logger.log_image(
                    key=target_key, images=target_images, **kwargs
                )

    # local-testing 1.0
    # def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     batch = next(iter(trainer.val_dataloaders))
    #     self.log_predicted_images(batch, trainer, pl_module,
    #                             directory="val")

    # local-testing 1.1
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.sanity_checking:
            return

        batch = next(iter(trainer.train_dataloader))
        self.log_predicted_images(batch, trainer, pl_module, directory="train")

        val_dl = trainer.val_dataloaders if type(trainer.val_dataloaders) == DataLoader else trainer.val_dataloaders[0]
        batch = next(iter(val_dl))
        self.log_predicted_images(batch, trainer, pl_module, directory="val")
