from typing import Any

import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger

from constants import WANDB_RUN_NAME, WANDB_PROJECT_NAME


class LogPredictionSamplesCallback(Callback):

    def log_predicted_images(self, batch, trainer: "pl.Trainer", pl_module: "pl.LightningModule", directory: str) -> None:

        x_hats, likelihoods = pl_module(batch)
        for task in pl_module.tasks:
            x_hats_task = x_hats[task]

            # todo: delete this before anybody made fun of you
            if len(pl_module.tasks) == 1:
                x_task = batch
            else:
                x_task = batch[task]

            WandbLogger(name=WANDB_RUN_NAME,
                        project=WANDB_PROJECT_NAME,
                        log_model="all").log_image(
                key=f'{directory}/{task}/predicted',
                images=[xh for xh in x_hats_task])

            # show target only once
            if trainer.global_step < 100:
                WandbLogger(name=WANDB_RUN_NAME,
                            project=WANDB_PROJECT_NAME,
                            log_model="all").log_image(
                    key=f'{directory}/{task}/target',
                    images=[x for x in x_task])

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if trainer.global_step % 100 != 0:
            return None


        with torch.no_grad():
            pl_module.eval()
            batch = next(iter(trainer.train_dataloader))[:16].to(pl_module.device)
            self.log_predicted_images(batch, trainer, pl_module,
                                      directory="train")

            batch = next(iter(trainer.val_dataloaders[0]))[:16].to(pl_module.device)
            self.log_predicted_images(batch, trainer, pl_module,
                                      directory="val")
            pl_module.train()
