from typing import Any

import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger


class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger: WandbLogger):
        super(LogPredictionSamplesCallback, self).__init__()

        self.wandb_logger = wandb_logger

    def log_predicted_images(self, batch, trainer: "pl.Trainer", pl_module: "pl.LightningModule", directory: str) -> None:

        # unfortunately this has to be here :(
        for task in pl_module.tasks:
            batch[task] = batch[task].to(pl_module.device)

        x_hats, _ = pl_module(batch)

        for task in pl_module.tasks:
            x_hats_task = x_hats[task]

            pred_key = f'{directory}/{task}/predicted'
            target_key = f'{directory}/{task}/target'

            kwargs = {}
            if task == "semantic":
                pred_images = target_images = [torch.zeros(xh[0].shape) for xh in x_hats_task]  # log black images
                kwargs["masks"] = [{"predictions": {"mask_data": torch.argmax(xh, dim=0).cpu().numpy()}} for xh in x_hats_task]
            else:
                pred_images = [xh for xh in x_hats_task]
                target_images = [x for x in batch[task]]

            self.wandb_logger.log_image(key=pred_key,
                                        images=pred_images,
                                        **kwargs)

            # log target only once
            if trainer.global_step < 100:
                kwargs = {}
                if task == "semantic":
                    kwargs["masks"] = [{"ground_truth": {"mask_data": x.cpu().numpy()}} for x in batch[task].squeeze(1)]

                self.wandb_logger.log_image(
                    key=target_key,
                    images=target_images,
                    **kwargs)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:

        if trainer.global_step % 15000 != 0:
            return None

        with torch.no_grad():
            pl_module.eval()
            batch = next(iter(trainer.train_dataloader))
            self.log_predicted_images(batch, trainer, pl_module,
                                      directory="train")

            batch = next(iter(trainer.val_dataloaders[0]))
            self.log_predicted_images(batch, trainer, pl_module,
                                      directory="val")
            pl_module.train()
