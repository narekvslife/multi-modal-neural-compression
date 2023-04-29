# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from compressai.models import ScaleHyperprior

from pytorch_msssim import ms_ssim
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio

from compressai.layers import GDN
from compressai.zoo import bmshj2018_hyperprior
from compressai.models.utils import conv, deconv

from datasets.task_configs import task_parameters

from loss_balancing import UncertaintyWeightingStrategy


class MultiTaskMixedLatentCompressor(pl.LightningModule):
    """
    A single Compressor network with multiple input/output heads and "mixed" latents for all the tasks.

    This version has one encoder input head per input task with task-specific input dimensions,
    which are later mixed in the shared encoder, which produces the latents.

    These latents are later passed to the task-specific decoders to estimate the according input
    """

    def __init__(
        self,
        compressor_backbone_class: type,
        tasks: Tuple[str],
        input_channels: Tuple[int],
        output_channels: Tuple[int],
        latent_channels: int,
        conv_channels: int,
        lmbda: float = 1,
        pretrained: bool = False,
        quality: int = 4,
        learning_rate_main=1e-5,
        learning_rate_aux=1e-3,
        **kwargs,
    ):
        """
        :param: compressor_backbone_class - type of the backbone
                                            compresion model
        :param: tasks - list of task names
        :param: input_channels - tuple with the number of channels
                                 for each task
        :param: conv_channels - number of channels in the convolutions
                                of each input head.
                                The compressor backbone gets
                                len(tasks) * conv_channels channels as input
        :param: latent_channels - number of channels in the latent code (M)
        :param: lmbda - multiplier of the reconstruction loss in the total loss
        :param: pretrained - whether to use pretrained backbone compressor
        :param: quality - quality of the pretrained compressor
        """
        super().__init__()
        self.save_hyperparameters()
        self.compressor_backbone_class = compressor_backbone_class

        self.tasks = tasks
        self.n_tasks = len(self.tasks)

        self.input_channels = input_channels
        self.output_channels = output_channels

        assert self.n_tasks == len(self.input_channels)

        self.latent_channels = latent_channels
        self.conv_channels = conv_channels

        self.lmbda = lmbda

        self.pretrained = pretrained

        if self.pretrained:
            print(
                "Note that pretrained models"
                "have fixed size of latents,"
                "independent of specified 'latent_channels'"
            )

        self.automatic_optimization = False

        self.quality = quality

        self.learning_rate_main = learning_rate_main
        self.learning_rate_aux = learning_rate_aux

        self.kwargs = kwargs

        self.model: nn.ModuleDict = self._build_model()

        self.loss_balancer = UncertaintyWeightingStrategy(self.n_tasks)

        # TODO: move this to config
        self.metrics = {"psnr": peak_signal_noise_ratio, "ms-ssim": ms_ssim}

    def get_model_name(self):
        return self.__class__.__name__

    def _build_heads(
        self,
        input_channels: Union[Tuple[int], int],
        output_channels_per_head: Union[Tuple[int], int],
        is_deconv=False,
    ) -> nn.ModuleList:
        """
        :param: input_channels  - an integer or list of integers
                                  specifying the number of input
                                  *channels for each task*.
        :param: output_channels_per_head - an integer or list of integers
                                  specifying the the number of output
                                  *channels for each task*.
        """

        if type(input_channels) == int:
            input_channels = tuple(input_channels for _ in range(self.n_tasks))
        
        assert len(input_channels) == self.n_tasks
            
        if type(output_channels_per_head) == int:
            output_channels_per_head = [output_channels_per_head for _ in range(self.n_tasks)]

        assert len(output_channels_per_head) == self.n_tasks

        list_of_modules = []

        for t_i in range(self.n_tasks):
            i_c = input_channels[t_i]
            pto_c = output_channels_per_head[t_i]  # output channels for each task (head)
            pti_c = (
                i_c // 2 if is_deconv else pto_c // 2
            )  # intermediate channels for each task (head)

            if is_deconv:
                head = nn.Sequential(
                    deconv(i_c, pti_c),
                    GDN(pti_c, inverse=True),
                    deconv(pti_c, pti_c),
                    GDN(pti_c, inverse=True),
                    deconv(pti_c, pto_c, stride=1),
                    GDN(pto_c, inverse=True),
                    conv(pto_c, pto_c, kernel_size=3, stride=1),
                )
            else:
                head = nn.Sequential(
                    conv(i_c, pti_c, kernel_size=3, stride=1),
                    GDN(pti_c),
                    conv(pti_c, pto_c),
                    GDN(pto_c),
                    conv(pto_c, pto_c),
                    GDN(pto_c),
                )

            list_of_modules.append(head)

        return nn.ModuleList(list_of_modules)

    def _build_compression_backbone(self, input_channels: int, latent_channels: int) -> nn.Module:
        """
        :param N: - number of channels for each convolution layer of the compression model
        :param M: - number of latent channels (in the latent code) that later will be compressed
        :return:
        """
        if self.pretrained:
            if self.compressor_backbone_class == ScaleHyperprior:
                model = bmshj2018_hyperprior(
                    self.quality,
                    metric="mse",
                    pretrained=True,
                    progress=True,
                    **self.kwargs,
                )
            else:
                raise ValueError(
                    f"No pretrained model available of class {self.compressor_backbone_class}"
                )

        else:
            model = self.compressor_backbone_class(N=input_channels, M=latent_channels, **self.kwargs)

        if latent_channels != model.M:
            print(
                f"Note that the pretrained {self.compressor_backbone_class} has a fixed latent size M={model.M}, "
                f"which is different from the specified M={latent_channels}"
            )

        # This is the part that i have to deal with because in the CompressAI models
        # the default input and output dimension is a hardcoded 3
        model.g_a[0] = conv(input_channels,input_channels)
        model.g_s[-1] = deconv(input_channels, input_channels)

        return model

    def _build_model(self) -> nn.ModuleDict:
        """
        x1 -> task_enc1 -> t_1 ->  ↓                              -> task_dec1 -> x1_hat

        x2 -> task_enc2 -> t_2 -> [+] -> t -> compressor -> t_hat -> task_dec2 -> x2_hat

        x3 -> task_enc3 -> t_3 ->  ↑                              -> task_dec3 -> x3_hat

        Compressor backbone outputs 3N channels (t_hat), each task decoder gets _all_ 3N channels
        """

        model = nn.ModuleDict()

        # first we need to build the task-specific input heads
        model["input_heads"] = self._build_heads(
            input_channels=self.input_channels, output_channels_per_head=self.conv_channels
        )

        # Note that we multiply self.conv_channels by the number of tasks,
        # because we will have self.conv_channels channels from each encoder head
        # and should provide N channels for each decoder head
        total_task_channels = self.conv_channels * self.n_tasks

        # these task-specific channels are stacked and passed to the default CompressAI model
        model["compressor"] = self._build_compression_backbone(
            input_channels=total_task_channels, latent_channels=self.latent_channels
        )

        # now that mixed representations should be passed to task-specific output heads
        model["output_heads"] = self._build_heads(
            total_task_channels, self.output_channels, is_deconv=True
        )

        return model

    def forward_input_heads(self, batch) -> torch.Tensor:
        """
        :param: batch - expected to be of the following form
                {
                 "task1": [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1],
                 "task2": [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2],
                  ...
                 "taskM": [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M],
                }

        :returns: embeddings after each head stacked on the channel dimension.
                  Shape - (B, sum(task_channels), W, H)
        """
        # t_is is a list of task-specific embeddings t_is = [t_1, ..., t_len(self.tasks)]
        t_is = [
            self.model["input_heads"][i](batch[task])
            for i, task in enumerate(self.tasks)
        ]

        # we now concatenate them at the channel dimensions to pass through the compressor backbone
        return torch.concat(t_is, dim=1)

    def forward_output_heads(self, batch) -> Dict[str, torch.Tensor]:
        """
        :param: batch - expected to be of the shape (B, sum(task_channels), W, H)

        :returns: Task specific predictions
                {
                 "task1": [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1,
                 "task2": [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2,
                  ...
                 "taskM": [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M,
                }
        """
        x_hats = {}

        for task_n, task in enumerate(self.tasks):
            x_hats[task] = self.model["output_heads"][task_n](batch)

        return x_hats

    def forward(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        :param: batch - expected to be of the following form
                {
                 "task1": [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1],
                 "task2": [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2],
                  ...
                 "taskM": [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M],
                }

        :returns:(
                {
                 "task1": [torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat],
                 "task2": [torch_tensor_1_2_hat, ..., torch_tensor_B_2_hat],
                  ...
                 "taskM": [torch_tensor_1_M_hat, ..., torch_tensor_B_M_hat],
                },
                {"y": y_likelihoods, "z": z_likelihoods}
            )
        """

        stacked_t = self.forward_input_heads(batch)
        # compressor_inputs = self.forward_heads(batch, "input_heads")
        # torch.concat(compressor_inputs, dim=1)

        compressor_outputs = self.model["compressor"](stacked_t)
        # compressor_outputs = self.model["compressor"](compressor_inputs)

        stacked_t_hat = compressor_outputs["x_hat"]
        # compressor_outputs_hat = compressor_outputs["x_hat"]

        # {"y": y_likelihoods, "z": z_likelihoods}
        stacked_t_likelihoods = compressor_outputs["likelihoods"]
        # compressor_outputs_likelihoods = compressor_outputs["likelihoods"]

        # x_hats = {"task1": [torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat], ... }
        x_hats = self.forward_output_heads(stacked_t_hat)

        return x_hats, stacked_t_likelihoods

    def reconstruction_loss(self, x_hat, x, loss_type: str = "mse") -> torch.Tensor:
        """
        Given a batch of images, this function returns the reconstruction loss

        :param: batch - expected to be of the following form:
                [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1]

        :param: reconstruction_loss_type - which loss function to use*

        :returns: the reconstruction loss
        """

        if loss_type == "mse":
            loss = F.mse_loss(x, x_hat, reduction="none")
            # sum over all dimensions, average over batch dimension
            # and over channels so that images with different channels have the same effect
            loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0]) / x.shape[1]
        elif loss_type == "l1":
            loss = F.l1_loss(x, x_hat, reduction="none")
            # sum over all dimensions, average over batch dimension
            # and over channels so that images with different channels have the same effect
            loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0]) / x.shape[1]
        elif loss_type == "cross-entropy":
            loss = F.cross_entropy(
                input=x_hat, target=x.squeeze(1).long(), reduction="mean"
            )
        elif loss_type == "l1":
            raise NotImplementedError("l1 not implemented yet")
        elif loss_type == "ms-ssim":
            raise NotImplementedError("ms-ssim not implemented yet")
        else:
            raise NotImplementedError(
                "reconstruction_loss_type should be one of [mse, ms-ssim]"
            )

        return loss

    def multitask_loss(self, x, x_hats, log_dir: str):
        task_losses = dict()
        logs = dict()

        for task in self.tasks:
            loss_name = task_parameters[task]["loss_function"]
            task_losses[task] = self.reconstruction_loss(
                x=x[task], x_hat=x_hats[task], loss_type=loss_name
            )

            logs[f"{log_dir}/{task}/{loss_name}"] = task_losses[task]

        task_losses = self.loss_balancer(task_losses)

        weighted_loss = sum(task_losses.values())

        return weighted_loss, logs

    def average_metrics(self, x, x_hats, log_dir: str) -> Dict[str, float]:
        logs = {}
        for metric_name, metric_function in self.metrics.items():
            for task in self.tasks:
                task_prediction = x_hats[task]
                task_target = x[task]

                # TODO: move this to some general config
                if task == "semantic":
                    value_multiplier = 1
                    data_range = 17
                    task_prediction = (
                        torch.argmax(task_prediction, dim=1).unsqueeze(1).float()
                    )
                    task_target = task_target
                else:
                    value_multiplier = 255
                    data_range = 255

                logs[f"{log_dir}/{task}/{metric_name}"] = metric_function(
                    task_prediction * value_multiplier,
                    task_target * value_multiplier,
                    data_range=data_range,
                )

        return logs

    def _get_task_likelihoods(
        self, likelihoods: Dict[str, torch.Tensor], task: str
    ) -> Dict[str, torch.Tensor]:
        """
        Note that since in this model the information about all the tasks is mixed - the only way to decompress a task
        is to use all the parts of the latent -- thus all the likelihoods refer to each task

        :param likelihoods:
        :param task:
        :return:
        """

        # TODO: document why multiple keys in likelihoods
        return likelihoods

    def _get_number_of_pixels(self, x_hats: Dict[str, torch.Tensor], task: str) -> int:
        """
        Number of pixels that the code of each task stores

        :param x_hats:
        :param task:
        :return:
        """
        B, _, H, W = torch.stack(
            [torch.tensor(x_hats[task].shape) for task in self.tasks]
        ).sum(0)
        return B * H * W

    def _compression_loss(
        self, likelihoods: Dict[str, torch.Tensor], num_pixels
    ) -> float:
        """
        :param likelihoods: dictionary with likelihoods that need to be considered for this compression estimate
        :param num_pixels: number of pixels that these likelihoods encode
        """

        compression_loss = 0

        # TODO: document why multiple keys
        for _, _likelihoods in likelihoods.items():
            compression_loss += torch.log(_likelihoods).sum()

        compression_loss /= -torch.log(torch.tensor(2))
        compression_loss /= num_pixels

        return compression_loss

    def multitask_compression_loss(
        self,
        likelihoods: Dict[str, torch.Tensor],
        x_hats: Dict[str, torch.Tensor],
        log_dir: str,
    ) -> Tuple[float, Dict[str, float]]:
        """

        :param likelihoods:
        :param x_hats:
        :param log_dir:

        :return:
        """

        total_loss = 0
        logs = dict()

        for task in self.tasks:
            task_likelihoods = self._get_task_likelihoods(likelihoods, task)

            # NOTE! That here we get the number of pixels for 1 task, but the codes in mixed latent model
            # are actually same for ALL tasks
            num_pixels = self._get_number_of_pixels(x_hats, task)

            task_loss = self._compression_loss(
                likelihoods=task_likelihoods, num_pixels=num_pixels
            )

            total_loss += task_loss

            logs[f"{log_dir}/{task}/compression_loss"] = task_loss

        return total_loss, logs

    def auxiliary_loss(self):
        return self.model["compressor"].entropy_bottleneck.loss()

    def training_step(self, batch, batch_idx):
        """
        Note that we do manual optimization and not an automatic one (which comes with pytorch lightning)
        because we have two parameter sets to optimize with two different optimizers

        :param batch:
        :param batch_idx:
        :return:
        """

        # main optimization
        main_opt, aux_opt = self.optimizers()

        x_hats, likelihoods = self.forward(batch)

        reconstruction_loss, other_rec_logs = self.multitask_loss(
            x=batch, x_hats=x_hats, log_dir="train"
        )

        compression_loss, other_comp_logs = self.multitask_compression_loss(
            likelihoods=likelihoods, x_hats=x_hats, log_dir="train"
        )

        loss = self.lmbda * reconstruction_loss + compression_loss

        log_dict = {
            "train/rec_loss": reconstruction_loss,
            "train/compression_loss": compression_loss,
            "train/loss": loss,
        }

        log_dict.update(other_rec_logs)
        log_dict.update(other_comp_logs)

        main_opt.zero_grad()
        self.manual_backward(loss)
        main_opt.step()

        aux_loss = self.auxiliary_loss()

        log_dict["train/aux_loss"] = aux_loss

        aux_opt.zero_grad()
        self.manual_backward(loss=aux_loss)
        aux_opt.step()

        metric_logs = self.average_metrics(x=batch, x_hats=x_hats, log_dir="train")

        log_dict.update(metric_logs)

        self.log_dict(
            log_dict, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    # TODO: rewrite train and val, they are almost copy-pasted
    def validation_step(self, batch, batch_idx):
        x_hats, likelihoods = self.forward(batch)

        reconstruction_loss, other_rec_logs = self.multitask_loss(
            x=batch, x_hats=x_hats, log_dir="val"
        )

        compression_loss, other_comp_logs = self.multitask_compression_loss(
            likelihoods=likelihoods, x_hats=x_hats, log_dir="val"
        )

        loss = self.lmbda * reconstruction_loss + compression_loss

        log_dict = {
            "val/rec_loss": reconstruction_loss,
            "val/compression_loss": compression_loss,
            "val/loss": loss,
        }

        log_dict.update(other_rec_logs)
        log_dict.update(other_comp_logs)

        metric_logs = self.average_metrics(x=batch, x_hats=x_hats, log_dir="val")

        log_dict.update(metric_logs)

        self.log_dict(
            log_dict, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return loss

    def get_main_parameters(self):
        return set(
            p for n, p in self.model.named_parameters() if not n.endswith(".quantiles")
        )

    def get_auxiliary_parameters(self):
        return set(
            p for n, p in self.model.named_parameters() if n.endswith(".quantiles")
        )

    def configure_optimizers(self):
        main_optimizer = torch.optim.Adam(
            self.get_main_parameters(), lr=self.learning_rate_main
        )
        auxiliary_optimizer = torch.optim.Adam(
            self.get_auxiliary_parameters(), lr=self.learning_rate_aux
        )

        return {"optimizer": main_optimizer}, {"optimizer": auxiliary_optimizer}

    def update_bottleneck_quantiles(self):
        self.model["compressor"].entropy_bottleneck.update()

    def compress(self, batch):
        self.update_bottleneck_quantiles()

        x = self.forward_input_heads(batch)
        ans = self.model["compressor"].compress(
            x
        )  # {"strings": [y_strings, z_strings],"shape": z.size()[-2:]}

        return ans

    def decompress(self, strings, shape):
        compressor = self.model["compressor"]

        compressor.entropy_bottleneck.update()

        x_hat = self.model["compressor"].decompress(strings, shape)
        return self.forward_output_heads(x_hat)
