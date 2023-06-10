# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from pytorch_msssim import ms_ssim
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio

from compressai.layers import GDN
from compressai.models.utils import conv, deconv
from compressai.models.base import get_scale_table

from datasets import task_configs

from loss_balancing import UncertaintyWeightingStrategy


class MultiTaskCompressor(pl.LightningModule):
    """
    Compressor network with multiple input/output heads. 

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

        self.automatic_optimization = False

        self.learning_rate_main = learning_rate_main
        self.learning_rate_aux = learning_rate_aux

        self.kwargs = kwargs

        self.model: nn.ModuleDict = self._build_model()

        self.loss_balancer = UncertaintyWeightingStrategy(self.n_tasks)

        # TODO: move this to config
        self.metrics = {"psnr": peak_signal_noise_ratio, "ms-ssim": ms_ssim}

    def get_model_name(self):
        return self.__class__.__name__

    def _get_number_of_pixels(self, x_hats: Dict[str, torch.Tensor], task: str) -> int:
        """
        Number of pixels that the code of each task stores

        :param x_hats:
        :param task:
        :return:
        """
        B, _, H, W = x_hats[task].shape

        return B * H * W

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

        model = self.compressor_backbone_class(N=input_channels, M=latent_channels, **self.kwargs)

        # This is the part that i have to deal with because in the CompressAI models
        # the default input and output dimension is a hardcoded 3
        model.g_a[0] = conv(input_channels,input_channels)
        model.g_s[-1] = deconv(input_channels, input_channels)

        return model

    def _build_model(self) -> nn.ModuleDict:

        # This is architecture specific
        raise NotImplementedError()

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

        # forward() is model-specific
        raise NotImplementedError()

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
            # and over channels so that images with different number of channels have the same effect
            loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0]) / x.shape[1]
        elif loss_type == "l1":
            loss = F.l1_loss(x, x_hat, reduction="none")
            # sum over all dimensions, average over batch dimension
            # and over channels so that images with different number of channels have the same effect
            loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0]) / x.shape[1]
        elif loss_type == "cross-entropy":
            loss = F.cross_entropy(
                input=x_hat, target=x.squeeze(1).long(), reduction="mean"
            )
        elif loss_type == "ms-ssim":
            raise NotImplementedError("ms-ssim not implemented yet")
        else:
            raise NotImplementedError(
                "reconstruction_loss_type should be one of [mse, ms-ssim]"
            )

        return loss

    def multitask_reconstruction_loss(self, x, x_hats, log_dir: str):
        task_losses = dict()
        logs = dict()

        for task in self.tasks:
            loss_name = task_configs.task_parameters[task]["loss_function"]
            task_losses[task] = self.reconstruction_loss(
                x=x[task], x_hat=x_hats[task], loss_type=loss_name
            )

            logs[f"{log_dir}/{task}/{loss_name}"] = task_losses[task]

        task_losses = self.loss_balancer(task_losses)

        weighted_loss = sum(task_losses.values())

        return weighted_loss, logs

    def _single_task_compression_loss(
        self, likelihoods: torch.Tensor, num_pixels
    ) -> float:
        """
        Compute Bits Per Pixel

        :param likelihoods: dictionary with likelihoods that need to be considered for this compression estimate
        :param num_pixels: number of pixels that these likelihoods encode
        """

        compression_loss = 0

        compression_loss = torch.log(likelihoods).sum()

        compression_loss /= -torch.log(torch.tensor(2))
        compression_loss /= num_pixels

        return compression_loss
    
    def _get_task_likelihoods(
        self, likelihoods: Dict[str, torch.Tensor], task: str
    ) -> Dict[str, torch.Tensor]:
        
        # _get_task_likelihoods(...) is specific to the architecture
        raise NotImplementedError()

    def multitask_compression_loss(
        self,
        all_likelihoods: Dict[str, torch.Tensor],
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

            task_likelihoods = self._get_task_likelihoods(all_likelihoods, task)

            task_num_pixels = self._get_number_of_pixels(x_hats, task)

            # --- TODO: This part is very ScaleHyperprior specific 
            task_compression_loss = 0
            for latent_type in ('y', 'z'):
                task_compression_loss += self._single_task_compression_loss(
                    likelihoods=task_likelihoods[latent_type],
                    num_pixels=task_num_pixels
                )
            # --- 

            logs[f"{log_dir}/{task}/compression_loss"] = task_compression_loss
    
            total_loss += task_compression_loss

        # --- TODO: This part is very ScaleHyperprior specific 
        total_loss -= self._single_task_compression_loss(
                likelihoods=all_likelihoods["z"],
                num_pixels=task_num_pixels
            ) * (self.n_tasks - 1)
        # --- 
        
        # when computing _single_task_compression_loss for each task
        # we averaged over the number of pixels in each task
        # but when computing the total BPP 
        # we should average over the total number of pixels
        total_loss /= self.n_tasks

        return total_loss, logs

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

    def auxiliary_loss(self):
        return self.model["compressor"].entropy_bottleneck.loss()

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

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            main_optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-8
        )
        
        auxiliary_optimizer = torch.optim.Adam(
            self.get_auxiliary_parameters(), lr=self.learning_rate_aux
        )

        return {"optimizer": main_optimizer,
                "lr_scheduler": {"scheduler": sch}
                },\
                {"optimizer": auxiliary_optimizer}

    def __step(self, batch, is_train):

        if is_train:
            log_dir = "train"
        else:
            log_dir = "val"

        x_hats, likelihoods = self.forward(batch)

        reconstruction_loss, other_rec_logs = self.multitask_reconstruction_loss(
            x=batch, x_hats=x_hats, log_dir=log_dir
        )

        compression_loss, other_comp_logs = self.multitask_compression_loss(
            all_likelihoods=likelihoods, x_hats=x_hats, log_dir=log_dir
        )

        loss = self.lmbda * reconstruction_loss + compression_loss

        log_dict = {
            f"{log_dir}/rec_loss": reconstruction_loss,
            f"{log_dir}/compression_loss": compression_loss,
            f"{log_dir}/loss": loss,
        }
        
        log_dict.update(other_rec_logs)
        log_dict.update(other_comp_logs)

        if is_train:
            main_opt, aux_opt = self.optimizers()

            main_opt.zero_grad()
            self.manual_backward(loss)
            main_opt.step()

            # Auxilary optimization
            aux_loss = self.auxiliary_loss()

            log_dict[f"{log_dir}/aux_loss"] = aux_loss

            aux_opt.zero_grad()
            self.manual_backward(loss=aux_loss)
            aux_opt.step()

            # Step for learning rate scheduler
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()

        metric_logs = self.average_metrics(x=batch, x_hats=x_hats, log_dir=log_dir)

        log_dict.update(metric_logs)

        self.log_dict(
            log_dict, on_step=is_train, on_epoch=not is_train, sync_dist=True, prog_bar=True
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.__step(batch, is_train=True)


    def validation_step(self, batch, batch_idx):
        return self.__step(batch, is_train=False)


    def update_bottleneck_values(self):
        self.model["compressor"].gaussian_conditional.update_scale_table(get_scale_table())
        entr = self.model["compressor"].entropy_bottleneck.update()
        return  entr
    
    def forward(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        stacked_t = self.forward_input_heads(batch)

        compressor_outputs = self.model["compressor"](stacked_t)

        stacked_t_hat = compressor_outputs["x_hat"]

        # {"y": y_likelihoods, "z": z_likelihoods}
        stacked_t_likelihoods = compressor_outputs["likelihoods"]

        # x_hats = {"task1": [torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat], ... }
        x_hats = self.forward_output_heads(stacked_t_hat)
        
        return x_hats, stacked_t_likelihoods

    def compress(self, batch, print_info: bool = False):
        x = self.forward_input_heads(batch)
        ans = self.model["compressor"].compress(x)  

        if print_info:
            number_of_bytes = 0
            for latents in ans["strings"]:
                for bit_string in latents:
                    number_of_bytes += len(bit_string)

            B, _, H, W = batch[self.tasks[0]].shape

            bpp = number_of_bytes * 8 / B / H / W / self.n_tasks
            print(f"Number of actual bytes in a string is: {number_of_bytes}, which gives a BPP = {bpp:.2f}")

            stacked_t = self.forward_input_heads(batch)

            stacked_t_likelihoods = self.model["compressor"](stacked_t)["likelihoods"]

            compression_loss, _ = self.multitask_compression_loss(
                all_likelihoods=stacked_t_likelihoods, x_hats=batch, log_dir=""
            )
            print(f"Estimated BPP (compression loss) is: {compression_loss.item():.2f}")

        return ans, number_of_bytes

    def decompress(self, strings, shape):        
        # stacked_latent_values = self.model["compressor"].decompress(strings, shape)["x_hat"]

        # had to rewrite here the decompress() function 
        # from ScaleHyperprior because in the end they
        # do .clamp(0, 1) which messes everything up for us
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.model["compressor"].entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.model["compressor"].h_s(z_hat)
        indexes = self.model["compressor"].gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.model["compressor"].gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.model["compressor"].g_s(y_hat)

        return self.forward_output_heads(x_hat)
