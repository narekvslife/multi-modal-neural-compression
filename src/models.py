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

from loss_balancing import UncertaintyWeightingStrategy, NoWeightingStrategy


class MultiTaskMixedLatentCompressor(pl.LightningModule):
    """
    A single MeanScaleHyperPrior network with mixed latents for all the tasks.

    This version has one encoder input head per input task with task-specific input dimensions,
    which are later mixed in the shared encoder, which produces the latents.

    These shared latents are later passed to the task-specific decoders to estimate the according input
    """

    def __init__(
            self,
            compression_model_class: type,
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
            **kwargs
    ):
        """
        :param: compression_model_class - type of the backbone compresion model
        :param: tasks - list of task names
        :param: input_channels - tuple with the number of channels of each task
        :param: conv_channels - number of channels in the convolutions of each input head.
                                which means that the compressor backbone get len(tasks) * conv_channels inputs as input
        :param: latent_channels - number of channels in the latent code (M)
        :param: lmbda - multiplier of the reconstruction loss in the total loss
        :param: pretrained - whether to use pretrained backbone compressor
        :param: quality - quality of the pretrained compressor
        """
        super().__init__()
        self.save_hyperparameters()
        self.compression_model_class = compression_model_class

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
            print("Note that pretrained models have fixed size of latents, independent of specified 'latent_channels'")

        self.automatic_optimization = False

        self.quality = quality

        self.learning_rate_main = learning_rate_main
        self.learning_rate_aux = learning_rate_aux

        self.kwargs = kwargs

        self.model: nn.ModuleDict = self._build_model()

        self.loss_balancer = UncertaintyWeightingStrategy(self.n_tasks)

        # TODO: move this to config
        self.metrics = {"psnr": peak_signal_noise_ratio, "ms-ssim": ms_ssim}

    def __build_heads(self,
                      input_channels: Union[Tuple[int], int],
                      output_channels: Union[Tuple[int], int],
                      is_deconv=False) -> nn.ModuleList:
        """
        :param: input_channels  - an integer or list of integers specifying the number of input channels of each task.
        :param: output_channels - an integer or list of integers specifying the the number output channels for each t.
        """

        if type(input_channels) == int:
            input_channels = tuple(input_channels for _ in range(self.n_tasks))

        if type(output_channels) == int:
            per_task_output_channels = [output_channels for _ in range(self.n_tasks)]
        else:
            per_task_output_channels = output_channels

        list_of_modules = []

        for t_i in range(self.n_tasks):
            i_c = input_channels[t_i]
            pto_c = per_task_output_channels[t_i]
            pti_c = per_task_output_channels[t_i] * 2 if is_deconv else per_task_output_channels[t_i] // 2

            if is_deconv:
                head = nn.Sequential(deconv(i_c, pti_c),
                                     GDN(pti_c, inverse=True),
                                     deconv(pti_c, pti_c),
                                     GDN(pti_c, inverse=True),
                                     deconv(pti_c, pto_c, stride=1),
                                     GDN(pto_c, inverse=True),
                                     conv(pto_c, pto_c, kernel_size=3, stride=1))
            else:
                head = nn.Sequential(conv(i_c, pti_c, kernel_size=3, stride=1),
                                     GDN(pti_c),
                                     conv(pti_c, pto_c),
                                     GDN(pto_c),
                                     conv(pto_c, pto_c),
                                     GDN(pto_c))

            list_of_modules.append(head)

        return nn.ModuleList(list_of_modules)

    def __get_compression_network(self, N: int, M: int) -> nn.Module:
        """
        :param N: - number of channels for each convolution layer channels to the compression model
        :param M: - number of latent channels that later will be compressed
        :return:
        """
        if self.pretrained:

            if self.compression_model_class == ScaleHyperprior:
                model = bmshj2018_hyperprior(self.quality,
                                             metric='mse',
                                             pretrained=True,
                                             progress=True,
                                             **self.kwargs)
            else:
                raise ValueError(f"No pretrained model available of class {self.compression_model_class}")

        else:
            model = self.compression_model_class(N=N,
                                                 M=M,
                                                 **self.kwargs)

        if M != model.M:
            print(f"Note that the pretrained {self.compression_model_class} has a fixed latent size M={model.M}, "
                  f"which is different from the specified M={M}")

        # This is the part that i have to deal with because in the CompressAI models
        # the default input and output dimensions is a hardcoded 3

        model.g_a[0] = conv(N, model.N)
        model.g_s[-1] = deconv(model.N, N)

        return model

    def _build_model(self) -> nn.ModuleDict:
        """
        x1 -> task_enc1 -> t_1 ->  ↓                              -> task_dec1 -> x1_hat

        x2 -> task_enc2 -> t_2 -> [+] -> t -> compressor -> t_hat -> task_dec2 -> x2_hat

        x3 -> task_enc3 -> t_3 ->  ↑                              -> task_dec3 -> x3_hat

        **kwargs - additional arguments for the "compressor" model
        """

        model = nn.ModuleDict()

        # first we need to build the task-specific input heads
        model["input_heads"] = self.__build_heads(input_channels=self.input_channels,
                                                  output_channels=self.conv_channels)

        # Note that we multiply self.conv_channels by the number of tasks,
        # because we will have self.conv_channels channels from each encoder head
        # and should provide N channels for each decoder head
        total_task_channels = self.conv_channels * self.n_tasks

        # these task-specific channels are stacked and passed to the default CompressAI model
        model["compressor"] = self.__get_compression_network(N=total_task_channels,
                                                             M=self.latent_channels)

        # now that mixed representations should be passed to task-specific output heads
        model["output_heads"] = self.__build_heads(total_task_channels, self.output_channels, is_deconv=True)

        return model

    def forward_input_heads(self, batch) -> torch.Tensor:
        """
        :param: batch - expected to be of the following form
                {
                 "task1": [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1,
                 "task2": [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2,
                  ...
                 "taskM": [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M,
                }

        :returns: embeddings after each head stacked on the channel dimension.
                  Shape - (B, sum(task_channels), W, H)
        """
        # t_is is a list of task-specific embeddings t_is = [t_1, ..., t_len(self.tasks)]
        t_is = [self.model["input_heads"][i](batch[task]) for i, task in enumerate(self.tasks)]

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

        stacked_t_preds = self.model["compressor"](stacked_t)

        stacked_t_hat = stacked_t_preds["x_hat"]
        stacked_t_likelihoods = stacked_t_preds["likelihoods"]  # {"y": y_likelihoods, "z": z_likelihoods}

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
        elif loss_type == "cross-entropy":
            loss = F.cross_entropy(input=x_hat, target=x.squeeze(1).long(), reduction="mean")
        elif loss_type == "l1":
            raise NotImplementedError("l1 not implemented yet")
        elif loss_type == "ms-ssim":
            raise NotImplementedError("ms-ssim not implemented yet")
        else:
            raise NotImplementedError("reconstruction_loss_type should be one of [mse, ms-ssim]")

        return loss

    def multitask_loss(self, x, x_hats, log_dir: str):

        task_losses = dict()
        logs = dict()

        for task in self.tasks:
            loss_name = task_parameters[task]["loss_function"]
            task_losses[task] = self.reconstruction_loss(x=x[task],
                                                         x_hat=x_hats[task],
                                                         loss_type=loss_name)

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
                    data_range = 17  # think maybe it makes sense to make this 255 since we use the image as a png anyway..?
                    task_prediction = torch.argmax(task_prediction, dim=1).unsqueeze(1).float()
                    task_target = task_target
                else:
                    value_multiplier = 255
                    data_range = 255

                logs[f"{log_dir}/{task}/{metric_name}"] = metric_function(task_prediction * value_multiplier,
                                                                          task_target * value_multiplier,
                                                                          data_range=data_range)

        return logs

    def __get_number_of_pixels(self, x_hats) -> int:
        B, _, H, W = torch.stack([torch.tensor(x_hats[task].shape) for task in self.tasks]).sum(0)
        return B * H * W

    def compression_loss(self, likelihoods: Dict[str, torch.Tensor], num_pixels: int):
        compression_loss = torch.log(likelihoods["y"]).sum()
        compression_loss += torch.log(likelihoods["z"]).sum()
        compression_loss /= -torch.log(torch.tensor(2)) * num_pixels

        return compression_loss

    def training_step(self, batch, batch_idx):
        """
        Note that we do manual optimization and not an automatic one (which comes with pytorch lightning)
        because we have two parameters to optimize
        :param batch:
        :param batch_idx:
        :return:
        """

        # main optimization
        main_opt, aux_opt = self.optimizers()

        x_hats, likelihoods = self.forward(batch)

        weighted_rec_loss, rec_logs = self.multitask_loss(x=batch, x_hats=x_hats, log_dir="train")

        compression_loss = self.compression_loss(likelihoods=likelihoods,
                                                 num_pixels=self.__get_number_of_pixels(x_hats))

        loss = self.lmbda * weighted_rec_loss + compression_loss

        log_dict = {
            "train/mse": weighted_rec_loss,
            "train/compression_loss": compression_loss,
            "train/loss": loss}

        log_dict.update(rec_logs)

        main_opt.zero_grad()
        self.manual_backward(loss)
        main_opt.step()

        aux_loss = self.model["compressor"].entropy_bottleneck.loss()

        log_dict["train/aux_loss"] = aux_loss

        aux_opt.zero_grad()
        self.manual_backward(loss=aux_loss)
        aux_opt.step()

        metric_logs = self.average_metrics(x=batch, x_hats=x_hats, log_dir="train")

        log_dict.update(metric_logs)

        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_hats, likelihoods = self.forward(batch)

        weighted_rec_loss, rec_logs = self.multitask_loss(x=batch, x_hats=x_hats, log_dir="val")

        compression_loss = self.compression_loss(likelihoods=likelihoods,
                                                 num_pixels=self.__get_number_of_pixels(x_hats))

        loss = self.lmbda * weighted_rec_loss + compression_loss

        log_dict = {
            "val/rec_loss": weighted_rec_loss,
            "val/compression_loss": compression_loss,
            "val/loss": loss,
        }

        log_dict.update(rec_logs)

        metric_logs = self.average_metrics(x=batch, x_hats=x_hats, log_dir="val")

        log_dict.update(metric_logs)

        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def get_main_parameters(self):
        return set(p for n, p in self.model.named_parameters() if not n.endswith(".quantiles"))

    def get_auxilary_parameters(self):
        return set(p for n, p in self.model.named_parameters() if n.endswith(".quantiles"))

    def configure_optimizers(self):
        main_optimizer = torch.optim.Adam(self.get_main_parameters(), lr=self.learning_rate_main)
        # lr_schedulers = {"scheduler": ReduceLROnPlateau(main_optimizer,
        #                                                 threshold=0.2,
        #                                                 factor=0.5,
        #                                                 min_lr=1e-9,
        #                                                 mode="min"),
        #                  "monitor": ["train_loss", "val_loss"]}

        auxilary_optimizer = torch.optim.Adam(self.get_auxilary_parameters(), lr=self.learning_rate_aux)
        # return {"optimizer": main_optimizer, "scheduler": lr_schedulers}, {"optimizer": auxilary_optimizer}
        return {"optimizer": main_optimizer}, {"optimizer": auxilary_optimizer}

    def update_bottleneck_quantiles(self):
        self.model["compressor"].entropy_bottleneck.update()

    def compress(self, batch):
        self.update_bottleneck_quantiles()

        x = self.forward_input_heads(batch)
        ans = self.model["compressor"].compress(x)  # {"strings": [y_strings, z_strings],"shape": z.size()[-2:]}
        return ans

    def decompress(self, strings, shape):
        compressor = self.model["compressor"]

        compressor.entropy_bottleneck.update()

        x_hat = self.model["compressor"].decompress(strings, shape)
        return self.forward_output_heads(x_hat)


class SingleTaskCompressor(MultiTaskMixedLatentCompressor):
    """
        A single compressor network which compresses a single task.

        This is basically the compression_model_class but with variable number of channels for the input.
        OR a MultiTaskMixedLatentCompressor with the list of tasks including only a single task
    """

    def __init__(
            self,
            compression_model_class: type,
            tasks: Tuple[str],
            input_channels: Tuple[int],
            latent_channels: int,
            conv_channels: int,
            lmbda: float = 1,
            pretrained: bool = False,
            quality: int = 4,
            learning_rate_main=1e-5,
            learning_rate_aux=1e-3,
            **kwargs
    ):
        """
        :param: compression_model_class - type of the backbone compression model
        :param: task - the name of the task
        :param: input_channels - tuple with the number of channels of each task
        :param: latent_channels - number of channels in the latent space
        """

        assert len(tasks) == 1

        super().__init__(compression_model_class=compression_model_class,
                         tasks=tasks,
                         input_channels=input_channels,
                         conv_channels=conv_channels,
                         latent_channels=latent_channels,
                         lmbda=lmbda,
                         pretrained=pretrained,
                         quality=quality,
                         learning_rate_main=learning_rate_main,
                         learning_rate_aux=learning_rate_aux,
                         **kwargs)

        # we don't need any multi-task loss balancing when we only have a single loss
        self.loss_balancer = NoWeightingStrategy()


class MultiTaskSeparableLatentCompressor(MultiTaskMixedLatentCompressor):

    def __init__(
            self,
            compression_model_class: type,
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
            **kwargs
    ):
        super().__init__(compression_model_class=compression_model_class,
                         tasks=tasks,
                         input_channels=input_channels,
                         output_channels=output_channels,
                         conv_channels=conv_channels,
                         latent_channels=latent_channels,
                         lmbda=lmbda,
                         pretrained=pretrained,
                         quality=quality,
                         learning_rate_main=learning_rate_main,
                         learning_rate_aux=learning_rate_aux,
                         **kwargs)

        def _build_model(self):
            pass

        def forward(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

            # todo: Note also different compression losses. I think they should be separated by channels
            pass
