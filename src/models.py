from typing import Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from compressai.models import ScaleHyperprior

from pytorch_msssim import ms_ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio

from compressai.layers import GDN
from compressai.zoo import bmshj2018_hyperprior
from compressai.models.utils import conv, deconv


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
            latent_channels: int,
            lmbda: float = 0.5,
            pretrained: bool = False,
            quality: int = 4,
            **kwargs
    ):
        """
        :param: compression_model_class - type of the backbone compresion model
        :param: tasks - list of task names
        :param: input_channels - tuple with the number of channels of each task
        :param: latent_channels - number of channels in the latent space
        :param: lmbda - multiplier of the compression loss in the total loss
=        """
        super().__init__()
        self.save_hyperparameters()
        self.compression_model_class = compression_model_class

        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.input_channels = input_channels
        assert self.n_tasks == len(self.input_channels)

        self.latent_channels = latent_channels

        self.lmbda = lmbda

        self.pretrained = pretrained

        self.automatic_optimization = False

        self.quality = quality

        if self.pretrained:
            print("Note that pretrained models have their own (fixed) number of latent channels independent of specified M")

        self.kwargs = kwargs

        self.model: nn.ModuleDict = self._build_model()

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
                                     deconv(pti_c, pto_c),
                                     GDN(pto_c, inverse=True),
                                     conv(pto_c, pto_c, kernel_size=3, stride=1))
            else:
                head = nn.Sequential(conv(i_c, pti_c),
                                     GDN(pti_c),
                                     conv(pti_c, pto_c),
                                     GDN(pto_c))

            list_of_modules.append(head)

        return nn.ModuleList(list_of_modules)

    def __get_compression_network(self, N: int, M: int) -> nn.Module:
        """

        :param N: - number of input channels to the compression model
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

        # this is the part that i have to deal with because in the compressai models
        # the default input and output dimensions is a hardcoded 3, rather than N!
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
        # We chose a 1/2 of the size of the latent to be the sum of the channels of the input specific embeddings
        # which means that each task t_i gets: latent_channels // 2 // self.n_tasks channels

        # !Note! that there is no particular reason for this choice other then the automatic
        # control of the latent size to be discrete and always equal for all tasks
        # TODO: maybe change this idk...
        t_i_channels: int = self.latent_channels // 2 // self.n_tasks

        # first we need to build the task-specific input heads
        model["input_heads"] = self.__build_heads(input_channels=self.input_channels,
                                                  output_channels=t_i_channels)

        total_task_channels = t_i_channels * self.n_tasks

        # these task-specific are stacked and passed to the default compressai model
        model["compressor"] = self.__get_compression_network(N=total_task_channels, M=self.latent_channels)

        # now that mixed representations should be passed to task-specific output heads
        model["output_heads"] = self.__build_heads(total_task_channels, self.input_channels, is_deconv=True)

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

    def forward_output_heads(self, batch) -> torch.Tensor:
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

        # now pass through the main compressor network
        stacked_t_preds = self.model["compressor"](stacked_t)

        stacked_t_hat = stacked_t_preds["x_hat"]
        stacked_t_likelihoods = stacked_t_preds["likelihoods"]  # {"y": y_likelihoods, "z": z_likelihoods}

        # {"task1": [torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat], ... }
        x_hats = self.forward_output_heads(stacked_t_hat)

        return x_hats, stacked_t_likelihoods

    def reconstruction_loss(self, x, x_hat, loss_type: str = "mse") -> torch.Tensor:
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

        elif loss_type == "ms-ssim":
            raise NotImplementedError("ms-ssim not implemented yet")
        else:
            raise NotImplementedError("reconstruction_loss_type should be one of [mse, ms-ssim]")

        return loss

    # TODO: the loss/metric functions below should be merged into one function, the code is copy-pasted
    def multitask_reconstruction_loss(self, x, x_hats):
        reconstruction_loss = 0
        for task in self.tasks:
            reconstruction_loss += self.reconstruction_loss(x=x[task],
                                                            x_hat=x_hats[task],
                                                            loss_type="mse")
        return reconstruction_loss / self.n_tasks

    def multitask_psnr(self, x, x_hats):
        psnr = 0
        for task in self.tasks:
            psnr += peak_signal_noise_ratio(x_hats[task] * 255, x * 255, data_range=255)

        return psnr / self.n_tasks

    def multitask_ms_ssim(self, x, x_hats):
        msssim = 0
        for task in self.tasks:
            msssim += ms_ssim(x_hats[task] * 255, x * 255, data_range=2550)

        return msssim / self.n_tasks

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

        rec_loss = self.multitask_reconstruction_loss(x=batch, x_hats=x_hats)

        compression_loss = self.compression_loss(likelihoods=likelihoods,
                                                 num_pixels=self.__get_number_of_pixels(x_hats))

        loss = rec_loss + self.lmbda * compression_loss

        main_opt.zero_grad()
        self.manual_backward(loss)
        main_opt.step()

        aux_loss = self.model["compressor"].entropy_bottleneck.loss()

        aux_opt.zero_grad()
        self.manual_backward(loss=aux_loss)
        aux_opt.step()

        log_dict = {
            "train/rec_loss": rec_loss,
            "train/compression_loss": compression_loss,
            "train/loss": loss,
            "train/psnr": self.multitask_psnr(x=batch, x_hats=x_hats),
            "train/ms-ssim": self.multitask_ms_ssim(x=batch, x_hats=x_hats),

            "aux/loss": aux_loss,
        }

        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_hats, likelihoods = self.forward(batch)

        rec_loss = self.multitask_reconstruction_loss(x=batch, x_hats=x_hats)

        compression_loss = self.compression_loss(likelihoods=likelihoods,
                                                 num_pixels=self.__get_number_of_pixels(x_hats))

        loss = rec_loss + self.lmbda * compression_loss

        log_dict = {
            "val/rec_loss": rec_loss,
            "val/compression_loss": compression_loss,
            "val/loss": loss,
            "val/psnr": self.multitask_psnr(x=batch, x_hats=x_hats),
            "val/ms_ssim": self.multitask_ms_ssim(x=batch, x_hats=x_hats)
        }

        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def get_main_parameters(self):
        return set(p for n, p in self.model.named_parameters() if not n.endswith(".quantiles"))

    def get_auxilary_parameters(self):
        return set(p for n, p in self.model.named_parameters() if n.endswith(".quantiles"))

    def configure_optimizers(self):
        main_optimizer = torch.optim.Adam(self.get_main_parameters(), lr=1e-4)
        lr_schedulers = {"scheduler": ReduceLROnPlateau(main_optimizer, threshold=5, factor=0.5, min_lr=1e-9), "monitor": ["train_loss", "val_loss"]}

        auxilary_optimizer = torch.optim.Adam(self.get_auxilary_parameters(), lr=1e-4)
        return {"optimizer": main_optimizer, "scheduler": lr_schedulers}, {"optimizer": auxilary_optimizer}

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
            task: str,
            input_channels: int,
            latent_channels: int,
            lmbda: float = 0.5,
            pretrained: bool = False,
            quality: int = 4,
            **kwargs
    ):
        """
        :param: compression_model_class - type of the backbone compresion model
        :param: task - the name of the task
        :param: input_channels - tuple with the number of channels of each task
        :param: latent_channels - number of channels in the latent space
        """
        super().__init__(compression_model_class=compression_model_class,
                         tasks=(task,),
                         input_channels=(input_channels, ),
                         latent_channels=latent_channels,
                         lmbda=lmbda,
                         pretrained=pretrained,
                         quality=quality,
                         **kwargs)

        self.task = task

    def __get_number_of_pixels(self, x) -> int:
        B, _, H, W = x.shape
        return B * H * W

    def multitask_reconstruction_loss(self, x, x_hats):
        return super().reconstruction_loss(x, x_hats[self.task])

    def multitask_psnr(self, x, x_hats):
        return peak_signal_noise_ratio(preds=x_hats[self.task] * 255,
                                       target=x * 255,
                                       data_range=255)

    def multitask_ms_ssim(self, x, x_hats):
        return ms_ssim(x * 255, x_hats[self.task] * 255, data_range=255)

    def forward_input_heads(self, batch) -> torch.Tensor:
        """
        :param: batch - expected to be of the following form:
                [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1]

        :returns: ([torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat], {"y": y_likelihoods, "z": z_likelihoods})
        """
        batch_task_tensor = dict()
        batch_task_tensor[self.task] = batch
        stacked_task_embd = super().forward_input_heads(batch_task_tensor)

        return stacked_task_embd
