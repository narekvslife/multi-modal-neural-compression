from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from compressai.models.utils import conv, deconv
from compressai.layers import GDN


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
            **kwargs
    ):
        """
        :param: compression_model_class - type of the backbone compresion model
        :param: tasks - list of task names
        :param: input_channels - tuple with the number of channels of each task
        :param: latent_channels - number of channels in the latent space
        """
        super().__init__()
        self.compression_model_class = compression_model_class

        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.input_channels = input_channels

        assert self.n_tasks == len(self.input_channels)

        self.latent_channels = latent_channels
        self.model: nn.ModuleDict = self.__build_model(**kwargs)

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
                                     GDN(pto_c, inverse=True))
            else:
                head = nn.Sequential(conv(i_c, pti_c),
                                     GDN(pti_c),
                                     conv(pti_c, pto_c),
                                     GDN(pto_c))

            list_of_modules.append(head)

        return nn.ModuleList(list_of_modules)

    def __build_model(self, **kwargs) -> nn.ModuleDict:
        """
        x1 -> task_enc1 -> t_1 ->  ↓                                      -> task_dec1 -> x1_hat

        x2 -> task_enc2 -> t_2 -> [+] -> t -> compressor_network -> t_hat -> task_dec2 -> x2_hat

        x3 -> task_enc3 -> t_3 ->  ↑                                      -> task_dec3 -> x3_hat

        **kwargs - additional arguments for the compressor_network model
        """

        model = nn.ModuleDict()
        # We chose a 1/2 of the size of the latent to be the sum of the channels of the input specific embeddings
        # which means that each task t_i gets: latent_channels // 2 // self.n_tasks channels

        # !Note! that there is no particular reason for this choice other then the automatic
        # control of the latent size to be descrete and always equal for all tasks
        # TODO: maybe change this idk...
        t_i_channels: int = self.latent_channels // 2 // self.n_tasks

        # first we need to build the task-specific input heads
        model["input_heads"] = self.__build_heads(input_channels=self.input_channels,
                                                  output_channels=t_i_channels)

        total_task_channels = t_i_channels * self.n_tasks

        # these task-specific are stacked and passed to the default compressai model
        N = total_task_channels
        M = self.latent_channels
        model["compressor_network"] = self.compression_model_class(N=N,
                                                                   M=M,
                                                                   **kwargs)
        # this is the part that i have to deal with because in the compressai
        # the default input and output dimensions is a hardcoded 3, rather than N!
        model["compressor_network"].g_a = nn.Sequential(
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        model["compressor_network"].g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
        )

        # now that mixed representations should be passed to task-specific output heads
        model["output_heads"] = self.__build_heads(N, self.input_channels, is_deconv=True)

        return model

    def forward(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        :param: batch - expected to be of the following form
                {
                 "task1": [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1,
                 "task2": [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2,
                  ...
                 "taskM": [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M,
                }

        :returns:(
                {
                 "task1": [torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat,
                 "task2": [torch_tensor_1_2_hat, ..., torch_tensor_B_2_hat,
                  ...
                 "taskM": [torch_tensor_1_M_hat, ..., torch_tensor_B_M_hat,
                },
                {"y": y_likelihoods, "z": z_likelihoods}
            )
        """

        # t_is is a list of task-specific embeddings t_is = [t_1, ..., t_len(self.tasks)]
        t_is = [self.model["input_heads"][i](batch[task]) for i, task in enumerate(self.tasks)]

        # we now concatenate them at the channel dimensions to pass through the compressor backbone
        t = torch.concat(t_is, dim=1)

        # now pass through the main compressor network
        t_preds = self.model["compressor_network"](t)

        t_hat = t_preds["x_hat"]

        # {"y": y_likelihoods, "z": z_likelihoods}
        likelihoods = t_preds["likelihoods"]

        x_hats = {}
        for task_n, task in enumerate(self.tasks):
            x_hats[task] = self.model["output_heads"][task_n](t_hat)

        return x_hats, likelihoods

    def _get_reconstruction_loss(self, x, x_hat, loss_type: str = "mse") -> torch.Tensor:
        """
        Given a batch of images, this function returns the reconstruction loss

        :param: batch - expected to be of the following form:
                {
                 "task1": [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1,
                 "task2": [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2,
                  ...
                 "taskM": [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M,
                }
        :param: reconstruction_loss_type - which loss function to use*

        *Note that here we assume using the same loss for all the tasks.
         For current implementation, where all the input tasks are images of sort
         this makes complete sense.

         :returns:
        """

        if loss_type == "mse":
            loss = F.mse_loss(x, x_hat, reduction="none")
            # sum over all dimensions, average over batch dimension
            loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])

        elif loss_type == "ms-ssim":
            raise NotImplementedError("ms-ssim not implemented yet")
        else:
            raise NotImplementedError("reconstruction_loss_type should be one of [mse, ms-ssim]")

        return loss

    def __get_number_of_pixels(self, x_hats):
        if type(x_hats) == dict:
            return torch.stack([torch.tensor(x_hats[task].shape) for task in self.tasks]).sum(0)
        else:
            return x_hats.shape

    def _get_compression_loss(self, x, x_hats, likelihoods):

        B, _, H, W = self.__get_number_of_pixels(x_hats)

        num_pixels = B * H * W
        compression_loss = torch.log(likelihoods["y"]).sum()
        compression_loss += torch.log(likelihoods["z"]).sum()
        compression_loss /= -torch.log(torch.tensor(2)) * num_pixels

        return compression_loss

    def get_loss(self, batch, lmbd=0.5):
        x_hats, likelihoods = self.forward(batch)

        reconstruction_loss = 0
        compression_loss = 0

        for task in self.tasks:
            reconstruction_loss += self._get_reconstruction_loss(x=batch[task],
                                                                 x_hat=x_hats[task],
                                                                 loss_type="mse")

        compression_loss += self._get_compression_loss(x=batch, x_hats=x_hats, likelihoods=likelihoods)
        return reconstruction_loss + lmbd * compression_loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("test_loss", loss)

    def get_main_parameters(self):
        return set(p for n, p in self.model.named_parameters() if not n.endswith(".quantiles"))

    def get_auxilary_parameters(self):
        return set(p for n, p in self.model.named_parameters() if n.endswith(".quantiles"))

    def configure_optimizers(self):
        main_optimizer = torch.optim.Adam(self.get_main_parameters(), lr=1e-3)
        auxilary_optimizer = torch.optim.Adam(self.get_auxilary_parameters(), lr=1e-3)

        return {"main_optimizer": main_optimizer,
                "auxilary_optimizer": auxilary_optimizer,
                "monitor": "val_loss"}

    def compress(self, batch):
        pass

    def decompress(self, batch):
        pass


class SingleTaskCompressor(MultiTaskMixedLatentCompressor):
    """
        A single MeanScaleHyperPrior network which compresses a single task.

        This is basically MeanScaleHyperPrior but with variable number of channels for the input.
        OR a MultiTaskMixedLatentCompressor with the list of tasks including only a single task
    """

    def __init__(
            self,
            compression_model_class: type,
            task: str,
            input_channels: int,
            latent_channels: int,
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
                         **kwargs)

        self.task = task

    def get_loss(self, batch, lmbd=0.5):
        x_hats, likelihoods = self.forward(batch)

        reconstruction_loss = 0
        compression_loss = 0

        reconstruction_loss += self._get_reconstruction_loss(x=batch,
                                                             x_hat=x_hats,
                                                             loss_type="mse")

        compression_loss += self._get_compression_loss(x=batch, x_hats=x_hats, likelihoods=likelihoods)
        return reconstruction_loss + lmbd * compression_loss

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param: batch - expected to be of the following form:
                [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1]

        :returns: ([torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat], {"y": y_likelihoods, "z": z_likelihoods})
        """
        batch_task_tensor = dict()
        batch_task_tensor[self.task] = batch
        x_hats, likelihoods = super().forward(batch_task_tensor)

        return x_hats[self.task], likelihoods
