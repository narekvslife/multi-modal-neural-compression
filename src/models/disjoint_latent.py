from typing import Tuple, Dict, Union

import torch
import torch.nn as nn

from compressai.layers import GDN
from compressai.models.utils import deconv
from models import MultiTaskCompressor


from utils import DummyModule


class MultiTaskDisjointLatentCompressor(MultiTaskCompressor):
    """
    A compressor network which compresses multiple tasks with the codes being separable by task.
    Meaning we would only need a subset of codes to decode a subset of tasks.

    This is basically the MultiTaskMixedLatentCompressor but with latents being separable.

    Schema:

    x1 -> task_enc1 -> t_1 ->  ↓                                          t_hat_1 -> task_dec1 -> x1_hat
    x2 -> task_enc2 -> t_2 -> [+] -> t -> compressor(w/o g_s function) -> t_hat_2 -> task_dec2 -> x2_hat
    x3 -> task_enc3 -> t_3 ->  ↑                                          t_hat_3 -> task_dec3 -> x3_hat


    Note that we removed g_s from the compressor backbone, by default (in CompressAI module) it is implemented s.t.
    it would mix the M=self.latent_size channels of it's latents in the decoder(g_s), which will not allow us to
    control which parts of the latent encodes which task's information (by controlling the update from backprop).

    Now - compressor backbone outputs M=self.latent_size channels, we skip the g_s function and
    each of out task decoder(output_heads) gets M // self.n_tasks channels.

    This way we get separation of which task is encoded in which channel of the latent.

    Each t_hat_i in this case is of shape M // self.n_tasks/
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
        self.latent_channels_per_task = latent_channels // len(tasks)

        super().__init__(
            compressor_backbone_class=compressor_backbone_class,
            tasks=tasks,
            input_channels=input_channels,
            output_channels=output_channels,
            conv_channels=conv_channels,
            latent_channels=latent_channels,
            lmbda=lmbda,
            learning_rate_main=learning_rate_main,
            learning_rate_aux=learning_rate_aux,
            **kwargs,
        )

        if self.latent_channels % self.n_tasks != 0:
            print(
                "!! Note that we need the same number of latent channels for each task,"
                f"but the number of latent_channels ({self.latent_channels}) "
                f"is not a multiple of the number of tasks ({self.n_tasks}) "
                f"so the latent_channels is automatically reset to ({self.latent_channels_per_task * self.n_tasks})"
            )
            self.latent_channels = self.latent_channels_per_task * self.n_tasks

    def __get_task_channels(
        self, tensor: torch.Tensor, task: str
    ) -> torch.Tensor:
        """
        This function expects a 4d tensor of type (B, C, H, W) and returns a subset of values which refer to a particular
        task. This can be done because each task takes exactly self.latent_channels_per_task channels in the latent and
        we know the order of the tasks

        :param latent_tensor:
        :param task:
        :return:
        """

        assert len(tensor.shape) == 4

        task_n = self.tasks.index(task)

        channel_l = task_n * self.latent_channels_per_task
        channel_r = (task_n + 1) * self.latent_channels_per_task

        return tensor[:, channel_l:channel_r, :, :]
    
    def _get_task_likelihoods(
        self, likelihoods: Dict[str, torch.Tensor], task: str
    ) -> Dict[str, torch.Tensor]:
        """
        In this model the information about each task is in the specific channels of the "y" latent.
        The number of channels is equal for all tasks. Note that we still use all of the "z" latents

        :param likelihoods:
        :param task:
        :return:
        """

        return {
            "y": self.__get_task_channels(likelihoods["y"], task),
            "z": likelihoods["z"],
        }

    def _build_heads(
        self,
        input_channels: Union[Tuple[int], int],
        output_channels_per_head: Union[Tuple[int], int],
        is_deconv=False,
    ) -> nn.ModuleList:
        """
        We need to override this because we removed g_s function from compressor backbone, and the remaining number
        of deconvolutions is note enough to recover the latent size to the size of initial data.

        This way, for each of the output heads we need to add additional deconv layers

        :param: input_channels  - an integer or list of integers specifying the number of input channels of each task.
        :param: output_channels_per_head - an integer or list of integers specifying the the number output channels for each t.
        """

        if is_deconv:
            module_list = nn.ModuleList()

            # g_s that we removed had the following dimensions (cahnelwise)
            # It went from self.conv_channels * n_tasks to self.latent_size having self.conv_channels in the middle
            # Because we removed the g_s part, we need to make up for it in each task-specific part. 
            # That's why each additional parts will have self.conv_channels // self.n_tasks in the middle
            conv_channels = self.conv_channels // self.n_tasks
            output_heads = super()._build_heads(self.conv_channels, output_channels_per_head, is_deconv)
        
            for i in range(self.n_tasks):
                
                # In the beginning of each output head we prepend additional deconv layers
                # to make up for the removed g_s of the compressor backbone.
                upsample = nn.Sequential(
                    # This are additional upsample layers to make up for the removed g_s
                    deconv(input_channels, conv_channels),
                    GDN(conv_channels, inverse=True),
                    deconv(conv_channels, conv_channels),
                    GDN(conv_channels, inverse=True),
                    deconv(conv_channels, conv_channels),
                    GDN(conv_channels, inverse=True),
                    deconv(conv_channels, self.conv_channels),
                    
                    # these are the decoder heads
                    output_heads[i])
                
                module_list.append(upsample)

            return module_list
        else:
            return super()._build_heads(input_channels, output_channels_per_head, is_deconv)

    def _build_model(self) -> nn.ModuleDict:

        model = nn.ModuleDict()

        model["input_heads"] = self._build_heads(
            input_channels=self.input_channels, output_channels_per_head=self.conv_channels
        )

        total_task_channels = self.conv_channels * self.n_tasks

        model["compressor"] = self._build_compression_backbone(
            input_channels=total_task_channels, latent_channels=self.latent_channels
        )
        model["compressor"].g_s = DummyModule()

        model["output_heads"] = self._build_heads(
            self.latent_channels_per_task, self.output_channels, is_deconv=True
        )

        return model

    def forward(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        stacked_t = self.forward_input_heads(batch)

        compressor_outputs = self.model["compressor"](stacked_t)

        stacked_t_hat = compressor_outputs["x_hat"]

        # {"y": y_likelihoods, "z": z_likelihoods}
        stacked_t_likelihoods = compressor_outputs["likelihoods"]

        # x_hats = {"task1": [torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat], ... }
        x_hats = {}
        for task_i, task in enumerate(self.tasks):
            task_values = self.__get_task_channels(stacked_t_hat, task)
            x_hats[task] = self.model["output_heads"][task_i](task_values)

        return x_hats, stacked_t_likelihoods
