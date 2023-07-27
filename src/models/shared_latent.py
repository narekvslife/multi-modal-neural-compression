from typing import Dict, Tuple

import torch
import torch.nn as nn

from models import MultiTaskDisjointLatentCompressor
from utils import DummyModule

class MultiTaskSharedLatentCompressor(MultiTaskDisjointLatentCompressor):
    """
    A compressor network which compresses multiple tasks s.t. the code of each task consists of a single "shared"
    code, which will be the same for all tasks and one task-specific code for each task.

    l_s acts here as a shared latent representation (code) which is concatenated to every task-specific code l_i.
    Thus, to recover a task_i one would need to store the shared representation l_s and the task-specific l_i.

    Schema:

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
        n_tasks = len(tasks)
        if latent_channels % (n_tasks + 1) != 0:    
            new_latent_channels = latent_channels // (n_tasks + 1) * (n_tasks + 1)
            print("\n\n Note that we're overriding the number of channels"
                  f"from {latent_channels} to {new_latent_channels}"
                  "To make sure we can dedicate even number of channels"
                  "to each task and the shared part\n\n")
            latent_channels = new_latent_channels
        

        self.task_specific_channels_n = latent_channels // (n_tasks + 1)

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
            self.task_specific_channels_n * 2, self.output_channels, is_deconv=True
        )

        return model

    def __get_shared_channels(
        self, tensor: torch.Tensor
    ) -> torch.Tensor:
        
        return tensor[:, -self.task_specific_channels_n: , :, :]

    def _get_task_channels(
        self, tensor: torch.Tensor, task: str
    ) -> torch.Tensor:
        """
        This function expects a 4d tensor of type (B, C, H, W) and returns a subset of values which refer to a particular
        task. 

        :param latent_tensor:
        :param task:
        :return:
        """

        assert len(tensor.shape) == 4

        task_index = self.tasks.index(task)

        channel_l = task_index * self.task_specific_channels_n
        channel_r = (task_index + 1) * self.task_specific_channels_n

        return tensor[:, channel_l: channel_r, :, :]
    
    def _get_task_likelihoods(
        self, likelihoods: Dict[str, torch.Tensor], task: str
    ) -> Dict[str, torch.Tensor]:

        if task == "shared":
            return  self.__get_shared_channels(likelihoods["y"])
        else:
            return self._get_task_channels(likelihoods, task)

    def multitask_compression_loss(
        self,
        all_likelihoods: Dict[str, torch.Tensor],
        x_hats: Dict[str, torch.Tensor],
        log_dir: str,
    ) -> Tuple[float, Dict[str, float]]:

        # at this point we only have task-specific parts, without the shared part
        total_loss, logs = super().multitask_compression_loss(all_likelihoods, x_hats, log_dir)

        shared_likelihoods = self._get_task_likelihoods(all_likelihoods, "shared")
        shared_compression_loss = self._bits_per_pixel(
                likelihoods=shared_likelihoods,
                num_pixels=self._get_number_of_pixels()
            )
        
        logs[f"{log_dir}/shared/compression_loss"] = shared_compression_loss
        total_loss += shared_compression_loss
        
        # --- TODO: This part is very ScaleHyperprior specific 
        task_num_pixels = self._get_number_of_pixels(x_hats, self.tasks[0])
        z_compression_loss = self._bits_per_pixel(
                    likelihoods=all_likelihoods['z'],
                    num_pixels=task_num_pixels)
        
        logs[f"{log_dir}/shared/compression_loss"] += z_compression_loss
        # --- 

        return total_loss, logs
    
    def forward_output_heads(self, stacked_latent_values):
        B, _, H, W = stacked_latent_values.shape

        x_hats = {}

        for task_i, task in enumerate(self.tasks):
            task_values = self._get_task_channels(stacked_latent_values, task)
            shared_values = self.__get_shared_channels(stacked_latent_values)

            stacked_values = torch.stack([task_values, shared_values], dim=1).reshape((B, -1, H, W))

            x_hats[task] = self.model["output_heads"][task_i](stacked_values)

        return x_hats
