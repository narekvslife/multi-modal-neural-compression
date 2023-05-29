# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Dict

import torch
import torch.nn as nn

from models import MultiTaskCompressor


class MultiTaskMixedLatentCompressor(MultiTaskCompressor):
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
        learning_rate_main=1e-5,
        learning_rate_aux=1e-3,
        **kwargs,
    ):
        
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


    def get_model_name(self):
        return self.__class__.__name__
    
    def _get_task_likelihoods(
        self, likelihoods: Dict[str, torch.Tensor], task: str
    ) -> Dict[str, torch.Tensor]:
        """
        Note that in this setup the information about all the tasks
        is mixed - the only way to decompress a single task 
        is to use the entire latent vector.

        :param likelihoods:
        :param task:
        :return:
        """
        return likelihoods
    
    def multitask_compression_loss(
        self,
        all_likelihoods: Dict[str, torch.Tensor],
        x_hats: Dict[str, torch.Tensor],
        log_dir: str,
    ) -> Tuple[float, Dict[str, float]]:
        
        total_loss, logs = super().multitask_compression_loss(
            all_likelihoods=all_likelihoods, x_hats=x_hats, log_dir=log_dir
        )
        
        return total_loss / self.n_tasks, logs
    
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
    
    def forward_output_heads(self, stacked_latent_values):
        # x_hats = {"task1": [torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat], ... }
        x_hats = {}
        
        for task_n, task in enumerate(self.tasks):
            x_hats[task] = self.model["output_heads"][task_n](stacked_latent_values)

        return x_hats
