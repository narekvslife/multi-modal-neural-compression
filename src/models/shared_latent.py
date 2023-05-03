from typing import Tuple

import torch

from models import MultiTaskDisjointLatentCompressor

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

        

    def _get_task_channels(
        self, tensor: torch.Tensor, task: str
    ) -> torch.Tensor:
        """
        This function expects a 4d tensor of type (B, C, H, W) and returns a subset of values which refer to a particular
        task. 

        Since each of our heads takes in self.latent_channels_per_task channels as input
        we can dedicate half of those channels to be the shared ones and the rest
        to be task-specific.

        For this we dedicate self.latent_channels_per_task // 2 channels for each task to the task-specific
        part of the latent. And the remainning channels to 

        :param latent_tensor:
        :param task:
        :return:
        """

        assert len(tensor.shape) == 4

        B, _, H, W = tensor.shape

        task_index = self.tasks.index(task)

        task_specific_channels_n = self.latent_channels // (self.n_tasks + 1)

        channel_l = task_index * task_specific_channels_n
        channel_r = (task_index + 1) * task_specific_channels_n

        task_specific_channels = tensor[:, channel_l: channel_r, :, :]

        shared_channels = tensor[:, -task_specific_channels_n: , :, :]

        return torch.stack([task_specific_channels, shared_channels], dim=1).reshape((B, -1, H, W))
