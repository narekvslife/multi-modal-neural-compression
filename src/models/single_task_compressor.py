# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
from models import MultiTaskMixedLatentCompressor

from loss_balancing import NoWeightingStrategy


class SingleTaskCompressor(MultiTaskMixedLatentCompressor):
    """
    A compressor network which compresses only a single task.

    This is basically the compressor_backbone_class but with variable number of channels for the input.
    OR a MultiTaskMixedLatentCompressor with the list of tasks including only a single task
    """

    def __init__(
        self,
        compressor_backbone_class: type,
        tasks: Tuple[str],
        input_channels: Tuple[int],
        latent_channels: int,
        conv_channels: int,
        lmbda: float = 1,
        learning_rate_main=1e-5,
        learning_rate_aux=1e-3,
        **kwargs
    ):
        """
        :param: compressor_backbone_class - type of the backbone compression model
        :param: task - the name of the task
        :param: input_channels - tuple with the number of channels of each task
        :param: latent_channels - number of channels in the latent space
        """

        assert len(tasks) == 1

        super().__init__(
            compressor_backbone_class=compressor_backbone_class,
            tasks=tasks,
            input_channels=input_channels,
            conv_channels=conv_channels,
            latent_channels=latent_channels,
            lmbda=lmbda,
            learning_rate_main=learning_rate_main,
            learning_rate_aux=learning_rate_aux,
            **kwargs
        )

        # we don't need any multi-task loss balancing when we only have a single loss
        self.loss_balancer = NoWeightingStrategy()
