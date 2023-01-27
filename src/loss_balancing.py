# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
import torch.nn as nn


class NoWeightingStrategy(nn.Module):
    """
    No weighting strategy
    """

    def __init__(self, **kwargs):
        super(NoWeightingStrategy, self).__init__()

    def forward(self, task_losses):
        return task_losses


class UncertaintyWeightingStrategy(nn.Module):
    """
        Uncertainty weighting strategy for a multitask loss
    """

    def __init__(self, num_tasks: int):
        super(UncertaintyWeightingStrategy, self).__init__()

        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, torch.Tensor]]:
        """

        :param task_losses:
        :return:
        """
        losses_tensor = torch.stack(list(task_losses.values()))
        non_zero_losses_mask = (losses_tensor != 0.0)

        # calculate weighted losses
        losses_tensor = torch.exp(-self.log_vars) * losses_tensor + self.log_vars

        # if some loss was 0 (i.e. task was dropped), weighted loss should also be 0 - as no information was gained
        losses_tensor *= non_zero_losses_mask

        total_loss = losses_tensor.sum()

        # return dictionary of weighted task losses
        weighted_task_losses = task_losses.copy()
        weighted_task_losses.update(zip(weighted_task_losses, losses_tensor))

        return total_loss
