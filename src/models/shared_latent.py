from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from models import MultiTaskMixedLatentCompressor


class MultiTaskSharedLatentCompressor(MultiTaskMixedLatentCompressor):
    """
    A compressor network which compresses multiple tasks s.t. the code of each task consists of a single "shared"
    code, which will be the same for all tasks and one task-specific code for each task.

    l_s acts here as a shared latent representation (code) which is concatenated to every task-specific code l_i.
    Thus, to recover a task_i one would need to store the shared representation l_s and the task-specific l_i.

    Schema:

    x1 -> enc1 -> t_1 ->  ↓                            -> task_enc1 -> l_1 [+] l_s -> task_dec1 -> x1_hat
    x2 -> enc2 -> t_2 -> [+] -> t -> compressor -> l_s -> task_enc2 -> l_2 [+] l_s -> task_dec2 -> x2_hat
    x3 -> enc3 -> t_3 ->  ↑                            -> task_enc3 -> l_3 [+] l_s -> task_dec3 -> x3_hat
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
        pretrained: bool = False,
        quality: int = 4,
        learning_rate_main=1e-5,
        learning_rate_aux=1e-3,
        **kwargs,
    ):
        """

        :param compressor_backbone_class:
        :param tasks:
        :param input_channels:
        :param output_channels:
        :param latent_channels:
        :param conv_channels:
        :param lmbda:
        :param pretrained:
        :param quality:
        :param learning_rate_main:
        :param learning_rate_aux:
        :param gamma: multiplier in loss for the "shared" part
        :param kwargs:
        """
        super().__init__(
            compressor_backbone_class=compressor_backbone_class,
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
            **kwargs,
        )

    def _get_number_of_pixels(self, x_hats: Dict[str, torch.Tensor], task: str) -> int:
        """
        Number of pixels that the code of each task stores

        :param x_hats:
        :param task:
        :return:
        """
        B, _, H, W = x_hats[task].shape
        return B * H * W

    def __build_task_compressors(self, N: int, M: int):
        compressors = nn.ModuleList()

        for _ in self.tasks:
            compressors.append(self._build_compression_backbone(N=N, M=M))

        return compressors

    def _build_model(self) -> nn.ModuleDict:
        """
        Here our "compressor backbone" consists not of a single compressor, but rather it has 1 compressor for the
        shared latent and a separate compressor for each task

        :return:
        """
        model = nn.ModuleDict()

        # first we build the task-specific input heads
        model["input_heads"] = self._build_heads(
            input_channels=self.input_channels, output_channels=self.conv_channels
        )

        # Note that we multiply self.conv_channels by the number of tasks,
        # because after the encoder heads we will have self.conv_channels channels from __each__ of the encoder heads
        total_task_channels = self.conv_channels * self.n_tasks

        model["compressor_shared"] = self._build_compression_backbone(
            N=total_task_channels, M=self.latent_channels
        )

        model["compressors_tasks"] = self.__build_task_compressors(
            N=total_task_channels, M=self.latent_channels
        )

        # Each decoder head gets as the sum of task specific and shared latent, which still has #total_task_channels
        model["output_heads"] = self._build_heads(
            total_task_channels, self.output_channels, is_deconv=True
        )

        return model

    # TODO: rewrite this, forward_output_heads and forward_input_heads as a single general function (?)
    def __forward_task_compressors(
        self, shared_latents: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """
        :param: batch - expected to be of the shape (B, self.conv_channels * self.n_tasks, _, _)

        :returns:
            Task specific latents
                (
                 [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1],
                 [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2],
                  ...
                 [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M],
                )
                where M is the number of tasks, B is the batch size

            and

            Task specific likelihoods
                {
                 "task1": {"y": torch_tensor_1_y_likelihoods, "z": torch_tensor_1_z_likelihoods}
                 "task2": {"y": torch_tensor_2_y_likelihoods, "z": torch_tensor_2_z_likelihoods}
                  ...
                 "taskM": {"y": torch_tensor_M_y_likelihoods, "z": torch_tensor_M_z_likelihoods}
                }
        """
        task_latents = list()
        task_likelihoods = dict()

        for task_n, task in enumerate(self.tasks):
            task_preds = self.model["compressors_tasks"][task_n](shared_latents)
            task_latents.append(task_preds["x_hat"])
            task_likelihoods[task] = task_preds["likelihoods"]

        return task_latents, task_likelihoods

    def forward_output_heads(self, batch) -> Dict[str, torch.Tensor]:
        """
        :param: batch - expected to be of the shape (B, #shared_channels + #task_channels , _, _)

        where #shared_channels = #task_channels = self.conv_channels

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
            x_hats[task] = self.model["output_heads"][task_n](batch[task_n])

        return x_hats

    def forward(
        self, batch
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        :param: batch - expected to be of the following form
                {
                 "task1": [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1],
                 "task2": [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2],
                  ...
                 "taskM": [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M],
                }

        :returns:
            1. Task specific predictions:
                {
                 "task1": [torch_tensor_1_1_hat, ..., torch_tensor_B_1_hat],
                 "task2": [torch_tensor_1_2_hat, ..., torch_tensor_B_2_hat],
                  ...
                 "taskM": [torch_tensor_1_M_hat, ..., torch_tensor_B_M_hat],
                }

            2. Likelihoods
                {
                 "y": torch_tensor_shared_y_likelihoods,
                 "z": torch_tensor_shared_z_likelihoods,
                 "task1": {"y": torch_tensor_1_y_likelihoods, "z": torch_tensor_1_z_likelihoods},
                 "task2": {"y": torch_tensor_2_y_likelihoods, "z": torch_tensor_2_z_likelihoods},
                  ...
                 "taskM": {"y": torch_tensor_M_y_likelihoods, "z": torch_tensor_M_z_likelihoods}
                }
        """

        stacked_t = self.forward_input_heads(batch)

        shared_compressor_outputs = self.model["compressor_shared"](stacked_t)

        shared_latents = shared_compressor_outputs["x_hat"]
        likelihoods = shared_compressor_outputs[
            "likelihoods"
        ]  # {"y": y_likelihoods, "z": z_likelihoods}

        task_latents, task_likelihoods = self.__forward_task_compressors(shared_latents)

        # add task_latents to shared_latents and pass the sum through task-specific decoder (output) head
        for i in range(self.n_tasks):
            task_latents[i] += shared_latents

        x_hats = self.forward_output_heads(task_latents)

        likelihoods.update(task_likelihoods)
        return x_hats, likelihoods

    def auxiliary_loss(self):
        loss = self.model["compressor_shared"].entropy_bottleneck.loss()

        for i in range(self.n_tasks):
            loss += self.model["compressors_tasks"][i].entropy_bottleneck.loss()

        return loss

    def _get_task_likelihoods(
        self, likelihoods: Any, task: str
    ) -> Dict[str, torch.Tensor]:
        """

        The cost fo storing latents of task i is the cost of storing: shared_y, shared_z, task_i_y, task_i_z

        # todo: document why multiple keys in likelihoods
        :param likelihoods: {
            "y": torch.Tensor,  # shared y likelihoods
            "z": torch.Tensor,  # shared z likelihoods
            "task_1": {"y", torch.Tensor, "z", torch.Tensor},
            "task_2": {"y", torch.Tensor, "z", torch.Tensor},
             ...
            "task_M": {"y", torch.Tensor, "z", torch.Tensor}
        }

        :param task:
        :return:
        """

        return {"y_task": likelihoods[task]["y"], "z_task": likelihoods[task]["z"]}

    def multitask_compression_loss(
        self,
        likelihoods: Dict[str, torch.Tensor],
        x_hats: Dict[str, torch.Tensor],
        log_dir: str,
    ) -> Tuple[float, Dict[str, float]]:
        # this compression loss includes only task-specific losses, but we also need to consider the shared part
        compression_loss, logs = super().multitask_compression_loss(
            likelihoods, x_hats, log_dir
        )

        # the shared codes are same for ALL input/output tasks
        total_pixels = sum(
            (self._get_number_of_pixels(x_hats, task) for task in self.tasks)
        )

        shared_code_compression_loss = self._compression_loss(
            {"y": likelihoods["y"], "z": likelihoods["z"]}, total_pixels
        )

        logs[f"{log_dir}/shared/compression_loss"] = shared_code_compression_loss

        return compression_loss, logs

    # TODO: does this need to be redefined? Probably not, seems like we can optimize all the quantiles using one opt.
    # def configure_optimizers(self):
    #     pass
