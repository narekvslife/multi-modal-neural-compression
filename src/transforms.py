from typing import List, Dict, Tuple, Callable, Union

import torch

from torchvision import transforms
import torchvision.transforms.functional as TF

from constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def normalize_uint16(image):
    return image / 65535


def make_transforms(tasks):
    # per-task transforms
    transforms_dict = {}

    for task in tasks:
        task_transforms = [transforms.Resize((256, 256), interpolation=TF.InterpolationMode.NEAREST),
                           transforms.ToTensor()]

        if task == "rgb":
            task_transforms.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                                        std=IMAGENET_DEFAULT_STD))
        elif task == "depth":
            task_transforms.append(normalize_uint16)

        transforms_dict[task] = transforms.Compose(task_transforms)

    return transforms_dict


def make_collate_fn(tasks: Union[List[str], str]) -> Callable:
    """
    :param tasks: either a list or a single task
    :return: function which preprocesses a bach for multiple tasks or for a single one
    """
    global my_collate_fn

    def my_collate_fn(list_of_inputs: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
        """
        Assume B = len(list_of_inputs) is the batch size
               M = (list_of_inputs.keys()) is the number of tasks

        We want to turn the input *list* of B dictionaries with M keys each
        into an output *dictionary* of M keys with B values each, so that we
        have per-task batches.

        We want to turn

        (1) this type of input:
        [
            ({
            "task1": torch_tensor_1_1,
            "task2": torch_tensor_1_2,
             ...
            "taskN": torch_tensor_1_N,
            },
            class_number1),
            ...

            ({
            "task1": torch_tensor_B_1,
            "task2": torch_tensor_B_2,
             ...
            "taskN": torch_tensor_B_N,
            },
            class_numberN),
        ]

        (2) or this type of input:
        ([torch_tensor_1, torch_tensor_2, ..., torch_tensor_B], class_number1)

        Into this type of output:
        {
            "task1": [torch_tensor_1_1, torch_tensor_2_1, ..., torch_tensor_B_1],
            "task2": [torch_tensor_1_2, torch_tensor_2_2, ..., torch_tensor_B_2],
             ...
            "taskM": [torch_tensor_1_M, torch_tensor_2_M, ..., torch_tensor_B_M]
        }
        """
        single_task = type(tasks) == str

        task_list = [tasks, ] if single_task else tasks

        ans = {task: [] for task in task_list}  # TODO: change this to a global var with set of tasks (?)

        # to all the leetcode bros: im so sorry for these loops :c
        for task in task_list:
            for input_item in list_of_inputs:

                item_type = type(input_item[0])

                # if dict, then the input was in form (1)
                if item_type == dict:
                    data_item = input_item[0][task]
                elif item_type == torch.Tensor:
                    data_item = input_item[0]
                else:
                    raise NotImplementedError(f"Type {item_type} was unexpected")

                ans[task].append(data_item)

            ans[task] = torch.stack(ans[task])

        return ans if not single_task else ans[tasks]
    return my_collate_fn
