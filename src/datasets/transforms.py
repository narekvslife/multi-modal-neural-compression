from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Optional, List, Tuple, Dict, Union, Callable

from . import task_configs

try:
    import accimage
except ImportError:
    pass

MAKE_RESCALE_0_1_NEG1_POS1 = lambda n_chan: transforms.Normalize([0.5] * n_chan, [0.5] * n_chan)
RESCALE_0_1_NEG1_POS1 = transforms.Normalize([0.5], [0.5])  # This needs to be different depending on num out chans
MAKE_RESCALE_0_MAX_NEG1_POS1 = lambda maxx: transforms.Normalize([maxx / 2.], [maxx * 1.0])
RESCALE_0_255_NEG1_POS1 = transforms.Normalize([127.5, 127.5, 127.5], [255, 255, 255])
MAKE_RESCALE_0_MAX_0_POS1 = lambda maxx: transforms.Normalize([0.0], [maxx * 1.0])


def get_transform(task: str, image_size=Optional[int]):
    if task in ['rgb', 'normal', 'reshading']:
        transform = transform_8bit
    elif task in ['mask_valid']:
        transform = transforms.ToTensor()
    elif task in ['keypoints2d', 'keypoints3d', 'depth_euclidean', 'depth_zbuffer', 'edge_texture', 'edge_occlusion']:
        #         return transform_16bit_int
        transform = transform_16bit_single_channel
    elif task in ['principal_curvature', 'curvature']:
        transform = transform_8bit_n_channel(2)
    elif task in ['segment_semantic']:  # stored as a 1 channel image (H,W) where each pixel value is a different class
        transform = transform_dense_labels
    elif task in ['class_object', 'class_scene']:
        transform = torch.tensor
        image_size = None
    else:
        raise NotImplementedError("Unknown transform for task {}".format(task))

    if 'clamp_to' in task_configs.task_parameters[task]:
        minn, maxx = task_configs.task_parameters[task]['clamp_to']
        if minn > 0:
            raise NotImplementedError(
                "Rescaling (min1, max1) -> (min2, max2) not implemented for min1, min2 != 0 (task {})".format(task))
        transform = transforms.Compose([
            transform,
            MAKE_RESCALE_0_MAX_0_POS1(maxx)])

    if image_size is not None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transform])

    return transform


pil_to_np = lambda img: np.array(img)

# For semantic segmentation
transform_dense_labels = lambda img: torch.Tensor(np.array(img)).long()  # avoids normalizing

# Transforms to a 3-channel tensor and then changes [0,1] -> [-1, 1]
transform_8bit = transforms.Compose([
    pil_to_np,
    transforms.ToTensor(),
    #         MAKE_RESCALE_0_1_NEG1_POS1(3),
])


# Transforms to a n-channel tensor and then changes [0,1] -> [-1, 1]. Keeps only the first n-channels
def transform_8bit_n_channel(n_channel=1, crop_channels=False):
    if crop_channels:
        crop_channels_fn = lambda x: x[:n_channel] if x.shape[0] > n_channel else x
    else:
        crop_channels_fn = lambda x: x
    return transforms.Compose([
        pil_to_np,
        transforms.ToTensor(),
        crop_channels_fn,
        #             MAKE_RESCALE_0_1_NEG1_POS1(n_channel),
    ])


# Transforms to a 1-channel tensor and then changes [0,1] -> [-1, 1].
def transform_16bit_single_channel(im):
    im = transforms.ToTensor()(np.array(im))
    im = im.float() / (2 ** 15 - 1.0)
    #     return RESCALE_0_1_NEG1_POS1(im)
    return im


def transform_16bit_n_channel(n_channel=1):
    if n_channel == 1:
        return transform_16bit_single_channel  # PyTorch handles these differently
    else:
        return transforms.Compose([
            pil_to_np,
            transforms.ToTensor(),
            #             MAKE_RESCALE_0_1_NEG1_POS1(n_channel),
        ])


# from torchvision import get_image_backend, set_image_backend
# import accimage
# set_image_backend('accimage')
import torchvision.io


def default_loader(path):
    if '.npy' in path:
        return np.load(path)
    elif '.json' in path:
        raise NotImplementedError("Not sure how to load files of type: {}".format(os.path.basename(path)))
    else:
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(img.mode)
        # return img.convert(img.mode)
        # return img.convert('RGB')


# Faster than pil_loader, if accimage is available
def accimage_loader(path):
    return accimage.Image(path)


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
        [
            {
            "task1": torch_tensor_1_1,
            "task2": torch_tensor_1_2,
             ...
            "taskN": torch_tensor_1_N,
            },
            ...
            {
            "task1": torch_tensor_B_1,
            "task2": torch_tensor_B_2,
             ...
            "taskN": torch_tensor_B_N,
            }
        ]

        (3) or this type of input:
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

                # for cases (1) and (3)
                if type(input_item) == tuple:
                    input_item = input_item[0]

                item_type = type(input_item)

                # if dict, then the input was in form (1) or (2)
                if item_type == dict:
                    data_item = input_item[task]
                elif item_type == torch.Tensor:
                    data_item = input_item
                else:
                    raise NotImplementedError(f"Type {item_type} was unexpected")

                ans[task].append(data_item)

            ans[task] = torch.stack(ans[task])
        return ans if not single_task else ans[tasks]

    return my_collate_fn
