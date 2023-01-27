import os

from typing import List

import torch.utils.data as data

from .transforms import default_loader, get_transform

NUM_TRAIN = 50000
NUM_VAL = 5000
NUM_TEST = 5000
EXT_DICT = {
    'depth_euclidean': 'png',
    'rgb': 'png',
    'normal': 'png',
    'semantic': 'png'
}
SEM1_CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 255)


class CLEVRDataset(data.Dataset):
    '''
    PyTorch Dataset for Taskonomized CLEVR.

    Args:
        data_path: Root directory of Taskonomy dataset
        tasks: Choice of domain ID group: List of multiple specific IDs.
        split: One of {train, val, test}. (default: train)
        image_size: Input image size. (default: 256)
    '''

    def __init__(self,
                 data_path: str,
                 tasks: List[str],
                 split: str = 'train',
                 image_size: int = 256):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.tasks = tasks
        self.image_size = image_size

    def __len__(self):
        if self.split == 'train':
            return NUM_TRAIN
        if self.split == 'val':
            return NUM_VAL
        if self.split == 'test':
            return NUM_TEST

    def __getitem__(self, index):
        task_dict = {}

        for task in self.tasks:
            path = os.path.join(self.data_path, task, self.split,
                                f'point_{index}_view_0_domain_{task}.{EXT_DICT[task]}')
            x = default_loader(path)
            t = get_transform(task=task, image_size=self.image_size)
            x = t(x)
            if task == 'principal_curvature':
                x = x[:2]
            elif task == 'rgb':
                x = x[:3]
            elif task == 'semantic':
                # for CLEVR the semseg image is a 3-channel image
                # R = shape + 10 * size = size,shape
                # G = color + 10 * material = material,color
                # B = instance
                x = x[1]  # for now, let's only consider material_color as the semantic mask

                for i, class_ in enumerate(SEM1_CLASSES):
                    x[x == class_] = i

                x = x.unsqueeze(0).float()
            elif task == 'reshading':
                x = x[[0]]
            
            task_dict[task] = x

        return task_dict
