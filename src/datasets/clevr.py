import os

from typing import List

import torch.utils.data as data

from .transforms import default_loader, get_transform

NUM_TRAIN = 50000
NUM_VAL = 5000
NUM_TEST = 5000
EXT_DICT = {
    'curvature_meshes': 'ply',
    'depth_euclidean': 'png',
    'depth_zbuffer': 'png',
    'edge_occlusion': 'png',
    'edge_texture': 'png',
    'keypoints2d': 'png',
    'keypoints3d': 'png',
    'normal': 'png',
    'obj': 'obj',
    'ply': 'ply',
    'point_info': 'json',
    'principal_curvature': 'png',
    'reshading': 'png',
    'rgb': 'png',
    'sceneinfo': 'json',
    'segment_unsup25d': 'png',
    'segment_unsup2d': 'png',
    'semantic': 'png'
}


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
            elif task == 'reshading':
                x = x[[0]]
            task_dict[task] = x

        return task_dict
