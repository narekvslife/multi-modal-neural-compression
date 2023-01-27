# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

task_parameters = {
    'depth_euclidean': {
        'in_channels': 1,
        'out_channels': 1,
        'clamp_to': (0.0, 8000.0 / (2 ** 15 - 1)),  # Same as consistency
        #         'mask_val': 1.0,
        'loss_function': 'mse'
    },
    'rgb': {
        'in_channels': 3,
        'out_channels': 3,
        'loss_function': 'mse'
    },
    'semantic': {
        'in_channels': 1,
        'out_channels': 17,
        'loss_function': 'cross-entropy'
    },
}

PIX_TO_PIX_TASKS = ['colorization', 'edge_texture', 'edge_occlusion', 'keypoints3d', 'keypoints2d', 'reshading',
                    'depth_zbuffer', 'depth_euclidean', 'curvature', 'autoencoding', 'denoising', 'normal',
                    'inpainting', 'segment_unsup2d', 'segment_unsup25d', 'semantic', ]
FEED_FORWARD_TASKS = ['class_object', 'class_scene', 'room_layout', 'vanishing_point']
SINGLE_IMAGE_TASKS = PIX_TO_PIX_TASKS + FEED_FORWARD_TASKS
SIAMESE_TASKS = ['fix_pose', 'jigsaw', 'ego_motion', 'point_match', 'non_fixated_pose']
