# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

task_parameters = {
    "depth_euclidean": {
        "in_channels": 1,
        "out_channels": 1,
        "clamp_to": (0.0, 8000.0 / (2**15 - 1)),  # Same as consistency
        #         'mask_val': 1.0,
        "loss_function": "mse",
    },
    "rgb": {"in_channels": 3, "out_channels": 3, "loss_function": "mse"},
    "semantic": {
        "in_channels": 1,
        "out_channels": 17,
        "loss_function": "cross-entropy",
    },
    "normal": {
        "in_channels": 3,
        "out_channels": 3,
        "mask_val": 0.502,
        "loss_function": "mse",
    },
    "mono": {"in_channels": 1, "out_channels": 1, "loss_function": "mse"},
}
