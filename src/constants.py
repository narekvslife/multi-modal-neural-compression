# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the timm and MAE-priv code base
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

MNIST = "mnist"
FASHION_MNIST = "fashion-mnist"
CLEVR = "clevr"

DATASET = CLEVR

SINGLE_TASK = "rgb"

WANDB_PROJECT_NAME = "vilab-compression"
WANDB_RUN_NAME = f"S-{DATASET}-{SINGLE_TASK}-lrplat"
