import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger

WANDB_LOGGER = None


def set_wandb_logger(run_name: str, project_name):
    global WANDB_LOGGER  # im so sorry. genuinely.

    WANDB_LOGGER = WandbLogger(name=run_name,
                               project=project_name,
                               log_model="all")


def get_wandb_logger():
    if WANDB_LOGGER:
        return WANDB_LOGGER
    else:
        raise ValueError("WANDB_LOGGER NOT SET")


def show_images(images: list):
    fig, axs = plt.subplots(1, len(images))

    for i in range(len(images)):
        axs[i].imshow(images[i])

    plt.show()
