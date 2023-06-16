import torch.nn as nn

import matplotlib.pyplot as plt

import torch
import wandb

from pytorch_lightning.loggers import WandbLogger

WANDB_LOGGER = None

def set_wandb_logger(run_name: str, project_name):
    global WANDB_LOGGER  # im so sorry. genuinely.

    WANDB_LOGGER = WandbLogger(name=run_name, project=project_name, log_model="all")


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

def load_wandb_checkpoint(run, artefact_path):
    raise NotImplemented()

def find_last_wandb_checkpoint(run) -> str:
    api = wandb.Api(overrides={"project": run.project, "entity": run.entity})
    artifact = run.use_artifact(api.artifact_versions("model", f"model-{run.id}")[0])
    artifact_dir = artifact.download()
    checkpoint_path = f"{artifact_dir}/model.ckpt"
    model_name = list(filter(lambda x: x[0] == "architecture_type", run.config.items()))[0][1]
    return checkpoint_path, model_name

def load_from_checkpoint(checkpoint_path, model_class):

    ckpt_params = torch.load(checkpoint_path, map_location="cuda:0")

    model = model_class(**ckpt_params["hyper_parameters"])
    model.load_state_dict(ckpt_params["state_dict"])

    return model


class DummyModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x