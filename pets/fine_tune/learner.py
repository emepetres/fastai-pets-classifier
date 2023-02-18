from pets import config
from pets.fine_tune.dataloaders import get_dataloaders

from fastai.vision.all import (
    set_seed,
    Learner,
    vision_learner,
    resnet34,
    error_rate,
)


def get_resnet_learner() -> Learner:
    set_seed(config.SEED)
    dls = get_dataloaders()
    return vision_learner(dls, resnet34, metrics=error_rate)


def get_vit_learner() -> Learner:
    set_seed(config.SEED)
    dls = get_dataloaders()
    return vision_learner(dls, "vit_tiny_patch16_224")
