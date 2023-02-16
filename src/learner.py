import config
from dataloaders import get_dataloaders

from fastai.vision import (
    set_seed,
    Learner,
    vision_learner,
    resnet34,
    error_rate,
)


def get_learner() -> Learner:
    set_seed(config.SEED)
    dls = get_dataloaders()
    return vision_learner(dls, resnet34, metrics=error_rate)
