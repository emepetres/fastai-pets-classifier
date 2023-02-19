from typing import List
from pets import config
from pets.unknown_imgs.dataloaders import get_dataloaders

from fastai.vision.all import (
    set_seed,
    Learner,
    vision_learner,
    resnet34,
    accuracy_multi,
    partial,
)


def get_learner() -> Learner:
    set_seed(config.SEED)
    dls = get_dataloaders()

    # The loss function is 0.5 and the metric is 0.95, why?
    # When we deploy the model we will make sure that it's set to 0.95,
    # but during training we don't want to bias the model towards extreme predictions.
    return vision_learner(dls, resnet34, metrics=partial(accuracy_multi, thresh=0.95))


def predict(checkpoint: str, path: str) -> List[str]:
    learn = get_learner()
    learn.load(checkpoint)
    learn.loss_func.thresh = 0.95

    return learn.predict(path)[0]
