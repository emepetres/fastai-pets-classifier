from functools import partial
from PIL import Image

import torch
from torch import nn
from torchvision.models import resnet34
from torch.optim import AdamW
from torchvision.transforms import ToTensor

from fastcore.xtras import (  # noqa: F401
    Path,
)  # to bring in some patched functionalities we will use later

from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import accuracy
from fastai.learner import Learner
from fastai.callback.schedule import (  # noqa: F811
    Learner,
)  # To get `fit_one_cycle`, `lr_find`, and more
from fastai.optimizer import OptimWrapper

from pets.pytorch.dataloaders import get_dataloaders


def get_learner(bs=64):
    dls = get_dataloaders(bs)

    num_classes = len(dls.valid.dataset.label_to_int)

    model = resnet34(pretrained=True)
    model.fc = nn.Linear(512, num_classes, bias=True)

    opt_func = partial(OptimWrapper, opt=AdamW)

    return Learner(
        dls,
        model.cuda(),
        opt_func=opt_func,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
    )


def predict(checkpoint: str, im: Image):
    learn = get_learner()
    learn.load(checkpoint)

    label_to_int = learn.dls.valid.dataset.label_to_int
    valid_transforms = learn.dls.valid.dataset.transforms

    net = learn.model
    net.eval()

    tfm_x = valid_transforms(ToTensor()(im))
    tfm_x = tfm_x.unsqueeze(0)  # it expects a batch

    with torch.no_grad():
        preds = net(tfm_x.cuda())
    pred = preds.argmax(dim=-1)[0]
    label = list(label_to_int.keys())[pred]

    return pred, label
