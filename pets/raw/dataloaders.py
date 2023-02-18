import re
from typing import Tuple
import numpy as np

from torch import nn
from torchvision.transforms import CenterCrop, RandomResizedCrop, Normalize
from torch.utils.data import DataLoader

from fastai.data.external import untar_data, URLs
from fastai.vision.data import imagenet_stats
from fastai.data.core import DataLoaders
from fastcore.xtras import (  # noqa: F401
    Path,
)  # to bring in some patched functionalities we will use later

from pets.raw.dataset import PetsDataset


def get_items_transforms():
    train_transforms = nn.Sequential(
        RandomResizedCrop((224, 224)), Normalize(*imagenet_stats)
    )

    # On validation we don't apply augmentations, only transformations
    # That's why RandomResizedCrop -> CenterCrop
    valid_transforms = nn.Sequential(CenterCrop((224, 224)), Normalize(*imagenet_stats))

    return train_transforms, valid_transforms


def get_datasets() -> Tuple[PetsDataset, PetsDataset]:
    # dataset
    dataset_path = untar_data(URLs.PETS)

    # transforms
    train_transforms = nn.Sequential(
        RandomResizedCrop((224, 224)), Normalize(*imagenet_stats)
    )
    # On validation we don't apply augmentations, only transformations
    # That's why RandomResizedCrop -> CenterCrop
    valid_transforms = nn.Sequential(CenterCrop((224, 224)), Normalize(*imagenet_stats))

    # labels
    label_pat = r"^(.*)_\d+\.jpg$"
    filenames = (dataset_path / "images").ls(file_exts=".jpg")
    labels = filenames.map(lambda x: re.findall(label_pat, x.name)[0].lower()).unique()
    label_to_int = {index: key for key, index in enumerate(labels)}

    # split
    shuffled_indexes = np.random.permutation(len(filenames))
    split = int(0.8 * len(filenames))
    train_indexes, valid_indexes = (shuffled_indexes[:split], shuffled_indexes[split:])
    train_fnames = filenames[train_indexes]
    valid_fnames = filenames[valid_indexes]

    train_transforms, valid_transforms = get_items_transforms()

    train_dataset = PetsDataset(train_fnames, train_transforms, label_to_int)
    valid_dataset = PetsDataset(valid_fnames, valid_transforms, label_to_int)

    return train_dataset, valid_dataset


def get_dataloaders():
    dtrain, dvalid = get_datasets()

    train_dataloader = DataLoader(dtrain, shuffle=True, drop_last=True, batch_size=64)

    # we can increase the validation batch size here,
    # as on validation we don't use grads
    valid_dataloader = DataLoader(
        dvalid,
        batch_size=128,
    )

    return DataLoaders(train_dataloader, valid_dataloader)
