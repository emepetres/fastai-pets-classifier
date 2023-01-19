# This import * is an extremely bad practice in modern Python programming.
# This shouldn’t be done in production
from fastai.vision.all import *

import config

if __name__ == "__main__":
    set_seed(config.SEED)

    path = untar_data(URLs.PETS)
    fnames = get_image_files(path / "images")
    pat = r"(.+)_\d+.jpg$"

    item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.0, 1.0))
    batch_tfms = [
        *aug_transforms(size=224, max_warp=0),
        Normalize.from_stats(*imagenet_stats),
    ]
    bs = 64

    # # dls = ImageDataLoaders.from_name_re(
    # #     path,  # The location of the data
    # #     fnames,  # A list of filenames
    # #     pat,  # A regex pattern to extract the labels
    # #     item_tfms=item_tfms,  # Transform augmentations to be applied per item
    # #     batch_tfms=batch_tfms,  # Transform augmentations to be applied per batch
    # #     bs=bs,  # How many examples should be drawn each time
    # # )

    pets = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(),
        get_y=RegexLabeller(pat=r"/([^/]+)_\d+.*"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    path_im = path / "images"
    dls = pets.dataloaders(path_im, bs=bs)

    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.load('stage_1')

    learn.fit_one_cycle(4, lr_max=slice(1e-6, 1e-4))

    learn.save('stage_2')
