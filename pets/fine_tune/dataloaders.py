from fastai.vision.all import (
    DataLoaders,
    untar_data,
    URLs,
    get_image_files,
    RandomResizedCrop,
    aug_transforms,
    Normalize,
    imagenet_stats,
    DataBlock,
    ImageBlock,
    CategoryBlock,
    RandomSplitter,
    RegexLabeller,
)


def get_dataloaders() -> DataLoaders:
    path = untar_data(URLs.PETS)
    # # fnames = get_image_files(path / "images")
    # # pat = r"(.+)_\d+.jpg$"

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
    return pets.dataloaders(path_im, bs=bs)
