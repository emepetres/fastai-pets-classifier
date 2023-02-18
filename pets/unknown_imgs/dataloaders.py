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
    MultiCategoryBlock,
    RandomSplitter,
    RegexLabeller,
    Pipeline,
)


def label_to_list(o):
    return [o]


def get_dataloaders() -> DataLoaders:
    path = untar_data(URLs.PETS)
    # # fnames = get_image_files(path / "images")
    # # pat = r"(.+)_\d+.jpg$"

    item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.0, 1.0))
    batch_tfms = [
        *aug_transforms(size=224, max_warp=0),
        Normalize.from_stats(*imagenet_stats),
    ]
    bs = 32

    pets = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(),
        get_y=Pipeline([RegexLabeller(pat=r"/([^/]+)_\d+.*"), label_to_list]),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    path_im = path / "images"
    return pets.dataloaders(path_im, bs=bs)
