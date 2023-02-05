from torch import nn
from torchvision.transforms import ToTensor

import re
from PIL import Image
from torch.utils.data import Dataset


class PetsDataset(Dataset):
    "A basic dataset that will return a tuple of (image, label)"

    def __init__(self, filenames: list, transforms: nn.Sequential, label_to_int: dict):
        self.filenames = filenames
        self.transforms = transforms
        self.label_to_int = label_to_int
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.filenames)

    def apply_x_transforms(self, filename):
        image = Image.open(filename).convert("RGB")
        tensor_image = self.to_tensor(image)
        return self.transforms(tensor_image)

    def apply_y_transforms(self, filename):
        label = re.findall(r"^(.*)_\d+\.jpg$", filename.name)[0].lower()
        return self.label_to_int[label]

    def __getitem__(self, index):
        filename = self.filenames[index]
        x = self.apply_x_transforms(filename)
        y = self.apply_y_transforms(filename)
        return (x, y)
