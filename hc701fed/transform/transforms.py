"""
Author: Ibrahim Almakky
Date: 13/02/2022
"""
import yaml
from torchvision import transforms


def random_horizontal_flip(params: dict):
    return transforms.RandomHorizontalFlip(**params)


def random_rotation(params: dict):
    return transforms.RandomRotation(**params)


def gaussian_blur(params: dict):
    return transforms.GaussianBlur(**params)


def resize(params: dict):
    return transforms.Resize(**params)

TRANSFORMS = {
    "horizontal_flip": random_horizontal_flip,
    "random_rotation": random_rotation,
    "gaussian_blur": gaussian_blur,
    "resize": resize,
}


def compose(transforms_strs: dict):
    """
    Input images are assumed to be tensors
    """
    transforms_list = []
    for name, params in transforms_strs.items():
        assert name in TRANSFORMS.keys()
        transforms_list.append(TRANSFORMS[name](params))
    transforms_composed = transforms.Compose(transforms_list)
    return transforms_composed


if __name__ == "__main__":
    # Test case
    transforms_file = open("./params/ct_transforms.yaml")
    transforms_params = yaml.load(
        transforms_file,
        Loader=yaml.FullLoader,
    )
    test_transforms = compose(transforms_params["train"])