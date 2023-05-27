from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
)


def train_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
        ]
    )
    return transforms


def val_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
        ]
    )
    return transforms
