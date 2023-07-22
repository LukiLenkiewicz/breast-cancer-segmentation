from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ScaleIntensityRanged
)


def train_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
            ScaleIntensityRanged(["image", "label"], a_min=0, a_max=255, b_min=0, b_max=1),
        ]
    )
    return transforms


def val_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
            ScaleIntensityRanged(["image", "label"], a_min=0, a_max=255, b_min=0, b_max=1),
        ]
    )
    return transforms
