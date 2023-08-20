from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandRotated,
    RandFlipd,
    Rand2DElasticd,
    ScaleIntensityd,
    RandAdjustContrastd,
)

import numpy as np


def train_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
            ScaleIntensityd(["image", "label"]),
            # RandFlipd(["image", "label"], prob=0.5),
            # RandAdjustContrastd(["image"], prob=0.5, gamma=2)
        ]
    )
    return transforms


def val_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
            ScaleIntensityd(["image", "label"]),
        ]
    )
    return transforms
