from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandRotated,
    RandFlipd,
    Rand2DElasticd,
    ScaleIntensityRanged,
)

import numpy as np


def train_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
            ScaleIntensityRanged(["image", "label"], a_min=0, a_max=255, b_min=0, b_max=1),
            RandRotated(["image", "label"], range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlipd(["image", "label"], spatial_axis=0, prob=0.5),
            Rand2DElasticd(["image", "label"], prob=1.0, spacing=(30, 30), magnitude_range=(5, 6), rotate_range=(np.pi / 4,), scale_range=(0.2, 0.2), translate_range=(100, 100), padding_mode="zeros",),
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
