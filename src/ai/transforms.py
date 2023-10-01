from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandScaleIntensityd,
    ToTensord,
)


def train_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
            ScaleIntensityd(["image", "label"]),
            # RandAdjustContrastd(keys=["image"], prob=0.50), 
            # RandGaussianNoised(keys=["image"], prob=0.25),
            # RandScaleIntensityd(keys=["image"], prob=0.5),
            ToTensord(["image", "label"])
        ]
    )
    return transforms


def val_transforms() -> Compose:
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
            ScaleIntensityd(["image", "label"]),
            ToTensord(["image", "label"])
        ]
    )
    return transforms
