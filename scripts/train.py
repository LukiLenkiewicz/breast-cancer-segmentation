from pathlib import Path

import pytorch_lightning as pl
import torch
import typer

from monai.data.dataset import Dataset
from torch.utils.data import (
    DataLoader,
    random_split,
)

from ai.model import UNet
from ai.segmentation_module import SegmentationModule
from ai.transforms import (
    train_transforms,
    val_transforms,
)
from ai.utils import get_data_paths


def get_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 16
) -> tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def get_datasets(train_paths: list[str], val_paths: list[str]) -> tuple[Dataset, Dataset]:
    train_dataset = Dataset(train_paths, transform=train_transforms())
    val_dataset = Dataset(val_paths, transform=val_transforms())
    return train_dataset, val_dataset


def train(input_path: Path):
    data_paths = get_data_paths(input_path)
    train_paths, val_paths = random_split(data_paths, [0.8, 0.2], generator=torch.Generator().manual_seed(123))
    train_ds, val_ds = get_datasets(train_paths, val_paths)
    train_dl, val_dl = get_dataloaders(train_ds, val_ds)

    model = UNet(input_channels=1)
    segmentation_module = SegmentationModule(model)

    trainer = pl.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(segmentation_module, train_dl, val_dl)


if __name__ == "__main__":
    typer.run(train)
