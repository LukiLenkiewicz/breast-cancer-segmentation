from pathlib import Path
import torch


def get_data_paths(input_path: Path) -> list[dict]:
    paths = []
    images = input_path / "images"
    labels = input_path / "labels"
    for image in sorted(images.iterdir()):
        label = labels / image.name
        d = dict(image=image, label=label)
        paths.append(d)
    return paths


def get_accuracy(y: torch.Tensor , y_pred: torch.Tensor) -> float:
    return float(torch.sum(y == y_pred)/torch.numel(y))
