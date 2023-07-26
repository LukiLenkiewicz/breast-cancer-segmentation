from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


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


def export_model(model: nn.Module, onnx_filename: str="exported-model.onnx", input_shape: Iterable=(16, 1, 256, 256)):
    sample_input = torch.randn(*input_shape)
    model.eval()

    torch.onnx.export(model,
                    sample_input,
                    onnx_filename,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True, 
                    input_names=["input"],
                    output_names=["output"])
