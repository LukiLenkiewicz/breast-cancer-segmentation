from pathlib import Path

import numpy as np
import typer

from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    SaveImaged,
)
from pqdm.processes import pqdm

from ai.utils import get_data_paths


def preprocess(input_path: Path, output_path: Path, n_jobs: int = 1):
    paths = get_data_paths(input_path)
    transforms = Compose(
        [
            LoadImaged(["image", "label"], reader="PILReader"),
            EnsureChannelFirstd(["image", "label"]),
            CenterSpatialCropd(["image", "label"], roi_size=(128, 128)),
            # Add more transforms
            SaveImaged(
                keys=["image"],
                output_postfix="",
                output_dir=output_path / "images",
                writer="PILWriter",
                output_dtype=np.float32,
                separate_folder=False,
                resample=False,
                output_ext=".png",
                print_log=False,
            ),
            SaveImaged(
                keys=["label"],
                output_postfix="",
                output_dir=output_path / "labels",
                writer="PILWriter",
                output_dtype=np.float32,
                separate_folder=False,
                resample=False,
                output_ext=".png",
                print_log=False,
            ),
        ]
    )
    pqdm(paths, transforms, n_jobs=n_jobs, exception_behaviour="immediate")


if __name__ == "__main__":
    typer.run(preprocess)
