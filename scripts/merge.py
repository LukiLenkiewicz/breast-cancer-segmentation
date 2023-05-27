from functools import reduce
from pathlib import Path

import typer

from PIL import (
    Image,
    ImageChops,
)


def group_files(input_path: Path) -> list[dict]:
    results = []

    for folder in input_path.iterdir():
        files = {}
        for file in sorted(folder.iterdir()):
            index = file.name.split(")")[0].split("(")[1]
            if files.get(index) is None:
                files[index] = {
                    "image": None,
                    "labels": [],
                }
            if "mask" not in file.name:
                files[index]["image"] = file
            else:
                files[index]["labels"].append(file)
        results.append(files)

    return results


def merge(input_path: Path, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = output_path / "images"
    label_path = output_path / "labels"
    image_path.mkdir(exist_ok=True)
    label_path.mkdir(exist_ok=True)

    grouped_files_per_folder = group_files(input_path)

    index = 0
    for grouped_files in grouped_files_per_folder:
        for files in grouped_files.values():
            index += 1

            image = Image.open(files["image"]).convert("L")
            image.save(image_path / f"{index:04}.png")
            labels = [Image.open(label).convert("L") for label in files["labels"]]

            if len(labels) == 1:
                label = labels[0]
            else:
                label = reduce(lambda x, y: ImageChops.lighter(x, y), labels)

            label.save(label_path / f"{index:04d}.png")


if __name__ == "__main__":
    typer.run(merge)
