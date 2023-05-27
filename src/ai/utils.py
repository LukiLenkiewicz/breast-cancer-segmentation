from pathlib import Path


def get_data_paths(input_path: Path) -> list[dict]:
    paths = []
    images = input_path / "images"
    labels = input_path / "labels"
    for image in sorted(images.iterdir()):
        label = labels / image.name
        d = dict(image=image, label=label)
        paths.append(d)
    return paths
