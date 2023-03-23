import glob

import torch.utils.data

from PIL import Image


def split_img_mask(path):
    make_list = sorted(glob.glob(path))
    mask = []
    img = []
    label = []
    for a in make_list:
        if a.endswith("mask.png"):
            mask.append(a)
            label.append(a.split()[0])
        elif a.endswith(").png"):
            img.append(a)

    data = {"img": img, "mask": mask}
    return data, label


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        image = Image.open(img_path, mode="r")

        mask_path = self.mask_path[index]
        mask = Image.open(mask_path, mode="r")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
