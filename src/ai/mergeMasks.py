import glob
import os

from PIL import (
    Image,
    ImageChops,
)


def merge_masks(path1, path2):
    image1 = Image.open(path1).convert("RGBA")
    image2 = Image.open(path2).convert("RGBA")
    blended = ImageChops.lighter(image1, image2)
    blended.save(path1)


def merge_all_masks(path):
    global im1
    make_list = sorted(glob.glob(path, recursive=True))
    print(make_list)
    for a in make_list:
        if a.endswith("mask.png"):
            im1 = a
        elif a.endswith("1.png"):
            im2 = a
            merge_masks(im1, im2)
            os.remove(a)
        elif a.endswith("2.png"):
            im2 = a
            merge_masks(im1, im2)
            os.remove(a)


directoryPath = ""  # path should end with /Dataset_BUSI_with_GT/**/*.png
merge_all_masks(directoryPath)
