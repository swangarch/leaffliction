#!/usr/bin/python3

import matplotlib.pyplot as plt
import os
import sys
from srcs import *
from numpy import ndarray as array


def get_transform_elements(img:array) -> tuple[list[array], list[str], list[str]]:
    objects = []
    names = []
    modes = []
    leaf_mask = get_leaf_mask(img)

    objects.append(img)
    names.append("Original")
    modes.append("viridis")

    objects.append(blur_img(img))
    names.append("Gaussian Blur")
    modes.append("viridis")

    objects.append(img_detect_leaf(img))
    names.append("Segmentation")
    modes.append("viridis")

    objects.append(hue_mask(img))
    names.append("Hue mask")
    modes.append("gray")

    objects.append(saturation_mask(blur_img(img)))
    names.append("Saturation mask")
    modes.append("gray")

    objects.append(binary_mask(img))
    names.append("Binary mask")
    modes.append("gray")

    objects.append(analyze_img(img, leaf_mask))
    names.append("Analyze object")
    modes.append("viridis")

    objects.append(contour_img(img, leaf_mask))
    names.append("Roi object")
    modes.append("viridis")

    objects.append(extract_pseudolandmarks(img, leaf_mask))
    names.append("Pseudolandmarks")
    modes.append("viridis")

    return objects, names, modes


def display_transform(axes, objects:list, names:list, modes:list) -> None:
    for i in range(0,3):
        for j in range(0,3):
            index = i * 3 + j
            if index < len(objects):
                axes[i][j].imshow(objects[index], cmap=modes[index])
                axes[i][j].set_title(names[index])
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
    plt.show()
    plt.close()


def transform(img: array, filename: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(filename, fontsize=16)
    elements, names, modes = get_transform_elements(img)
    display_transform(axes, elements, names, modes)
    plot_histogram(img)


def main():
    # try:
        if len(sys.argv) != 2:
            raise TypeError("Wrong number of arguments")
        path = sys.argv[1]
        if os.path.isfile(path):
            img = load_img(path)
            transform(img, path)
        else:
            print("Error: argument is not a file.")
    # except Exception as e:
    # 	print("Error:", e)


if __name__ == "__main__":
    main()
