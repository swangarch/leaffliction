#!/usr/bin/python3

import matplotlib.pyplot as plt
import os
import sys
from srcs import *
from numpy import ndarray as array
import argparse


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


def display_transform(axes, objects:list, names:list, modes:list, \
                      dst: str, op=0) -> None:
    for i in range(0,3):
        for j in range(0,3):
            index = i * 3 + j
            if index < len(objects):
                axes[i][j].imshow(objects[index], cmap=modes[index])
                axes[i][j].set_title(names[index])
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
    if op == 0:
        plt.show()
    else:
        plt.savefig(dst)

    plt.close()


def transform(filename: str, op=False, dst_path="") -> None:
    img = load_img(filename)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(filename, fontsize=16)
    elements, names, modes = get_transform_elements(img)
    display_transform(axes, elements, names, modes, dst_path, op)
    if op == False:
        plot_histogram(img)


def main():
    if len(sys.argv) == 2:
        path = sys.argv[1]
        if os.path.isfile(path):
            transform(path)
        else:
            print("Error: argument is not a file.")
    else:
        parser = argparse.ArgumentParser(
            description="Program to transform target images from source path \
                         and save the result to destination path",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("-src", "--source", type=str, required=True,
                        help="Path to the input file or directory")
        parser.add_argument("-dst", "--destination", type=str, required=True,
                            help="Path to save the output")
        args = parser.parse_args()
        src = args.source
        dst = args.destination
        if not os.path.exists(src):
            print(f"Error: The source path '{src}' does not exist")
            sys.exit(1)

        if not os.path.exists(dst):
            os.makedirs(dst)

        if os.path.isfile(src):
            transform(src)
        elif os.path.isdir(src):
            for f in os.listdir(src):
                path = os.path.join(src, f)
                if not os.path.isfile(path):
                    print(f"Error: {path} in the source directory is not a file")
                    continue
                transform(path, 1, dst)
        print(f"Transformation completed! Results saved to {dst}")

if __name__ == "__main__":
    main()
