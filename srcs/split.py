#!/usr/bin/python3

import os
import sys
import shutil
import random as rd


def move_file(src: str, dst: str, file: str) -> None:
    """Move a file from a source directory to the other.
    file argument is the filename."""
    try:
        f_src = os.path.join(src, file)
        if not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)
        f_dst = os.path.join(dst, file)
        shutil.move(f_src, f_dst)
    except Exception as e:
        print("Error:", e)


def split_data(path: str, name: str, target: str, ratio: float) -> None:
    """Split dataset with a given ratio, and move to target folder.
    If ratio is negative, move all files in the target folder."""
    path_test = os.path.join(target, path+name)
    os.makedirs(path_test, exist_ok=True)
    for subdir in os.listdir(path):
        files_in_subdir = os.listdir(os.path.join(path, subdir))
        if ratio < 0:
            [move_file(os.path.join(path, subdir),
                       os.path.join(path_test, subdir), file)
                for file in files_in_subdir]
        elif ratio > 0 and ratio < 1:
            num_test = int(len(files_in_subdir) * ratio)
            sample = rd.sample(files_in_subdir, num_test)
            [move_file(os.path.join(path, subdir),
                       os.path.join(path_test, subdir), file)
                for file in sample]


def split_dataset(path: str, target: str, has_test: bool) -> None:
    """Split the dataset, saved into a target"""
    os.makedirs(target, exist_ok=True)
    if has_test is True:
        split_data(path, "_test", target, 0.15)
        split_data(path, "_val", target, 0.2)
        split_data(path, "_train", target, -1)
    else:
        split_data(path, "_val", target, 0.15)
        split_data(path, "_train", target, -1)
    print(f"[Dataset <{path}> splitted and saved into <{target}>]")


def main():
    try:
        if len(sys.argv) == 2:
            path = sys.argv[1]
            target = "dataset"
            os.makedirs(target, exist_ok=True)
            split_data(path, "_test", target, 0.15)
            split_data(path, "_val", target, 0.2)
            split_data(path, "_train", target, -1)
        else:
            raise TypeError("""Wrong argument number,
                                Usage: python split.py <path>""")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
