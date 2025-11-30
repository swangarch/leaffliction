#!/usr/bin/python3

import os
import random
import re
from typing import List
import cv2
from plantcv import plantcv as pcv
import numpy as np
import sys


def list_files(folder: str) -> List[str]:
    """return a list of string containing all files' name in the fold"""
    return [f for f in os.listdir(folder)]


def extract_base_index(filename: str) -> int:
    """Extract numeric index from 'image(index).jpg'."""
    fname = filename.replace(" ", "").lower()
    m = re.search(r'image\((\d+)\)', fname)
    if not m:
        raise ValueError(f"Filename format wrong: {filename}")
    return int(m.group(1))


def aug_flip(img: np.ndarray) -> np.ndarray:
    """Horizontally flips the image"""
    return cv2.flip(img, 1)


def aug_rotate(img: np.ndarray, angle: float = None) -> np.ndarray:
    """Rotates the image by a random angle (default ±30°)"""
    h, w = img.shape[:2]
    if angle is None:
        angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)


def aug_shear(img: np.ndarray, max_shear=0.3) -> np.ndarray:
    """Applies a horizontal shear (slant) to the image"""
    h, w = img.shape[:2]
    sh = random.uniform(-max_shear, max_shear)
    M = np.array([[1, sh, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)


def aug_skew(img: np.ndarray, max_skew=0.3) -> np.ndarray:
    """Applies a vertical skew (top/bottom shift) to the image"""
    h, w = img.shape[:2]
    sk = random.uniform(-max_skew, max_skew)
    M = np.array([[1, 0, 0], [sk, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)


def aug_crop(img: np.ndarray, min_scale=0.7) -> np.ndarray:
    """
    Randomly crops a part of the image and resizes it back to original size
    """
    h, w = img.shape[:2]
    scale = random.uniform(min_scale, 0.95)
    new_w, new_h = int(w * scale), int(h * scale)
    x0 = random.randint(0, w - new_w) if w - new_w > 0 else 0
    y0 = random.randint(0, h - new_h) if h - new_h > 0 else 0
    cropped = img[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def aug_distort(img: np.ndarray, max_perturb=0.08) -> np.ndarray:
    """Applies a perspective distortion using 4 points perturbation"""
    h, w = img.shape[:2]
    margin_x = int(w * max_perturb)
    margin_y = int(h * max_perturb)

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([
        [random.randint(0, margin_x), random.randint(0, margin_y)],
        [w - 1 - random.randint(0, margin_x), random.randint(0, margin_y)],
        [w - 1 - random.randint(0, margin_x),
            h - 1 - random.randint(0, margin_y)],
        [random.randint(0, margin_x), h - 1 - random.randint(0, margin_y)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)


def apply_augmentation(method_name: str, img: np.ndarray,
                       aug_methods: list) -> np.ndarray:
    """apply augmentation function to image"""
    for name, fn in aug_methods:
        if name == method_name:
            return fn(img)
    raise ValueError(f"Unknown augmentation method: {method_name}")


def balance_dataset(root: str, aug_methods: list,
                    target_strategy: str = "max") -> None:
    """Function to balance dataset by using 6 methods to generate new images.
    Methods used: flip, rotate, skew, shear, crop, distort"""
    counts = {}

    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        if not os.path.isdir(sub_path):
            continue
        files = list_files(sub_path)
        counts[sub] = len(files)

    print("Counts:", counts)

    name_counter = {}

    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        if not os.path.isdir(sub_path):
            continue
        if target_strategy == "max":
            target = max(counts.values())
        else:
            target = int(np.ceil(sum(counts.values()) / len(counts)))

        print(f"Leave type: {sub} counts = {counts}, \
                augmentation target = {target}")

        os.makedirs(sub_path, exist_ok=True)
        files = list_files(sub_path)
        current_count = len(files)
        needed = target - current_count
        if needed <= 0:
            continue

        print(f"  Augmenting '{sub}' : need {needed} images.")

        method_cycle = [name for name, _ in aug_methods]
        num_methods = len(method_cycle)

        file_index = 0
        generated = 0

        while generated < needed:
            base_name = files[file_index]
            base_index_num = extract_base_index(base_name)
            base_img_path = os.path.join(sub_path, base_name)

            img = cv2.imdecode(np.fromfile(base_img_path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(base_img_path)

            if img is None:
                print(f"    Failed to read {base_img_path}, skipping.")
                file_index = (file_index + 1) % len(files)
                continue

            method_name = method_cycle[generated % num_methods]
            fn = dict(aug_methods)[method_name]
            aug_img = fn(img)
            h0, w0 = img.shape[:2]
            aug_img = cv2.resize(aug_img, (w0, h0))
            key = (sub_path, base_index_num, method_name)
            name_counter[key] = name_counter.get(key, 0) + 1
            n = name_counter[key]

            if n == 1:
                out_name = f"image({base_index_num})_{method_name}.jpg"
            else:
                out_name = f"image({base_index_num})_{method_name}({n}).jpg"

            out_path = os.path.join(sub_path, out_name)
            _, buf = cv2.imencode(".jpg", aug_img,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            buf.tofile(out_path)

            generated += 1
            file_index = (file_index + 1) % len(files)

            if generated % 10 == 0 or generated == needed:
                print(f"    Generated {generated}/{needed}: {out_name}")

    print("Augmentation complete.")


def process_single_file(file_path, aug_methods):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Cannot read image: {file_path}")

    folder = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1]

    print(f"Processing single file: {file_path}")

    for name, fn in aug_methods:
        aug_img = fn(img)
        out_name = f"{base}_{name}{ext}"
        out_path = os.path.join(folder, out_name)

        cv2.imwrite(out_path, aug_img)

        print(f"Displaying: {out_path}")
        pcv.plot_image(aug_img)

    print("All augmentations done!")


def main():
    if (len(sys.argv)) != 2:
        print("Usage: python3 Augmentation.py <path_to_file_to_augment>")
        return
    path = sys.argv[1]
    root_path = os.path.join(".", path)

    aug_methods = [
            ("flip", aug_flip),
            ("rotate", aug_rotate),
            ("skew", aug_skew),
            ("shear", aug_shear),
            ("crop", aug_crop),
            ("distort", aug_distort),
        ]

    if os.path.isfile(root_path):
        process_single_file(root_path, aug_methods)
    elif os.path.isdir(root_path):
        balance_dataset(root_path, aug_methods, target_strategy="max")
    else:
        print(f"Error: '{root_path}' not found. Run this script\
                from the project root where the 'image' folder lives.")


if __name__ == "__main__":
    main()
