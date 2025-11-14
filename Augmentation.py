#!/usr/bin/python3

import os
import random
import re
from typing import List
import cv2
import numpy as np


def list_files(folder: str) -> List[str]:
    """return a list of string containing all files' name in the fold"""
    return [f for f in os.listdir(folder)]


def next_image_index(folder: str, pattern=r"image\((\d+)\)\.jpg") -> int:
    """Find the next number to use for naming augmented images"""
    files = list_files(folder)
    max_idx = 0
    for f in files:
        m = re.match(pattern, f)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
    return max_idx + 1


def build_index_map(root_dir: str) -> dict[str, int]:
    """Build a map to find the max index of each subdirectory"""
    index_map = {}
    for sub in os.listdir(root_dir):
        path = os.path.join(root_dir, sub)
        if not os.path.isdir(path):
            continue
        index_map[path] = next_image_index(path)
    return index_map


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
                    target_strategy: str = "max", seed: int = 42) -> None:
    """Function to balance dataset by using 6 methods to generate new images.
    Methods used: flip, rotate, skew, shear, crop, distort"""
    random.seed(seed)
    np.random.seed(seed)
    imap = build_index_map(root)
    counts = {}
    
    for sub in os.listdir(root):
        path = os.path.join(root, sub)
        files = list_files(path)
        counts[sub] = len(files)

    print(counts)
    for sub in os.listdir(root):
        if target_strategy == "max":
            target = max(counts.values())
        else:
            target = int(np.ceil(sum(counts.values()) / len(counts)))

        print(f"Leave type: {sub} counts = {counts}, augmentation target = {target}")

        path = os.path.join(root, sub)
        os.makedirs(path, exist_ok=True)
        current_files = list_files(path)
        current_count = len(current_files)
        need = target - current_count
        if need <= 0:
            continue

        print(f"  Augmenting '{sub}' : need {need} images.")
        next_idx = imap[path]
        imap[path] += 1
        method_cycle = [name for name, _ in aug_methods]

        base_pool = current_files.copy()
        if not base_pool:
            print(f"    WARNING: no base images in {path}, skipping.")
            continue

        for i in range(need):
            base_name = random.choice(base_pool)
            base_path = os.path.join(path, base_name)
            img = cv2.imdecode(np.fromfile(base_path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(base_path)
            if img is None:
                print(f"    Failed to read {base_path}, skipping.")
                continue

            method_name = method_cycle[i % len(method_cycle)]

            aug_img = apply_augmentation(method_name, img, aug_methods)
            h0, w0 = img.shape[:2]
            aug_img = cv2.resize(aug_img, (w0, h0),
                                 interpolation=cv2.INTER_LINEAR)

            out_name = f"image({next_idx}).jpg"
            out_path = os.path.join(path, out_name)
            _, encimg = cv2.imencode(".jpg", aug_img,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            encimg.tofile(out_path)

            if (i + 1) % 10 == 0 or i == need - 1:
                print(f"    Generated {i + 1}/{need} =>\
                        {out_name} ({method_name})")
            next_idx += 1

    print("Augmentation complete.")


def main():
    root_dir = os.path.join(".", "leaves")
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' not found. Run this script\
                from the project root where the 'image' folder lives.")
        return

    aug_methods = [
            ("flip", aug_flip),
            ("rotate", aug_rotate),
            ("skew", aug_skew),
            ("shear", aug_shear),
            ("crop", aug_crop),
            ("distort", aug_distort),
        ]
    balance_dataset(root_dir, aug_methods, target_strategy="max", seed=123)


if __name__ == "__main__":
    main()
