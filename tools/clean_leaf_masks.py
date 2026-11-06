"""Export existing leaf cutouts and suppress exterior background shadows.

Run: .venv/bin/python tools/clean_leaf_masks.py
Only derived files are written; original images and annotations are preserved.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def clean_leaf_mask(rgb, mask):
    """Refine a mask with adaptive pigment seeds and image-edge graph cuts.

    HSV/chromaticity select reliable leaf seeds; they are not hard thresholds
    on the final silhouette. GrabCut learns foreground/background mixtures
    and penalizes cuts between similar neighbors. Shadow candidates use soft
    labels so gray highlights and dark leaf tissue can be recovered. Enclosed
    holes are restored to preserve lesions. The result is a subset of the mask.
    """
    leaf = np.asarray(mask) > 0
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape != (*leaf.shape, 3):
        raise ValueError('Image and mask dimensions must agree')
    if not leaf.any() or leaf.all():
        return leaf.astype(np.uint8) * 255
    smooth = cv2.GaussianBlur(rgb, (3, 3), 0)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    color = smooth.astype(np.float32)
    chroma = color / np.maximum(color.sum(axis=2, keepdims=True), 1)
    outside = cv2.dilate(leaf.astype(np.uint8),
                         np.ones((5, 5), np.uint8)) == 0
    if outside.sum() < 20:
        return leaf.astype(np.uint8) * 255
    reference = np.median(chroma[outside], axis=0)
    distance = np.linalg.norm(chroma - reference, axis=2)
    hue, saturation, value = (hsv[..., i] for i in range(3))
    threshold = max(35., float(np.percentile(saturation[leaf], 65)))
    core = cv2.erode(leaf.astype(np.uint8),
                     np.ones((3, 3), np.uint8)) > 0
    foreground = (core & (saturation >= threshold) & (value > 35)
                  & (distance > .035))
    if foreground.sum() < 20:
        # There is no trustworthy pigment cue (e.g. a fully gray leaf).
        # Leave the existing annotation intact rather than invent a contour.
        return leaf.astype(np.uint8) * 255
    background_value = float(np.median(value[outside]))
    shadow_saturation = max(
        25., float(np.percentile(saturation[foreground], 10)) * .65)
    candidates = ((distance < .06) & (saturation < shadow_saturation)
                  & (value < background_value * .98))
    candidates |= (value < 50) & (saturation < 100)
    _, labels = cv2.connectedComponents(
        (~leaf | candidates).astype(np.uint8), connectivity=8)
    exterior = np.unique(np.concatenate((labels[~leaf], labels[0], labels[-1],
                                         labels[:, 0], labels[:, -1])))
    exterior = exterior[exterior != 0]
    shadow = leaf & candidates & np.isin(labels, exterior)
    trimap = np.where(leaf, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    trimap[shadow] = cv2.GC_PR_BGD
    # Thin green stems may not survive erosion; seed them by green excess.
    # Hue alone is insufficient: olive-colored shadows also have green hue.
    stems = (leaf & (hue >= 25) & (hue <= 100) & (saturation > 35)
             & (value > 45) & (color[..., 1] > color[..., 0] * 1.03)
             & (color[..., 1] > color[..., 2] * 1.1))
    # Waxy highlights can turn green leaves blue-gray with low saturation.
    # Retain these only when their green excess and chromaticity differ from
    # the observed background; olive and purple cast shadows fail this cue.
    highlights = (core & (hue >= 35) & (hue <= 115) & (saturation > 12)
                  & (value > 45) & (distance > .025)
                  & (color[..., 1] > color[..., 0] * 1.025))
    foreground |= stems | highlights
    trimap[foreground] = cv2.GC_FGD
    # OpenCV's RNG is thread-local; keep batch and single-image results stable.
    cv2.setRNGSeed(42)
    cv2.grabCut(smooth, trimap, None, np.zeros((1, 65)), np.zeros((1, 65)),
                4, cv2.GC_INIT_WITH_MASK)
    result = leaf & ((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD))
    # Keep enclosed lesions instead of punching holes through dark tissue.
    flooded = np.pad(result.astype(np.uint8), 1)
    cv2.floodFill(flooded, None, (0, 0), 1)
    result |= (flooded[1:-1, 1:-1] == 0) & leaf
    return result.astype(np.uint8) * 255


def save_cutout(rgb, mask, path):
    rgba = np.dstack((rgb.copy(), mask))
    rgba[mask == 0, :3] = 0
    Image.fromarray(rgba).save(path, compress_level=1)


@lru_cache(maxsize=None)
def source_images(directory):
    images = {}
    for path in directory.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.setdefault(path.stem, []).append(path)
    return images


def process(mask_path):
    stem = mask_path.name.removesuffix('_leaf_mask.png')
    sources = source_images(mask_path.parent).get(stem, [])
    if len(sources) != 1:
        raise ValueError(
            f'Expected one source image for {mask_path}, got {sources}')
    with Image.open(sources[0]) as image:
        rgb = np.array(image.convert('RGB'))
    with Image.open(mask_path) as image:
        mask = np.array(image.convert('L'))
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f'Mask size differs from source: {mask_path}')
    mask = (mask > 0).astype(np.uint8) * 255
    clean = clean_leaf_mask(rgb, mask)
    prefix = mask_path.parent / stem
    save_cutout(rgb, mask, f'{prefix}_leaf_segmentation.png')
    Image.fromarray(clean).save(f'{prefix}_leaf_mask_clean.png')
    save_cutout(rgb, clean, f'{prefix}_leaf_segmentation_clean.png')
    before, after = int(np.count_nonzero(mask)), int(np.count_nonzero(clean))
    fraction = (before - after) / max(before, 1)
    return {'image': str(sources[0]), 'original_pixels': before,
            'clean_pixels': after, 'removed_pixels': before - after,
            'removed_fraction': fraction,
            'review_required': fraction > .15 or before == 0 or after == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path('leaves_segmentation'))
    parser.add_argument('--limit', type=int, default=None,
                        help='Process a sample first')
    args = parser.parse_args()
    masks = (sorted(args.path.rglob('*_leaf_mask.png'))
             if args.path.is_dir() else [args.path])
    if args.limit is not None:
        masks = masks[:args.limit]
    if not masks:
        parser.error('No leaf masks found')
    report = args.path if args.path.is_dir() else args.path.parent
    records, errors = [], []
    cv2.setNumThreads(1)
    for directory in {p.parent for p in masks}:
        source_images(directory)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process, path) for path in masks]
        for i, (path, future) in enumerate(zip(masks, futures), 1):
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({'mask': str(path), 'error': str(error)})
            if i % 250 == 0:
                print(f'{i}/{len(masks)}', flush=True)
    with (report / 'leaf_cleanup_manifest.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    review = sorted((r for r in records if r['review_required']),
                    key=lambda r: r['removed_fraction'], reverse=True)
    (report / 'leaf_cleanup_review.json').write_text(
        json.dumps(review, indent=2) + '\n')
    summary = {
        'processed': len(records),
        'changed': sum(r['removed_pixels'] > 0 for r in records),
        'review_required': sum(r['review_required'] for r in records),
        'errors': errors,
        'algorithm': 'adaptive-hsv-seeded-grabcut-v3',
        'status': 'automatic_refinement_needs_visual_review',
    }
    (report / 'leaf_cleanup_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
