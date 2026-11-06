"""Promote clean masks/cutouts to canonical names and regenerate overlays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from clean_leaf_masks import source_images


LEAF_ALPHA = .28
DISEASE_ALPHA = .6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_overlay(rgb, leaf, disease):
    inside = leaf > 0
    selected = np.any(disease != 0, axis=2)
    if np.any(selected & ~inside):
        raise ValueError('Disease annotation extends outside the leaf')
    overlay = rgb.copy()
    overlay[inside] = np.rint(
        (1 - LEAF_ALPHA) * rgb[inside] + LEAF_ALPHA * 255).astype(np.uint8)
    overlay[selected] = np.rint(
        (1 - DISEASE_ALPHA) * rgb[selected]
        + DISEASE_ALPHA * disease[selected]).astype(np.uint8)
    return overlay


def paths_for(mask_path):
    stem = mask_path.name.removesuffix('_leaf_mask_clean.png')
    parent = mask_path.parent
    sources = source_images(parent).get(stem, [])
    if len(sources) != 1:
        raise ValueError(f'Expected one source image for {mask_path}')
    return {
        'source': sources[0],
        'clean_mask': mask_path,
        'clean_segmentation': parent / f'{stem}_leaf_segmentation_clean.png',
        'mask': parent / f'{stem}_leaf_mask.png',
        'segmentation': parent / f'{stem}_leaf_segmentation.png',
        'disease': parent / f'{stem}_disease_mask.png',
        'overlay': parent / f'{stem}_overlay_mask.png',
    }


def check_images(paths, clean=True):
    mask_key = 'clean_mask' if clean else 'mask'
    cutout_key = 'clean_segmentation' if clean else 'segmentation'
    with Image.open(paths['source']) as image:
        rgb = np.array(image.convert('RGB'))
    with Image.open(paths[mask_key]) as image:
        if image.mode != 'L':
            raise ValueError(f'Expected grayscale mask: {paths[mask_key]}')
        leaf = np.array(image)
    with Image.open(paths['disease']) as image:
        if image.mode != 'RGB':
            raise ValueError(f'Expected RGB disease mask: {paths["disease"]}')
        disease = np.array(image)
    with Image.open(paths[cutout_key]) as image:
        if image.mode != 'RGBA':
            raise ValueError(f'Expected RGBA cutout: {paths[cutout_key]}')
        cutout = np.array(image)
    if leaf.shape != rgb.shape[:2] or disease.shape != rgb.shape:
        raise ValueError(f'Mask/source sizes differ: {paths[mask_key]}')
    if not set(np.unique(leaf)).issubset({0, 255}):
        raise ValueError(f'Nonbinary leaf mask: {paths[mask_key]}')
    np.testing.assert_array_equal(cutout[..., 3], leaf)
    np.testing.assert_array_equal(cutout[leaf > 0, :3], rgb[leaf > 0])
    return render_overlay(rgb, leaf, disease)


def preflight(mask_path):
    paths = paths_for(mask_path)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    check_images(paths)
    return {
        'paths': {key: str(path) for key, path in paths.items()},
        'sha256': {key: digest(paths[key]) for key in
                   ('clean_mask', 'clean_segmentation', 'source', 'disease')},
    }


def backup_record(record, root, backup_root):
    paths = {key: Path(value) for key, value in record['paths'].items()}
    for key in ('mask', 'segmentation', 'overlay'):
        path = paths[key]
        backup = backup_root / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        if digest(path) != digest(backup):
            raise ValueError(
                f'Existing backup differs from original: {path}')


def promote(record):
    paths = {key: Path(value) for key, value in record['paths'].items()}
    overlay = check_images(paths)
    with tempfile.NamedTemporaryFile(
            dir=paths['overlay'].parent,
            suffix='.png', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(overlay).save(temporary)
        paths['clean_mask'].replace(paths['mask'])
        paths['clean_segmentation'].replace(paths['segmentation'])
        temporary.replace(paths['overlay'])
    finally:
        temporary.unlink(missing_ok=True)


def verify(record):
    paths = {key: Path(value) for key, value in record['paths'].items()}
    for current, original in [('mask', 'clean_mask'),
                              ('segmentation', 'clean_segmentation'),
                              ('source', 'source'), ('disease', 'disease')]:
        if digest(paths[current]) != record['sha256'][original]:
            raise ValueError(f'Content changed unexpectedly: {paths[current]}')
    if paths['clean_mask'].exists() or paths['clean_segmentation'].exists():
        raise ValueError('Redundant clean file remains')
    expected = check_images(paths, clean=False)
    with Image.open(paths['overlay']) as image:
        np.testing.assert_array_equal(np.array(image), expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path('leaves_segmentation'))
    parser.add_argument('--backup-dir', type=Path,
                        default=Path(
                            'output/annotations_before_clean_promotion'))
    args = parser.parse_args()
    masks = sorted(args.path.rglob('*_leaf_mask_clean.png'))
    if not masks:
        parser.error('No clean masks to promote')
    if args.backup_dir.resolve().is_relative_to(args.path.resolve()):
        parser.error('Backup must be outside the annotation tree')
    canonical = {
        p.with_name(p.name.replace('_leaf_mask.png', '_leaf_mask_clean.png'))
        for p in args.path.rglob('*_leaf_mask.png')}
    cutouts = {
        p.with_name(p.name.replace(
            '_leaf_segmentation_clean.png', '_leaf_mask_clean.png'))
        for p in args.path.rglob('*_leaf_segmentation_clean.png')}
    if set(masks) != canonical or set(masks) != cutouts:
        parser.error('Canonical/clean mask and clean cutout sets do not match')
    for directory in {p.parent for p in masks}:
        source_images(directory)
    with ThreadPoolExecutor(4) as pool:
        records = list(pool.map(preflight, masks))
    print(f'Validated {len(records)} image sets', flush=True)
    with (args.path / 'annotation_finalization_manifest.jsonl').open('w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda r: backup_record(r, args.path, args.backup_dir),
                      records))
    info_path = args.path / 'segmentation_info.json'
    if info_path.exists():
        backup_info = args.backup_dir / info_path.name
        if not backup_info.exists():
            shutil.copy2(info_path, backup_info)
    print('Original masks, cutouts and overlays backed up', flush=True)
    with ThreadPoolExecutor(4) as pool:
        for i, _ in enumerate(pool.map(promote, records), 1):
            if i % 1000 == 0:
                print(f'{i}/{len(records)} promoted', flush=True)
    print('Verifying final annotations and source hashes', flush=True)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(verify, records))
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.update({
        'version': 'adaptive-hsv-seeded-grabcut-v3-finalized',
        'leaf_values': [0, 255],
        'leaf_semantics': 'white leaf, black background; clean mask promoted',
        'disease_semantics': 'original class colors; zero outside clean leaf',
        'overlay_alpha': {'leaf': LEAF_ALPHA, 'disease': DISEASE_ALPHA},
        'canonical_leaf_mask': '<stem>_leaf_mask.png',
        'canonical_leaf_segmentation': '<stem>_leaf_segmentation.png',
    })
    info_path.write_text(json.dumps(info, indent=2) + '\n')
    summary = {'promoted': len(records), 'overlays_updated': len(records),
               'verified': len(records), 'sources_and_disease_masks_unchanged':
               True, 'clean_suffix_files_remaining': 0,
               'backup_directory': str(args.backup_dir)}
    (args.path / 'annotation_finalization_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
