"""Clip disease annotations to clean leaf masks, with original backups."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image


def clip_disease_mask(disease, leaf):
    """Zero pixels outside the leaf; preserve class values inside."""
    disease = np.asarray(disease)
    leaf = np.asarray(leaf)
    if leaf.ndim != 2 or disease.shape[:2] != leaf.shape:
        raise ValueError('Disease and leaf mask dimensions must agree')
    result = disease.copy()
    result[leaf == 0] = 0
    return result


def leaf_path(disease_path):
    stem = disease_path.name.removesuffix('_disease_mask.png')
    clean = disease_path.with_name(f'{stem}_leaf_mask_clean.png')
    if clean.exists():
        return clean
    return disease_path.with_name(f'{stem}_leaf_mask.png')


def validate_pair(path):
    with Image.open(path) as disease, Image.open(leaf_path(path)) as leaf:
        if disease.mode not in {'RGB', 'L'}:
            raise ValueError(f'Unsupported disease mask mode: {path}')
        if disease.size != leaf.size or leaf.mode != 'L':
            raise ValueError(f'Incompatible clean leaf mask: {path}')
        if not set(np.unique(np.array(leaf))).issubset({0, 255}):
            raise ValueError(f'Clean leaf mask must be binary: {path}')


def process(path, root, backup_root):
    backup = backup_root / path.relative_to(root)
    backup.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the first original backup on reruns.
    if not backup.exists():
        shutil.copy2(path, backup)
    with Image.open(path) as image:
        disease = np.array(image)
    with Image.open(leaf_path(path)) as image:
        leaf = np.array(image)
    clean = clip_disease_mask(disease, leaf)
    nonzero = (np.any(disease != 0, axis=2)
               if disease.ndim == 3 else disease != 0)
    removed = int(np.count_nonzero(nonzero & (leaf == 0)))
    if removed:
        # Replace only after a complete PNG has been written successfully.
        with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix='.png', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            Image.fromarray(clean).save(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return {'disease_mask': str(path), 'leaf_mask': str(leaf_path(path)),
            'backup': str(backup), 'original_pixels': int(nonzero.sum()),
            'removed_pixels': removed,
            'remaining_pixels': int(nonzero.sum()) - removed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path('leaves_segmentation'))
    parser.add_argument('--backup-dir', type=Path,
                        default=Path('output/disease_masks_before_leaf_clean'))
    args = parser.parse_args()
    paths = sorted(args.path.rglob('*_disease_mask.png'))
    if not paths:
        parser.error('No disease masks found')
    if args.backup_dir.resolve().is_relative_to(args.path.resolve()):
        parser.error('Backup directory must be outside the annotation tree')
    # Reject missing or incompatible inputs before modifying annotations.
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(validate_pair, paths))
    print(f'Validated {len(paths)} pairs; updating disease masks', flush=True)
    records, errors = [], []
    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(process, p, args.path, args.backup_dir)
                   for p in paths]
        for i, (path, future) in enumerate(zip(paths, futures), 1):
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({'mask': str(path), 'error': str(error)})
            if i % 1000 == 0:
                print(f'{i}/{len(paths)}', flush=True)
    with (args.path / 'disease_cleanup_manifest.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    summary = {
        'processed': len(records),
        'changed': sum(r['removed_pixels'] > 0 for r in records),
        'removed_pixels': sum(r['removed_pixels'] for r in records),
        'backup_directory': str(args.backup_dir),
        'rule': 'disease_mask[leaf_mask_clean == 0] = 0',
        'errors': errors,
    }
    (args.path / 'disease_cleanup_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
