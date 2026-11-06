"""Verify overlay priority, canonical promotion and original backups."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from finalize_annotations import (backup_record, preflight, promote,
                                  render_overlay, verify)
from clean_disease_masks import leaf_path


class FinalizeAnnotationTests(unittest.TestCase):
    def test_overlay_priority_and_background(self):
        rgb = np.full((2, 2, 3), 50, np.uint8)
        leaf = np.array([[255, 255], [0, 0]], np.uint8)
        disease = np.zeros_like(rgb)
        disease[0, 1] = [230, 45, 60]
        overlay = render_overlay(rgb, leaf, disease)
        np.testing.assert_array_equal(overlay[0, 0], [107, 107, 107])
        np.testing.assert_array_equal(overlay[0, 1], [158, 47, 56])
        np.testing.assert_array_equal(overlay[1], rgb[1])
        disease[1, 1] = [255, 0, 0]
        with self.assertRaises(ValueError):
            render_overlay(rgb, leaf, disease)

    def test_promotion_preserves_clean_bytes_and_backs_up_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'annotations'
            root.mkdir()
            backup = Path(directory) / 'backup'
            rgb = np.full((6, 6, 3), 50, np.uint8)
            leaf = np.zeros((6, 6), np.uint8)
            leaf[1:5, 1:5] = 255
            rgba = np.dstack((rgb.copy(), leaf))
            rgba[leaf == 0, :3] = 0
            images = {
                'example.png': rgb,
                'example_leaf_mask.png': np.full((6, 6), 255, np.uint8),
                'example_leaf_segmentation.png': np.full((6, 6, 4), 255,
                                                         np.uint8),
                'example_overlay_mask.png': rgb,
                'example_leaf_mask_clean.png': leaf,
                'example_leaf_segmentation_clean.png': rgba,
                'example_disease_mask.png': np.zeros_like(rgb),
            }
            for name, data in images.items():
                Image.fromarray(data).save(root / name)
            original = {name: (root / name).read_bytes() for name in images}
            mask = root / 'example_leaf_mask_clean.png'
            record = preflight(mask)
            backup_record(record, root, backup)
            promote(record)
            verify(record)
            for suffix in ('leaf_mask', 'leaf_segmentation'):
                self.assertEqual(
                    (root / f'example_{suffix}.png').read_bytes(),
                    original[f'example_{suffix}_clean.png'])
                self.assertEqual(
                    (backup / f'example_{suffix}.png').read_bytes(),
                    original[f'example_{suffix}.png'])
            disease = root / 'example_disease_mask.png'
            self.assertEqual(
                leaf_path(disease), root / 'example_leaf_mask.png')
            self.assertFalse(list(root.glob('*_clean.png')))


if __name__ == '__main__':
    unittest.main()
