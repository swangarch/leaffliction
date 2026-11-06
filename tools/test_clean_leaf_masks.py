"""Regression checks: python -m unittest discover -s tools -p 'test_*.py'."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from clean_leaf_masks import clean_leaf_mask, process


class LeafCleanupTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((100, 100, 3), 180, np.uint8)
        self.mask = np.zeros((100, 100), np.uint8)
        self.rgb[20:80, 20:70] = [45, 125, 35]
        self.mask[20:80, 20:85] = 255
        self.mask[20:80, 73:85] = 0
        self.rgb[20:80, 70:73] = 65
        self.rgb[40:50, 40:50] = 30

    def test_shadow_removed_and_enclosed_lesion_preserved(self):
        clean = clean_leaf_mask(self.rgb, self.mask)
        self.assertFalse(clean[20:80, 72:73].any())
        self.assertTrue(clean[20:80, 20:70].all())
        self.assertTrue(np.all(clean <= self.mask))
        self.assertEqual(set(np.unique(clean)), {0, 255})

    def test_broad_olive_shadow_is_not_protected_as_green_leaf(self):
        rgb = np.full((100, 100, 3), [130, 125, 110], np.uint8)
        mask = np.zeros((100, 100), np.uint8)
        mask[20:80, 20:90] = 255
        rgb[20:80, 20:65] = [70, 120, 30]
        # This cast shadow has HSV hue 30, inside the old green interval.
        rgb[20:80, 65:90] = [65, 65, 50]
        clean = clean_leaf_mask(rgb, mask)
        self.assertFalse(clean[25:75, 70:88].any())
        self.assertTrue(clean[25:75, 25:60].all())

    def test_shadowed_green_tissue_and_thin_stem_are_preserved(self):
        rgb = self.rgb.copy()
        rgb[20:80, 20:70] = [25, 60, 42]
        mask = self.mask.copy()
        rgb[80:95, 42:45] = [100, 120, 65]
        mask[80:95, 42:45] = 255
        clean = clean_leaf_mask(rgb, mask)
        self.assertTrue(clean[25:75, 25:65].all())
        self.assertTrue(clean[82:93, 43].all())

    def test_waxy_blue_gray_leaf_highlight_is_preserved(self):
        rgb = np.full((100, 100, 3), [130, 110, 140], np.uint8)
        mask = np.zeros((100, 100), np.uint8)
        mask[20:80, 20:75] = 255
        rgb[20:80, 20:75] = [65, 95, 50]
        rgb[45:80, 40:75] = [150, 165, 175]
        clean = clean_leaf_mask(rgb, mask)
        self.assertTrue(clean[50:78, 45:73].all())

    def test_ambiguous_dark_leaf_interior_is_protected(self):
        self.rgb[20:80, 20:70] = [30, 29, 28]
        clean = clean_leaf_mask(self.rgb, self.mask)
        self.assertTrue(clean[25:75, 25:65].all())

    def test_no_background_and_empty_masks(self):
        for value in (0, 255):
            mask = np.full_like(self.mask, value)
            np.testing.assert_array_equal(
                clean_leaf_mask(self.rgb, mask), mask)

    def test_outputs_preserve_source_and_use_masks_as_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'example'
            source = base.with_suffix('.png')
            mask = Path(f'{base}_leaf_mask.png')
            Image.fromarray(self.rgb).save(source)
            Image.fromarray(self.mask).save(mask)
            originals = source.read_bytes(), mask.read_bytes()
            record = process(mask)
            self.assertGreater(record['removed_pixels'], 0)
            outputs = [
                ('leaf_segmentation', self.mask),
                ('leaf_segmentation_clean',
                 clean_leaf_mask(self.rgb, self.mask)),
            ]
            for suffix, expected in outputs:
                with Image.open(f'{base}_{suffix}.png') as image:
                    self.assertEqual(image.mode, 'RGBA')
                    result = np.array(image)
                np.testing.assert_array_equal(result[..., 3], expected)
                np.testing.assert_array_equal(
                    result[expected > 0, :3], self.rgb[expected > 0])
            self.assertEqual(
                originals, (source.read_bytes(), mask.read_bytes()))


if __name__ == '__main__':
    unittest.main()
