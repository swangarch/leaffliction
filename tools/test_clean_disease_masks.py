"""Checks for clipping class-colored disease masks without changing labels."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from clean_disease_masks import clip_disease_mask, process, validate_pair


class DiseaseCleanupTests(unittest.TestCase):
    def test_rgb_labels_inside_preserved_and_outside_zeroed(self):
        disease = np.array([[[230, 45, 60], [0, 180, 255]],
                            [[255, 220, 0], [0, 0, 0]]], dtype=np.uint8)
        leaf = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        clean = clip_disease_mask(disease, leaf)
        np.testing.assert_array_equal(clean[leaf > 0], disease[leaf > 0])
        self.assertFalse(clean[leaf == 0].any())
        self.assertTrue(disease[0, 1].any())

    def test_class_ids_and_empty_leaf(self):
        disease = np.array([[1, 2], [6, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(
            clip_disease_mask(disease, np.full((2, 2), 255)), disease)
        self.assertFalse(clip_disease_mask(disease, np.zeros((2, 2))).any())
        with self.assertRaises(ValueError):
            clip_disease_mask(disease, np.zeros((3, 3)))

    def test_original_backup_preserved_on_repeated_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'annotations'
            root.mkdir()
            backup = Path(directory) / 'backup'
            path = root / 'example_disease_mask.png'
            leaf = root / 'example_leaf_mask_clean.png'
            disease = np.full((4, 4, 3), [0, 180, 255], np.uint8)
            Image.fromarray(disease).save(path)
            original = path.read_bytes()
            Image.fromarray(np.eye(4, dtype=np.uint8) * 255).save(leaf)
            validate_pair(path)
            first = process(path, root, backup)
            second = process(path, root, backup)
            self.assertEqual(first['removed_pixels'], 12)
            self.assertEqual(second['removed_pixels'], 0)
            self.assertEqual((backup / path.name).read_bytes(), original)
            with Image.open(path) as image:
                self.assertEqual(image.mode, 'RGB')


if __name__ == '__main__':
    unittest.main()
