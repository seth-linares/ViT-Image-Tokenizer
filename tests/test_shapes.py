"""Tests for the SyntheticShapes dataset: determinism, balance, integration."""

import pytest
import torch

from vit_tokenizer import SyntheticShapes


class TestSyntheticShapes:
    def test_length_and_classes(self):
        dataset = SyntheticShapes(num_samples=30, image_size=32)
        assert len(dataset) == 30
        assert dataset.CLASSES == ("circle", "square", "triangle")

    def test_deterministic_across_instances(self):
        """Same (seed, index) must yield identical pixels — the dataset
        behaves like fixed files on disk."""
        first = SyntheticShapes(num_samples=10, image_size=32, seed=42)
        second = SyntheticShapes(num_samples=10, image_size=32, seed=42)
        for index in range(10):
            image_a, label_a = first[index]
            image_b, label_b = second[index]
            assert torch.equal(image_a, image_b)
            assert label_a == label_b

    def test_deterministic_regardless_of_access_order(self):
        dataset = SyntheticShapes(num_samples=10, image_size=32, seed=1)
        late = dataset[7][0]
        dataset[0], dataset[3]  # touch other indices in between
        assert torch.equal(dataset[7][0], late)

    def test_different_seeds_differ(self):
        first = SyntheticShapes(num_samples=5, image_size=32, seed=0)
        second = SyntheticShapes(num_samples=5, image_size=32, seed=1)
        assert not torch.equal(first[0][0], second[0][0])

    def test_exact_class_balance(self):
        dataset = SyntheticShapes(num_samples=30, image_size=32)
        labels = [dataset.render(i)[1] for i in range(30)]
        assert labels.count(0) == labels.count(1) == labels.count(2) == 10

    def test_tensor_shape_and_normalized_range(self):
        dataset = SyntheticShapes(num_samples=3, image_size=48)
        image, _ = dataset[0]
        assert image.shape == (3, 48, 48)
        # Default transform standardizes to mean 0.5 / std 0.5: range [-1, 1].
        assert image.min() >= -1.0 and image.max() <= 1.0

    def test_shape_visible(self):
        """Foreground (>=140) on dark background (<90) must produce real
        contrast: some but not all pixels above the threshold."""
        import numpy as np

        dataset = SyntheticShapes(num_samples=9, image_size=32, transform=lambda im: im)
        for index in range(9):
            image, _ = dataset[index]
            bright = int((np.asarray(image).max(axis=2) >= 140).sum())
            assert 0 < bright < 32 * 32

    def test_out_of_range_index_raises(self):
        dataset = SyntheticShapes(num_samples=3, image_size=32)
        with pytest.raises(IndexError):
            dataset.render(3)

    def test_invalid_num_samples_raises(self):
        with pytest.raises(ValueError):
            SyntheticShapes(num_samples=0)

    def test_dataloader_integration(self):
        from torch.utils.data import DataLoader

        dataset = SyntheticShapes(num_samples=12, image_size=32)
        images, labels = next(iter(DataLoader(dataset, batch_size=12)))
        assert images.shape == (12, 3, 32, 32)
        assert sorted(set(labels.tolist())) == [0, 1, 2]
