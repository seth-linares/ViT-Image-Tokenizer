"""Tests for the preprocessing transforms and the folder dataset."""

import pytest
import torch
from PIL import Image

from vit_tokenizer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Compose,
    ImageFolderDataset,
    Normalize,
    ResizeAndPad,
    default_transform,
    to_tensor,
)


def solid_image(width, height, color=(200, 30, 90)):
    return Image.new("RGB", (width, height), color)


class TestResizeAndPad:
    def test_output_size_is_square_target(self):
        transform = ResizeAndPad(image_size=224)
        for size in [(50, 400), (400, 50), (224, 224), (10, 10)]:
            assert transform(solid_image(*size)).size == (224, 224)

    def test_aspect_ratio_preserved(self):
        """A 2:1 image scaled into a 100x100 canvas occupies 100x50; the rest
        is padding, so content rows differ from padding rows."""
        transform = ResizeAndPad(image_size=100, background_color=(0, 0, 0))
        result = transform(solid_image(200, 100, color=(255, 255, 255)))
        tensor = to_tensor(result)
        assert tensor[:, 50, 50].sum() == pytest.approx(3.0)  # center: content
        assert tensor[:, 5, 50].sum() == pytest.approx(0.0)  # top edge: padding

    def test_content_is_centered(self):
        transform = ResizeAndPad(image_size=100, background_color=(0, 0, 0))
        tensor = to_tensor(transform(solid_image(200, 100, color=(255, 255, 255))))
        content_rows = (tensor.sum(dim=(0, 2)) > 0).nonzero().flatten()
        top_margin = content_rows[0].item()
        bottom_margin = 99 - content_rows[-1].item()
        assert abs(top_margin - bottom_margin) <= 1

    def test_converts_to_rgb(self):
        grayscale = Image.new("L", (64, 64), 128)
        assert ResizeAndPad(32)(grayscale).mode == "RGB"


class TestToTensor:
    def test_shape_and_range(self):
        tensor = to_tensor(solid_image(20, 10))
        assert tensor.shape == (3, 10, 20)
        assert tensor.dtype == torch.float32
        assert 0.0 <= tensor.min() and tensor.max() <= 1.0

    def test_channel_values(self):
        tensor = to_tensor(solid_image(4, 4, color=(255, 0, 51)))
        assert torch.allclose(tensor[0], torch.ones(4, 4))
        assert torch.allclose(tensor[1], torch.zeros(4, 4))
        assert torch.allclose(tensor[2], torch.full((4, 4), 51 / 255))


class TestNormalize:
    def test_standardizes_channels(self):
        normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        tensor = torch.rand(3, 8, 8)
        result = normalize(tensor)
        for c in range(3):
            expected = (tensor[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            assert torch.allclose(result[c], expected)

    def test_denormalize_round_trip(self):
        normalize = Normalize()
        tensor = torch.rand(3, 8, 8)
        assert torch.allclose(normalize.denormalize(normalize(tensor)), tensor, atol=1e-6)


class TestDefaultTransform:
    def test_end_to_end(self):
        transform = default_transform(image_size=64)
        tensor = transform(solid_image(123, 77))
        assert tensor.shape == (3, 64, 64)
        assert tensor.dtype == torch.float32

    def test_compose_order(self):
        doubler = Compose([lambda x: x + 1, lambda x: x * 10])
        assert doubler(0) == 10


class TestImageFolderDataset:
    @pytest.fixture
    def dataset_root(self, tmp_path):
        for class_name, color in [("cats", (255, 0, 0)), ("dogs", (0, 0, 255))]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(2):
                solid_image(40, 30, color).save(class_dir / f"{i}.png")
            (class_dir / "notes.txt").write_text("not an image")
        return tmp_path

    def test_discovers_classes_sorted(self, dataset_root):
        dataset = ImageFolderDataset(str(dataset_root), transform=to_tensor)
        assert dataset.classes == ["cats", "dogs"]
        assert dataset.class_to_index == {"cats": 0, "dogs": 1}

    def test_skips_non_images(self, dataset_root):
        dataset = ImageFolderDataset(str(dataset_root), transform=to_tensor)
        assert len(dataset) == 4

    def test_getitem_returns_tensor_and_label(self, dataset_root):
        dataset = ImageFolderDataset(str(dataset_root), transform=default_transform(64))
        tensor, label = dataset[0]
        assert tensor.shape == (3, 64, 64)
        assert label == 0

    def test_works_with_dataloader_and_tokenizer(self, dataset_root):
        """End to end: folder -> DataLoader batch -> token sequences."""
        from torch.utils.data import DataLoader

        from vit_tokenizer import ViTPatchEmbedding

        dataset = ImageFolderDataset(str(dataset_root), transform=default_transform(32))
        loader = DataLoader(dataset, batch_size=4)
        tokenizer = ViTPatchEmbedding(image_size=32, patch_size=16, d_model=64)
        images, labels = next(iter(loader))
        tokens = tokenizer(images)
        assert tokens.shape == (4, 5, 64)
        assert labels.tolist() == [0, 0, 1, 1]

    def test_corrupt_image_reports_path(self, dataset_root):
        bad_path = dataset_root / "cats" / "broken.jpg"
        bad_path.write_bytes(b"this is not a jpeg")
        dataset = ImageFolderDataset(str(dataset_root), transform=to_tensor)
        bad_index = next(
            i for i, (path, _) in enumerate(dataset.samples) if path.endswith("broken.jpg")
        )
        with pytest.raises(RuntimeError, match="broken.jpg"):
            dataset[bad_index]

    def test_missing_root_raises(self):
        with pytest.raises(NotADirectoryError):
            ImageFolderDataset("/nonexistent/path")

    def test_empty_root_raises(self, tmp_path):
        with pytest.raises(ValueError, match="class subdirectories"):
            ImageFolderDataset(str(tmp_path))
