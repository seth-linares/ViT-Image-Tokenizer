"""A self-contained synthetic dataset: classify circles, squares, triangles.

Why synthetic? It keeps the repository runnable anywhere — no downloads, no
licenses, no disk cache — while still being a *real* recognition task: the
shape varies in position, size, rotation, and color against a varying
background, so the model must learn shape, not memorize pixels.

Every sample is generated deterministically from ``(seed, index)``, so the
dataset behaves like a fixed on-disk dataset (same index -> same image,
verified in ``tests/test_shapes.py``) while occupying zero storage. Labels
cycle ``index % 3``, giving exact class balance.
"""

from __future__ import annotations

import math

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .dataset import Compose, Normalize, to_tensor

__all__ = ["SyntheticShapes"]


class SyntheticShapes(Dataset):
    """Procedurally generated shape-classification dataset.

    Args:
        num_samples: dataset length.
        image_size: square image side in pixels.
        seed: master seed; two datasets with equal seeds are identical.
        transform: PIL -> tensor transform. Defaults to ``to_tensor`` plus
            mean-0.5/std-0.5 normalization (ImageNet statistics would be
            meaningless for synthetic images).

    Yields ``(image_tensor, label)`` with labels indexing ``CLASSES``.
    """

    CLASSES = ("circle", "square", "triangle")

    def __init__(
        self,
        num_samples: int = 3000,
        image_size: int = 64,
        seed: int = 0,
        transform=None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.transform = transform if transform is not None else Compose(
            [to_tensor, Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __len__(self) -> int:
        return self.num_samples

    def render(self, index: int) -> tuple[Image.Image, int]:
        """Draw sample ``index`` as a PIL image (used directly by visualizers)."""
        if not 0 <= index < self.num_samples:
            raise IndexError(f"index {index} out of range for {self.num_samples} samples")
        # One private generator per (seed, index): determinism independent of
        # access order, DataLoader workers, or global RNG state.
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)

        def uniform(low: float, high: float) -> float:
            return low + (high - low) * torch.rand((), generator=generator).item()

        label = index % len(self.CLASSES)
        size = self.image_size
        # Dark background, bright shape: guaranteed contrast at any colors.
        background = tuple(int(uniform(0, 90)) for _ in range(3))
        foreground = tuple(int(uniform(140, 255)) for _ in range(3))
        radius = uniform(0.16, 0.34) * size
        center_x = uniform(radius + 2, size - radius - 2)
        center_y = uniform(radius + 2, size - radius - 2)
        rotation = uniform(0, 2 * math.pi)

        image = Image.new("RGB", (size, size), background)
        draw = ImageDraw.Draw(image)
        shape = self.CLASSES[label]
        if shape == "circle":
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=foreground,
            )
        else:
            sides = 4 if shape == "square" else 3
            vertices = [
                (
                    center_x + radius * math.cos(rotation + 2 * math.pi * k / sides),
                    center_y + radius * math.sin(rotation + 2 * math.pi * k / sides),
                )
                for k in range(sides)
            ]
            draw.polygon(vertices, fill=foreground)
        return image, label

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.render(index)
        return self.transform(image), label
