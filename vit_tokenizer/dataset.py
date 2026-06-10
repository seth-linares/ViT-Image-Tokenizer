"""Image loading and preprocessing, self-contained (no torchvision required).

Provides the three preprocessing steps a ViT needs before patchification:

1. :class:`ResizeAndPad` — scale to the target size *preserving aspect ratio*
   and pad the remainder, so images are never distorted and the result is
   always exactly ``image_size x image_size`` (hence divisible into patches
   whenever ``patch_size | image_size``).
2. :func:`to_tensor` — PIL image to a float ``(C, H, W)`` tensor in ``[0, 1]``.
3. :class:`Normalize` — per-channel standardization ``(x - mean) / std``,
   defaulting to the ImageNet statistics that nearly all pretrained vision
   models assume (docs/MATH.md section 8).

:class:`ImageFolderDataset` is a minimal stand-in for
``torchvision.datasets.ImageFolder``: each subdirectory of the root is a
class, labels are assigned by sorted directory name. If torchvision is
available you can use its ``ImageFolder`` with ``default_transform()``
interchangeably.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "Compose",
    "ImageFolderDataset",
    "Normalize",
    "ResizeAndPad",
    "default_transform",
    "to_tensor",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")


class ResizeAndPad:
    """Resize to fit inside ``image_size x image_size``, then pad to fill it.

    The scale factor is ``min(target/w, target/h)`` so the *larger* dimension
    lands exactly on the target and the image is never cropped or stretched.
    The shorter dimension is centered and the margins are filled with
    ``background_color``. Output is always RGB.
    """

    def __init__(
        self,
        image_size: int = 224,
        background_color: tuple[int, int, int] = (0, 0, 0),
        interpolation: Image.Resampling = Image.Resampling.LANCZOS,
    ) -> None:
        self.image_size = image_size
        self.background_color = background_color
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        scale = min(self.image_size / image.width, self.image_size / image.height)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, self.interpolation)

        canvas = Image.new("RGB", (self.image_size, self.image_size), self.background_color)
        offset = ((self.image_size - new_size[0]) // 2, (self.image_size - new_size[1]) // 2)
        canvas.paste(image, offset)
        return canvas

    def __repr__(self) -> str:
        return f"{type(self).__name__}(image_size={self.image_size})"


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float32 ``(C, H, W)`` tensor scaled to ``[0, 1]``.

    PIL stores pixels as ``(H, W, C)`` uint8 in ``[0, 255]``; networks expect
    channels-first floats, so we permute and divide by 255.
    """
    array = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    tensor = array.view(image.height, image.width, 3).permute(2, 0, 1)
    return tensor.float() / 255.0


class Normalize:
    """Per-channel standardization: ``x[c] = (x[c] - mean[c]) / std[c]``."""

    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Invert the normalization (useful for visualization)."""
        return tensor * self.std + self.mean

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(mean={self.mean.flatten().tolist()}, "
            f"std={self.std.flatten().tolist()})"
        )


class Compose:
    """Chain callables left to right: ``Compose([f, g, h])(x) == h(g(f(x)))``."""

    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value

    def __repr__(self) -> str:
        steps = ", ".join(repr(t) for t in self.transforms)
        return f"{type(self).__name__}([{steps}])"


def default_transform(
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> Compose:
    """The standard PIL-to-model-input pipeline: resize+pad, to tensor, normalize."""
    return Compose(
        [
            ResizeAndPad(image_size, background_color=background_color),
            to_tensor,
            Normalize(mean, std),
        ]
    )


class ImageFolderDataset(Dataset):
    """``root/<class_name>/<image>`` directory layout to ``(tensor, label)`` pairs.

    Class indices follow sorted subdirectory names, matching torchvision's
    convention. Files with non-image extensions are ignored. Unreadable or
    corrupt images raise immediately with the offending *path* in the message
    (an index alone, as in the PoC, leaves you grepping for the bad file).
    """

    def __init__(self, root: str, transform: Callable | None = None) -> None:
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Dataset root is not a directory: {root!r}")
        self.root = root
        self.transform = transform if transform is not None else default_transform()

        self.classes = sorted(
            entry.name for entry in os.scandir(root) if entry.is_dir()
        )
        if not self.classes:
            raise ValueError(
                f"No class subdirectories found in {root!r}. Expected layout: "
                "root/<class_name>/<image files>."
            )
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}

        self.samples: list[tuple[str, int]] = []
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    self.samples.append(
                        (os.path.join(class_dir, filename), self.class_to_index[class_name])
                    )
        if not self.samples:
            raise ValueError(f"No image files found under {root!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image.load()
                return self.transform(image), label
        except UnidentifiedImageError as error:
            raise RuntimeError(f"Corrupt or unreadable image file: {path!r}") from error
