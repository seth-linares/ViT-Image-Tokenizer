"""End-to-end demo: tokenize an image and visualize every stage.

Usage::

    python examples/demo.py                       # synthetic test image
    python examples/demo.py --image path/to.jpg   # your own image
    python examples/demo.py --patch-size 32 --d-model 256 --out docs/figures

Prints the shape of the tensor at each pipeline stage and writes the figure
gallery (patch grid, exploded patches, positional-encoding heatmap, position
similarity, frequency spectrum, per-dimension curves) to the output directory.
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")

import torch
from PIL import Image, ImageDraw

from vit_tokenizer import (
    Normalize,
    ResizeAndPad,
    ViTPatchEmbedding,
    patchify,
    sinusoidal_positional_encoding,
    to_tensor,
)
from vit_tokenizer.visualize import (
    plot_encoding_curves,
    plot_frequency_bands,
    plot_patch_grid,
    plot_patches,
    plot_position_similarity,
    plot_positional_encoding,
)


def synthetic_image(size: int = 448) -> Image.Image:
    """A colorful structured image so patch boundaries are easy to see."""
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                int(255 * x / size),
                int(255 * y / size),
                int(127 + 128 * math.sin(8 * math.pi * x / size) * math.sin(8 * math.pi * y / size)),
            )
    draw = ImageDraw.Draw(image)
    center, radius = size // 2, size // 4
    draw.ellipse(
        (center - radius, center - radius, center + radius, center + radius),
        outline=(255, 255, 255),
        width=size // 60,
    )
    draw.rectangle(
        (size // 8, size // 8, 3 * size // 8, 3 * size // 8),
        outline=(20, 20, 20),
        width=size // 80,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=str, default=None, help="input image path (default: synthetic)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--out", type=str, default="docs/figures", help="output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    source = Image.open(args.image) if args.image else synthetic_image()
    prepared = ResizeAndPad(args.image_size)(source)
    image_tensor = to_tensor(prepared)
    normalized = Normalize()(image_tensor)

    tokenizer = ViTPatchEmbedding(
        image_size=args.image_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
    )

    patches = patchify(normalized, args.patch_size)
    tokens = tokenizer(normalized)

    grid_h, grid_w = tokenizer.grid_shape
    print("Pipeline shape walkthrough")
    print(f"  input image          : {tuple(image_tensor.shape)}  (C, H, W)")
    print(f"  patch grid           : {grid_h} x {grid_w}  ->  N = {tokenizer.num_patches} patches")
    print(f"  flattened patches    : {tuple(patches.shape)}  (N, C*P^2)")
    print(f"  projected + [CLS] + PE: {tuple(tokens.shape)}  (N+1, d_model)")
    print(f"  trainable parameters : "
          f"{sum(p.numel() for p in tokenizer.parameters()):,}")

    encoding = sinusoidal_positional_encoding(tokenizer.sequence_length, args.d_model)
    figures = {
        "patch_grid.png": plot_patch_grid(image_tensor, args.patch_size),
        "patches_exploded.png": plot_patches(image_tensor, args.patch_size),
        "positional_encoding.png": plot_positional_encoding(encoding),
        "position_similarity.png": plot_position_similarity(encoding),
        "frequency_bands.png": plot_frequency_bands(args.d_model),
        "encoding_curves.png": plot_encoding_curves(encoding),
    }
    for filename, figure in figures.items():
        path = os.path.join(args.out, filename)
        figure.savefig(path, dpi=130, bbox_inches="tight")
        print(f"  wrote {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
