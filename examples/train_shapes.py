"""Train MiniViT on the synthetic shapes task — proof the tokenizer learns.

Runs entirely on CPU in about a minute, no downloads. The script trains a
small ViT (tokenizer included, end to end) to classify circles vs. squares
vs. triangles, then writes:

* ``shapes_samples.png``      — a grid of dataset samples
* ``training_curves.png``     — loss and validation accuracy
* ``attention_maps.png``      — last-layer [CLS] attention over a test image
* ``projection_filters.png``  — principal components of the learned patch filters

Usage::

    python examples/train_shapes.py                  # defaults: ~1 min on CPU
    python examples/train_shapes.py --epochs 8 --positional-encoding rotary
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from vit_tokenizer import MiniViT, SyntheticShapes
from vit_tokenizer.visualize import (
    plot_attention_map,
    plot_projection_filters,
    plot_training_curves,
)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    model.train()
    return correct / total


def plot_samples(dataset: SyntheticShapes, count: int = 12) -> plt.Figure:
    columns = 6
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.0 * columns, 2.2 * rows))
    for index, ax in enumerate(axes.flatten()):
        if index >= count:
            ax.axis("off")
            continue
        image, label = dataset.render(index)
        ax.imshow(image)
        ax.set_title(dataset.CLASSES[label], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("SyntheticShapes samples")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--val-samples", type=int, default=600)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--positional-encoding",
        default="sinusoidal-2d",
        choices=["sinusoidal-1d", "sinusoidal-2d", "learnable", "none", "rotary"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train_set = SyntheticShapes(args.train_samples, args.image_size, seed=args.seed)
    # Disjoint seed -> a genuinely held-out validation set.
    val_set = SyntheticShapes(args.val_samples, args.image_size, seed=args.seed + 1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = MiniViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=len(SyntheticShapes.CLASSES),
        positional_encoding=args.positional_encoding,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"MiniViT: {parameter_count:,} parameters, "
          f"{model.embedding.num_patches} patches/image, "
          f"positional encoding = {args.positional_encoding}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: dict[str, list[float]] = {"train loss": [], "validation accuracy": []}

    for epoch in range(args.epochs):
        running_loss, batches = 0.0, 0
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
            if step % 10 == 9:
                history["train loss"].append(running_loss / batches)
                history["validation accuracy"].append(evaluate(model, val_loader))
                running_loss, batches = 0.0, 0
        print(f"epoch {epoch + 1}/{args.epochs}: "
              f"loss {history['train loss'][-1]:.4f}, "
              f"val accuracy {history['validation accuracy'][-1]:.3f}")

    final_accuracy = evaluate(model, val_loader)
    print(f"final validation accuracy: {final_accuracy:.3f}")

    # --- figures ---------------------------------------------------------
    model.eval()
    sample_image, sample_label = val_set.render(0)
    sample_tensor, _ = val_set[0]
    with torch.no_grad():
        logits, attention_maps = model(sample_tensor, return_attention=True)
    prediction = SyntheticShapes.CLASSES[logits.argmax().item()]

    figures = {
        "shapes_samples.png": plot_samples(train_set),
        "training_curves.png": plot_training_curves(history),
        "attention_maps.png": plot_attention_map(
            sample_image,
            attention_maps[-1],
            args.patch_size,
            title=f"Last-layer [CLS] attention — true: "
                  f"{SyntheticShapes.CLASSES[sample_label]}, predicted: {prediction}",
        ),
        "projection_filters.png": plot_projection_filters(
            model.embedding.projection.weight, args.patch_size
        ),
    }
    for filename, figure in figures.items():
        path = os.path.join(args.out, filename)
        figure.savefig(path, dpi=130, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
