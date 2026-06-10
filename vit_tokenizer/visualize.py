"""Matplotlib visualizations for every stage of the tokenizer.

Each function builds and returns a ``matplotlib.figure.Figure`` so you can
show it interactively (``fig.show()``), save it (``fig.savefig(path)``), or
embed it in a notebook. ``examples/demo.py`` generates the full gallery.

Image arguments accept either a PIL image or a ``(C, H, W)`` float tensor in
``[0, 1]`` (use :meth:`vit_tokenizer.dataset.Normalize.denormalize` first if
the tensor is normalized).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .positional import frequency_bands
from .patching import patch_grid_shape

__all__ = [
    "plot_patch_grid",
    "plot_patches",
    "plot_positional_encoding",
    "plot_position_similarity",
    "plot_frequency_bands",
    "plot_encoding_curves",
    "plot_attention_map",
    "plot_projection_filters",
    "plot_training_curves",
]


def _to_array(image) -> np.ndarray:
    """Coerce a PIL image or (C, H, W) tensor in [0, 1] to an (H, W, C) array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, torch.Tensor):
        if image.dim() != 3:
            raise ValueError(f"Expected a (C, H, W) tensor, got shape {tuple(image.shape)}")
        return image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    raise TypeError(f"Expected a PIL image or torch.Tensor, got {type(image).__name__}")


def plot_patch_grid(image, patch_size: int, linewidth: float = 1.0) -> plt.Figure:
    """Overlay the patch boundaries on the image.

    Shows exactly how the image is carved into the ``N = (H/P)(W/P)`` patches
    that become the transformer's input tokens.
    """
    array = _to_array(image)
    height, width = array.shape[:2]
    grid_h, grid_w = patch_grid_shape((height, width), patch_size)

    fig, ax = plt.subplots(figsize=(6, 6 * height / width))
    ax.imshow(array)
    for row in range(1, grid_h):
        ax.axhline(row * patch_size - 0.5, color="white", linewidth=linewidth, alpha=0.9)
    for col in range(1, grid_w):
        ax.axvline(col * patch_size - 0.5, color="white", linewidth=linewidth, alpha=0.9)
    ax.set_title(f"{grid_h}×{grid_w} grid of {patch_size}×{patch_size} patches "
                 f"(N = {grid_h * grid_w} tokens)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_patches(image, patch_size: int, max_labels: int = 100) -> plt.Figure:
    """Explode the image into its patches, labeled with their sequence index.

    The labels make the raster ordering visible: index ``k`` sits at grid
    position ``(k // grid_w, k % grid_w)``. This is the order in which the
    flattened patches appear in the token sequence (after the [CLS] token).
    """
    array = _to_array(image)
    height, width = array.shape[:2]
    grid_h, grid_w = patch_grid_shape((height, width), patch_size)

    fig, axes = plt.subplots(grid_h, grid_w, figsize=(7, 7 * height / width))
    axes = np.atleast_2d(axes)
    for row in range(grid_h):
        for col in range(grid_w):
            ax = axes[row, col]
            patch = array[
                row * patch_size : (row + 1) * patch_size,
                col * patch_size : (col + 1) * patch_size,
            ]
            ax.imshow(patch)
            ax.set_xticks([])
            ax.set_yticks([])
            index = row * grid_w + col
            if grid_h * grid_w <= max_labels:
                ax.set_title(str(index), fontsize=7, pad=2)
    fig.suptitle(f"Patches in sequence order (raster scan, {grid_h}×{grid_w})")
    fig.tight_layout()
    return fig


def plot_positional_encoding(encoding: torch.Tensor, title: str | None = None) -> plt.Figure:
    """Heatmap of the encoding table: rows are positions, columns dimensions.

    The signature pattern: fast oscillation in the low (leftmost) dimensions
    and progressively slower stripes toward the high dimensions, mirroring the
    geometric frequency progression ``omega_j = base^(-2j/d)``.
    """
    table = encoding.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mesh = ax.imshow(table, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("dimension $i$")
    ax.set_ylabel("position $pos$")
    ax.set_title(title or f"Sinusoidal positional encoding ({table.shape[0]} positions "
                          f"× {table.shape[1]} dims)")
    fig.colorbar(mesh, ax=ax, label="$PE_{(pos,\\,i)}$")
    fig.tight_layout()
    return fig


def plot_position_similarity(encoding: torch.Tensor, metric: str = "dot") -> plt.Figure:
    """Pairwise similarity between all position vectors.

    For the sinusoidal encoding, ``PE[p] . PE[q] = sum_j cos(omega_j (p - q))``
    depends only on ``|p - q|`` (docs/MATH.md section 5.5), so the heatmap is a
    Toeplitz matrix: a bright diagonal band that fades with distance. That
    decay is the signal attention can exploit to favor nearby patches.
    """
    table = encoding.detach()
    if metric == "dot":
        similarity = table @ table.T
        label = "dot product"
    elif metric == "cosine":
        normalized = table / table.norm(dim=1, keepdim=True).clamp_min(1e-12)
        similarity = normalized @ normalized.T
        label = "cosine similarity"
    else:
        raise ValueError(f"metric must be 'dot' or 'cosine', got {metric!r}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mesh = ax.imshow(similarity.cpu().numpy(), cmap="viridis")
    ax.set_xlabel("position $q$")
    ax.set_ylabel("position $p$")
    ax.set_title(f"Position-to-position {label}")
    fig.colorbar(mesh, ax=ax, label=label)
    fig.tight_layout()
    return fig


def plot_frequency_bands(d_model: int, base: float = 10_000.0) -> plt.Figure:
    """Per-pair angular frequency and wavelength on log scales.

    Left: ``omega_j = base^(-2j/d)`` — a geometric decay, i.e. a straight line
    on a log axis. Right: the wavelength ``lambda_j = 2 pi / omega_j``, rising
    from ``2 pi`` positions to ``~ 2 pi base``. Long wavelengths are what let
    the encoding distinguish positions far beyond the training sequence
    length without repeating.
    """
    omega = frequency_bands(d_model, base).numpy()
    pair_index = np.arange(len(omega))
    wavelength = 2 * math.pi / omega

    fig, (ax_freq, ax_wave) = plt.subplots(1, 2, figsize=(11, 4))
    ax_freq.semilogy(pair_index, omega)
    ax_freq.set_xlabel("pair index $j$ (dims $2j$, $2j{+}1$)")
    ax_freq.set_ylabel("$\\omega_j = \\mathrm{base}^{-2j/d}$")
    ax_freq.set_title("Angular frequency per dimension pair")
    ax_freq.grid(True, which="both", alpha=0.3)

    ax_wave.semilogy(pair_index, wavelength)
    ax_wave.set_xlabel("pair index $j$")
    ax_wave.set_ylabel("$\\lambda_j = 2\\pi/\\omega_j$ (positions)")
    ax_wave.set_title("Wavelength per dimension pair")
    ax_wave.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Geometric frequency progression  (d_model={d_model}, base={base:g})")
    fig.tight_layout()
    return fig


def plot_encoding_curves(
    encoding: torch.Tensor,
    pairs: tuple[int, ...] = (0, 2, 8, 32),
) -> plt.Figure:
    """sin/cos value vs. position for a few dimension pairs.

    Reads the heatmap "one column pair at a time": low pairs complete many
    cycles over the sequence (fine-grained position), high pairs barely move
    (coarse position). Together they place every position uniquely, the same
    way digits of different significance place a number.
    """
    table = encoding.detach().cpu().numpy()
    num_positions, d_model = table.shape
    pairs = tuple(j for j in pairs if 2 * j + 1 < d_model)

    fig, axes = plt.subplots(len(pairs), 1, figsize=(9, 1.9 * len(pairs)), sharex=True)
    axes = np.atleast_1d(axes)
    positions = np.arange(num_positions)
    for ax, j in zip(axes, pairs):
        ax.plot(positions, table[:, 2 * j], label=f"$\\sin$ (dim {2 * j})")
        ax.plot(positions, table[:, 2 * j + 1], label=f"$\\cos$ (dim {2 * j + 1})")
        ax.set_ylabel(f"$j={j}$")
        ax.set_ylim(-1.15, 1.15)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("position $pos$")
    fig.suptitle("Encoding values across positions, selected dimension pairs")
    fig.tight_layout()
    return fig


def plot_attention_map(
    image,
    attention: torch.Tensor,
    patch_size: int,
    title: str | None = None,
) -> plt.Figure:
    """Overlay each head's [CLS]-to-patch attention on the image.

    ``attention`` is one layer's weights for one image, shape ``(H, S, S)``
    with ``S = N + 1``. Row 0 is the [CLS] token's query; its entries over
    columns ``1..N`` say how much each patch contributes to the image summary.
    Those ``N`` weights are reshaped onto the patch grid, bilinearly upsampled
    to pixel resolution, and drawn as a heatmap over the image — one panel per
    head plus the head average.
    """
    array = _to_array(image)
    height, width = array.shape[:2]
    grid_h, grid_w = patch_grid_shape((height, width), patch_size)

    attention = attention.detach()
    if attention.dim() != 3 or attention.shape[-1] != grid_h * grid_w + 1:
        raise ValueError(
            f"Expected attention of shape (H, S, S) with S = {grid_h * grid_w + 1}, "
            f"got {tuple(attention.shape)}"
        )
    num_heads = attention.shape[0]
    cls_to_patches = attention[:, 0, 1:]  # (H, N): drop CLS->CLS self-weight
    maps = torch.cat([cls_to_patches, cls_to_patches.mean(0, keepdim=True)])
    labels = [f"head {h}" for h in range(num_heads)] + ["head average"]

    columns = min(len(maps), 5)
    rows = math.ceil(len(maps) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, weights, label in zip(axes, maps, labels):
        grid = weights.reshape(1, 1, grid_h, grid_w)
        upsampled = torch.nn.functional.interpolate(
            grid, size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
        ax.imshow(array)
        ax.imshow(upsampled.cpu().numpy(), cmap="inferno", alpha=0.55)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(maps):]:
        ax.axis("off")
    fig.suptitle(title or "[CLS] → patch attention")
    fig.tight_layout()
    return fig


def plot_projection_filters(
    weight: torch.Tensor,
    patch_size: int,
    in_channels: int = 3,
    num_components: int = 28,
) -> plt.Figure:
    """Principal components of the learned patch-projection filters.

    Each of the ``d`` rows of the projection weight is a linear filter over a
    flattened ``(C, P, P)`` patch. Following the ViT paper (figure 7), we PCA
    the set of filters and display the leading components reshaped to patches:
    after training they typically resemble basis functions — edges, blobs,
    gratings — analogous to a CNN's first-layer kernels.
    """
    filters = weight.detach().cpu().numpy()
    if filters.shape[1] != in_channels * patch_size * patch_size:
        raise ValueError(
            f"weight columns ({filters.shape[1]}) != C*P^2 "
            f"({in_channels * patch_size * patch_size})"
        )
    centered = filters - filters.mean(axis=0, keepdims=True)
    # Rows are d samples in R^{CP^2}; right singular vectors are the PCs.
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    num_components = min(num_components, components.shape[0])

    columns = 7
    rows = math.ceil(num_components / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.5 * columns, 1.6 * rows))
    axes = np.atleast_1d(axes).flatten()
    for index, ax in enumerate(axes):
        if index >= num_components:
            ax.axis("off")
            continue
        component = components[index].reshape(in_channels, patch_size, patch_size)
        component = np.transpose(component, (1, 2, 0))
        low, high = component.min(), component.max()
        ax.imshow((component - low) / (high - low + 1e-12))
        ax.set_title(f"PC {index}", fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Principal components of the patch-projection filters")
    fig.tight_layout()
    return fig


def plot_training_curves(history: dict[str, list[float]]) -> plt.Figure:
    """Loss and accuracy curves from a training run.

    ``history`` maps curve names to per-evaluation values; keys containing
    ``"loss"`` go on the left axis (log scale), the rest on the right.
    """
    loss_keys = [k for k in history if "loss" in k.lower()]
    other_keys = [k for k in history if k not in loss_keys]

    fig, (ax_loss, ax_other) = plt.subplots(1, 2, figsize=(11, 4))
    for key in loss_keys:
        ax_loss.semilogy(history[key], label=key)
    ax_loss.set_xlabel("evaluation step")
    ax_loss.set_ylabel("loss (log scale)")
    ax_loss.set_title("Loss")
    ax_loss.grid(alpha=0.3)
    if loss_keys:
        ax_loss.legend()

    for key in other_keys:
        ax_other.plot(history[key], label=key)
    ax_other.set_xlabel("evaluation step")
    ax_other.set_title(" / ".join(other_keys) if other_keys else "metrics")
    ax_other.grid(alpha=0.3)
    if other_keys:
        ax_other.legend()
    fig.tight_layout()
    return fig
