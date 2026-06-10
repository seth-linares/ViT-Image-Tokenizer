# ViT Image Tokenizer

A tested, documented, from-scratch implementation of the part of a Vision
Transformer that nobody explains properly: how an image becomes a sequence of
tokens — plus the minimal transformer that proves the tokens work.

```
images (B, C, H, W)
  └─ patchify ──────────► (B, N, C·P²)     N = (H/P)(W/P) flattened patches
      └─ linear projection ► (B, N, d)     z = Wx + b, learnable
          └─ prepend [CLS] ► (B, N+1, d)   learnable summary token
              └─ + positional encoding ► (B, N+1, d)   ready for the transformer
                  └─ MiniViT: L × [pre-norm attention + MLP] ► logits
```

This began as a learning PoC inspired by Karpathy's *"Let's build the GPT
Tokenizer"* — the question being *"what is the image equivalent of a
tokenizer?"* It has since been rebuilt into a proper package: every design
choice is derived mathematically in [`docs/MATH.md`](docs/MATH.md), every
derived property is verified numerically by the test suite, and every stage
can be visualized — including training a small ViT end to end on CPU in about
a minute. The original scripts are preserved in [`legacy/`](legacy/) with a
post-mortem of their bugs.

## Highlights

- **Correct, minimal, dependency-light.** `torch`, `numpy`, `pillow`,
  `matplotlib` — no torchvision required.
- **The math is written down and tested.** [`docs/MATH.md`](docs/MATH.md)
  derives the patch algebra, the projection ↔ strided-convolution equivalence,
  the geometric frequency spectrum, the exact norm $\|PE\|^2 = d/2$, the
  Toeplitz similarity structure, the rotation-matrix relative-position
  property, the √d attention-scaling variance argument, the permutation-
  equivariance proof, the RoPE relative-position theorem, and the Xavier
  variance argument. Each claim cites the test that checks it; 118 tests pass.
- **It provably learns.** From-scratch multi-head attention (numerically
  identical to `nn.MultiheadAttention`), a pre-norm `MiniViT`, a zero-download
  synthetic shapes dataset, and a CPU training demo reaching ~96% held-out
  accuracy in a minute — with attention-map visualizations to show *what* it
  learned.
- **Visualize everything.** Patch grids, exploded patch sequences,
  encoding heatmaps, position-similarity matrices, frequency spectra,
  per-dimension curves, [CLS]-attention overlays, learned-filter principal
  components, training curves.
- **Five positional schemes**: sinusoidal 1-D (original Transformer),
  factorized sinusoidal 2-D (DETR/MAE-style), learnable (the actual ViT
  paper), rotary/RoPE (relative positions inside attention), or none (for
  ablations).

## Install

```bash
git clone https://github.com/seth-linares/ViT-Image-Tokenizer
cd ViT-Image-Tokenizer
pip install -e .          # add [dev] for pytest
```

## Quickstart

```python
import torch
from vit_tokenizer import ViTPatchEmbedding

tokenizer = ViTPatchEmbedding(
    image_size=224,
    patch_size=16,
    d_model=768,
    positional_encoding="sinusoidal-1d",   # or "sinusoidal-2d" | "learnable" | "none"
)

images = torch.randn(8, 3, 224, 224)
tokens = tokenizer(images)                 # (8, 197, 768) — 196 patches + [CLS]
```

`ViTPatchEmbedding` is an `nn.Module`: drop it in as the first layer of a ViT
and the projection matrix, bias, and class token train with the rest of the
model. Or use the included minimal ViT directly:

```python
from vit_tokenizer import MiniViT

model = MiniViT(
    image_size=64, patch_size=8, d_model=96, depth=3, num_heads=4,
    num_classes=3,
    positional_encoding="sinusoidal-2d",   # ... or "rotary" for RoPE attention
)
logits = model(torch.randn(8, 3, 64, 64))              # (8, 3)
logits, attn = model(images, return_attention=True)    # per-layer (B, H, S, S)
```

Loading real images from a `root/<class>/<image>` folder layout:

```python
from torch.utils.data import DataLoader
from vit_tokenizer import ImageFolderDataset, default_transform

dataset = ImageFolderDataset("path/to/data", transform=default_transform(image_size=224))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(loader))
tokens = tokenizer(images)                 # (32, 197, 768)
```

`default_transform` resizes preserving aspect ratio, pads to square, converts
to a tensor, and normalizes with ImageNet statistics — all implemented in
this repo (see [`vit_tokenizer/dataset.py`](vit_tokenizer/dataset.py)).

The lower-level pieces are exposed directly if you want to build things up
yourself:

```python
from vit_tokenizer import patchify, unpatchify, sinusoidal_positional_encoding

patches = patchify(images, patch_size=16)        # (32, 196, 768)
assert torch.equal(unpatchify(patches, 16, 224), images)   # exact inverse

pe = sinusoidal_positional_encoding(197, 768)    # (197, 768)
```

## Visual tour

Generate the full gallery (and a shape walkthrough) with:

```bash
python examples/demo.py                        # synthetic image
python examples/demo.py --image yourphoto.jpg --patch-size 16
```

### The image becomes a sequence

The image is carved into an evenly divisible grid; each cell is flattened in
raster order into one row of the token matrix:

![patch grid](docs/figures/patch_grid.png)

![exploded patches](docs/figures/patches_exploded.png)

### The positional encoding table

Rows are positions, columns are dimensions. Low dimensions oscillate fast
(fine position), high dimensions slowly (coarse position) — a continuous
analogue of binary counting:

![positional encoding heatmap](docs/figures/positional_encoding.png)

![encoding curves](docs/figures/encoding_curves.png)

### Why this particular encoding?

The dot product between two position vectors depends only on their *offset*
(the matrix below is constant along diagonals), giving attention a built-in
notion of "nearby". And the frequencies form a geometric progression spanning
wavelengths from $2\pi$ to $\approx 2\pi \cdot 10^4$ positions:

![position similarity](docs/figures/position_similarity.png)

![frequency bands](docs/figures/frequency_bands.png)

The full story — including the proof that a position shift is a fixed
block-rotation linear map, the property RoPE builds on — is in
[`docs/MATH.md` § 5](docs/MATH.md#5-sinusoidal-positional-encodings).

### Proof it learns: train a ViT on CPU in a minute

```bash
python examples/train_shapes.py                          # ~96% val accuracy, ~1 min CPU
python examples/train_shapes.py --positional-encoding rotary
```

The script trains a 355k-parameter `MiniViT` (tokenizer included, end to end)
to classify procedurally generated circles, squares, and triangles — no
downloads, fully deterministic. It writes training curves and, more
interestingly, the last layer's [CLS]→patch attention overlaid on a held-out
image: the heads visibly lock onto the shape, which is the spatial
information the tokenizer was built to carry.

![attention maps](docs/figures/attention_maps.png)

![training curves](docs/figures/training_curves.png)

The attention math (scaled dot-product, the √d variance argument, the
permutation-equivariance proof, RoPE's relative-position theorem) is in
[`docs/MATH.md` §§ 9–11](docs/MATH.md#9-scaled-dot-product-attention).

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

The suite is organized as executable math: equivalence of the two `div_term`
formulations seen in the wild, entrywise agreement with the paper's formula,
exact patchify/unpatchify round-trips, the linear-projection ↔ `Conv2d`
identity, elementwise equivalence with `nn.MultiheadAttention`, the
permutation-equivariance of attention (and its deliberate loss under RoPE),
the unit-variance effect of √d scaling checked statistically, gradient flow
to every learnable parameter, ViT-Base's published ~86M parameter count
reproduced from the formula, the geometric properties of the encodings
(norms, Toeplitz Gram matrix, rotation property), and an overfit-a-tiny-batch
test proving the full stack trains. CI runs the suite on every push.

## Repository layout

```
vit_tokenizer/
  patching.py     patchify / unpatchify — pure reindexings, exactly invertible
  positional.py   sinusoidal encodings (1-D, factorized 2-D) + RoPE
  embedding.py    ViTPatchEmbedding: projection + [CLS] + positions (nn.Module)
  attention.py    multi-head self-attention from scratch (optional RoPE)
  model.py        TransformerEncoderBlock + MiniViT (pre-norm, GELU MLP)
  shapes.py       SyntheticShapes: zero-download, deterministic dataset
  dataset.py      ResizeAndPad, to_tensor, Normalize, ImageFolderDataset
  visualize.py    figure builders for every stage
tests/                    118 tests; each MATH.md claim cites its test
docs/MATH.md              full derivations (tokenizer §§1–8, model §§9–11)
docs/figures/             generated gallery
examples/demo.py          tokenizer walkthrough + figure generation
examples/train_shapes.py  train MiniViT on CPU, attention/filter figures
legacy/                   the original PoC scripts, with a bug post-mortem
```

## What changed from the PoC

The original scripts (preserved in [`legacy/`](legacy/)) had the right ideas
and several real bugs — including a patch-flattening reshape that mixed pixels
across patches, learnable parameters stored on a `Dataset` where no optimizer
could see them, and an `__getitem__` that crashed on tensor-shape mismatches.
Each issue is documented in [`legacy/README.md`](legacy/README.md) alongside
the fix and the regression test that now guards it.

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Let's build the GPT Tokenizer — Karpathy](https://youtu.be/zduSFxRajkE)
- [Transformer Architecture: The Positional Encoding — Kazemnejad](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [The Illustrated Transformer — Alammar](https://jalammar.github.io/illustrated-transformer/)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Full bibliography in [`docs/MATH.md` § 12](docs/MATH.md#12-references)
