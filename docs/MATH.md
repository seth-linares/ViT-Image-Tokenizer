# The Mathematics of ViT Image Tokenization

Every equation in this document is implemented in `vit_tokenizer/` and
verified numerically by a named test in `tests/`. Where a property is claimed,
the test that checks it is cited so you can see the math run.

## Contents

1. [Notation and pipeline overview](#1-notation-and-pipeline-overview)
2. [Patch extraction](#2-patch-extraction)
3. [Linear projection](#3-linear-projection)
4. [The class token](#4-the-class-token)
5. [Sinusoidal positional encodings](#5-sinusoidal-positional-encodings)
6. [2-D positional encodings](#6-2-d-positional-encodings)
7. [Xavier initialization](#7-xavier-initialization)
8. [Input normalization](#8-input-normalization)
9. [Scaled dot-product attention](#9-scaled-dot-product-attention)
10. [Rotary positional embeddings (RoPE)](#10-rotary-positional-embeddings-rope)
11. [From tokens to a working model: MiniViT](#11-from-tokens-to-a-working-model-minivit)
12. [References](#12-references)

---

## 1. Notation and pipeline overview

| Symbol | Meaning | Default |
|---|---|---|
| $C$ | image channels | 3 |
| $H \times W$ | image height × width | $224 \times 224$ |
| $P$ | patch side length | 16 |
| $N$ | number of patches | $(H/P)(W/P) = 196$ |
| $d$ | model (embedding) dimension | 512 |
| $B$ | batch size | — |

The tokenizer is the composition

$$
\underbrace{x \in \mathbb{R}^{B \times C \times H \times W}}_{\text{images}}
\;\xrightarrow{\text{patchify}}\;
\underbrace{X \in \mathbb{R}^{B \times N \times CP^2}}_{\text{flattened patches}}
\;\xrightarrow{\;XW^\top + b\;}\;
\mathbb{R}^{B \times N \times d}
\;\xrightarrow{\;[\mathrm{CLS}]\;}\;
\mathbb{R}^{B \times (N+1) \times d}
\;\xrightarrow{\;+\,PE\;}\;
\underbrace{Z \in \mathbb{R}^{B \times (N+1) \times d}}_{\text{transformer input}}
$$

implemented in [`embedding.py`](../vit_tokenizer/embedding.py) as
`ViTPatchEmbedding.forward`.

Why a sequence at all? Self-attention is defined on *sets* of vectors; it is
**permutation-equivariant**: for any permutation matrix $\Pi$,
$\mathrm{Attn}(\Pi Z) = \Pi\,\mathrm{Attn}(Z)$. The architecture therefore has
no built-in notion of "where" — every piece of spatial information the
transformer will ever have must be injected here, by the positional encoding
(section 5).

---

## 2. Patch extraction

### 2.1 Counting and shapes

With $P \mid H$ and $P \mid W$, the image divides into a grid of
$g_h = H/P$ rows and $g_w = W/P$ columns of patches:

$$
N = g_h \, g_w = \frac{H}{P}\cdot\frac{W}{P} = \frac{HW}{P^2}.
$$

For ViT-Base/16: $N = 224^2/16^2 = 196$ patches, each containing
$CP^2 = 3 \cdot 256 = 768$ pixel values
(verified in `test_embedding.py::TestShapes::test_vit_base_dimensions`).

Patch $k \in \{0, \dots, N-1\}$ sits at grid coordinates

$$
(r, c) = \left(\left\lfloor k / g_w \right\rfloor,\; k \bmod g_w\right)
$$

— **raster order**: left to right, then top to bottom, like reading a page.
The flattened patch vector stacks the patch's pixels in $(C, P, P)$ order
(all of channel 0, then channel 1, …), each channel row-major.
`test_patching.py::TestPatchify::test_matches_naive_crop_loop` checks both
orderings against an explicit double loop.

### 2.2 Patchify is a permutation

`patchify` moves no data and computes nothing — it is a *reindexing*
(a permutation of the $BCHW$ entries):

$$
X[b,\; r g_w + c,\; \kappa P^2 + u P + v] \;=\; x[b,\; \kappa,\; rP+u,\; cP+v]
$$

for channel $\kappa$, intra-patch offsets $u, v \in \{0,\dots,P-1\}$. Because
it is a bijection on indices, it is exactly invertible:
`unpatchify(patchify(x)) == x` **bit-for-bit**, not merely approximately
(`test_patching.py::TestUnpatchify::test_round_trip_is_exact`).

### 2.3 Implementation: `unfold`, `permute`, `contiguous`

```python
patches = images.unfold(2, P, P).unfold(3, P, P)   # (B, C, g_h, g_w, P, P) — a view
patches = patches.permute(0, 2, 3, 1, 4, 5)        # (B, g_h, g_w, C, P, P) — still a view
patches = patches.contiguous().view(B, N, C * P * P)
```

`Tensor.unfold(dim, size, step)` exposes sliding windows by *changing strides*,
not copying memory. Two applications (one per spatial axis) produce all
$P \times P$ blocks.

The `permute` step is load-bearing. After unfolding, the channel axis sits
*outside* the patch-grid axes. Flattening directly from
$(C, g_h, g_w, P, P)$ — which is what the original PoC did — produces rows
that each contain $P^2$ pixels **from a single channel spanning several
different patches**: not patches at all. Moving $C$ inside the grid axes,
$(g_h, g_w, C, P, P)$, makes each row of the final `view` one complete patch.

`.contiguous()` is required because `view` only reinterprets memory that is
already laid out row-major in logical order; after `unfold` + `permute` the
logical and physical layouts disagree, so we materialize one copy.

---

## 3. Linear projection

### 3.1 Definition and parameter count

Each flattened patch $x \in \mathbb{R}^{CP^2}$ is mapped to the model
dimension by an affine map shared across all patches and all images:

$$
z = W x + b, \qquad W \in \mathbb{R}^{d \times CP^2},\quad b \in \mathbb{R}^d .
$$

(`nn.Linear` stores $W$ as `weight` of shape `(out, in)` and computes
$xW^\top + b$; same map, transposed convention.) Parameter count:

$$
\underbrace{d \cdot CP^2}_{\text{weight}} + \underbrace{d}_{\text{bias}}
\;\overset{\text{ViT-B/16}}{=}\; 768 \cdot 768 + 768 = 590{,}592 .
$$

$W$ and $b$ are **learnable** — they are how the model discovers which pixel
combinations within a patch are worth representing. This is why they must
live on an `nn.Module` (registered in `model.parameters()`, updated by the
optimizer), not on a `Dataset` as in the original PoC, where no optimizer
would ever see them
(`test_embedding.py::TestParameters::test_learnable_parameters_registered`,
`test_gradients_flow_to_all_parameters`).

### 3.2 Equivalence to a strided convolution

Production ViT implementations write the patch embedding as
`nn.Conv2d(C, d, kernel_size=P, stride=P)`. The two forms are *identical*,
not merely similar. A convolution with kernel size $=$ stride $= P$ evaluates,
at output location $(r, c)$ and output channel $m$:

$$
\mathrm{Conv}(x)[m, r, c]
= b_m + \sum_{\kappa=0}^{C-1}\sum_{u=0}^{P-1}\sum_{v=0}^{P-1}
K[m, \kappa, u, v]\; x[\kappa,\, rP+u,\, cP+v].
$$

Because the stride equals the kernel size, the receptive fields tile the image
without overlap — they are exactly the patches of section 2. The triple sum is
a dot product between kernel $m$ flattened in $(C, P, P)$ order and the patch
at $(r, c)$ flattened in the *same* order (this is why `patchify` uses that
layout). Hence with $K = \mathrm{reshape}(W,\ (d, C, P, P))$:

$$
\mathrm{Conv}(x)[m, r, c] \;=\; (W\,x_{\text{patch}(r,c)} + b)_m
\;=\; z_{\text{patch}(r,c)}[m].
$$

`test_embedding.py::TestForwardSemantics::test_projection_equals_strided_convolution`
constructs both modules with shared weights and asserts elementwise agreement.

---

## 4. The class token

A single learnable vector $x_{\text{cls}} \in \mathbb{R}^d$ is prepended to
every sequence:

$$
Z_0 = \big[\, x_{\text{cls}};\; z_1;\; \dots;\; z_N \,\big] \in \mathbb{R}^{(N+1) \times d}.
$$

It carries no image content. Its purpose is architectural: after $L$ layers
of self-attention, position 0 has attended to every patch, so its final state
$Z_L[0]$ serves as a *summary of the whole image* for the classification head
— a learned aggregation query, inherited from BERT's `[CLS]`. The *same*
vector (same memory, same gradients) is broadcast to every image in the batch
via `expand`.

Initialization is $\mathcal{N}(0, 0.02^2)$ truncated to $\pm 2\sigma$
(the BERT/ViT convention): near zero so it starts uninformative, nonzero so
its gradient signal is not degenerate at step 0.

---

## 5. Sinusoidal positional encodings

### 5.1 Definition

For position $pos \in \{0, \dots, N\}$ (position 0 is the class token) and
*pair index* $j \in \{0, \dots, d/2 - 1\}$:

$$
\boxed{\;
\omega_j = \theta^{-2j/d}, \qquad
PE_{(pos,\, 2j)} = \sin(\omega_j\, pos), \qquad
PE_{(pos,\, 2j+1)} = \cos(\omega_j\, pos)
\;}
$$

with base $\theta = 10{,}000$. Each of the $d/2$ dimension *pairs* is a
clock hand rotating at its own angular frequency $\omega_j$; a position is
encoded by the simultaneous reading of all $d/2$ clocks.
(`test_positional.py::TestSinusoidalEncoding::test_matches_paper_formula_entrywise`
checks every entry against this formula evaluated scalar by scalar.)

> **Reading the paper's notation.** Vaswani et al. write
> $PE_{(pos,2i)} = \sin\!\big(pos / 10000^{2i/d}\big)$. The "$2i$" in the
> *exponent* and the "$2i$" in the *column index* are the same number: the
> sine in even column $2i$ and the cosine in odd column $2i+1$ share one
> frequency. The exponent ranges over even integers $0, 2, 4, \dots$ not
> because odd dimensions are skipped from some smoother scheme, but because
> sine and cosine come in pairs — $d$ columns hold only $d/2$ distinct
> frequencies.

### 5.2 The frequency spectrum is geometric

The frequencies form a geometric progression with ratio $\theta^{-2/d}$:

$$
\frac{\omega_{j+1}}{\omega_j} = \theta^{-2/d} \quad\text{(constant)},
\qquad
\omega_0 = 1 \;\searrow\; \omega_{d/2-1} = \theta^{-(d-2)/d} \approx \tfrac{1}{\theta}.
$$

Equivalently the wavelengths $\lambda_j = 2\pi/\omega_j$ rise geometrically
from $2\pi$ positions to $\approx 2\pi\theta \approx 62{,}832$ positions
(`test_positional.py::TestFrequencyBands::test_geometric_progression`,
`test_endpoints`; visualized in `docs/figures/frequency_bands.png`).

This is the positional analogue of a multi-resolution number system: fast
clocks ($j$ small) distinguish neighboring positions but alias quickly; slow
clocks ($j$ large) barely move between neighbors but never repeat within any
realistic sequence. Compare binary counting, where bit $k$ flips with period
$2^{k+1}$ — here the "digits" are continuous and the radix is $\theta^{2/d}$.

### 5.3 The two `div_term` implementations are identical

The PoC's founding question. Most codebases compute the frequencies as

```python
div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(theta) / d))
```

while this repo writes the direct form `theta ** (-arange(0, d, 2) / d)`.
These are equal by the identity $a^x = e^{x \ln a}$:

$$
\exp\!\Big(2j \cdot \frac{-\ln\theta}{d}\Big)
= \exp\!\Big(\ln\theta \cdot \frac{-2j}{d}\Big)
= \theta^{-2j/d} = \omega_j .
$$

The historical preference for `exp`/`log` is numerical folklore (and avoids
recomputing a float power per element); on float32 the two agree to relative
error $\sim 10^{-7}$
(`test_positional.py::TestFrequencyBands::test_matches_exp_log_form`,
upgraded from the original `test.py`).

### 5.4 Boundedness and exact norm

Every entry lies in $[-1, 1]$, and by the Pythagorean identity applied to
each pair,

$$
\|PE_{pos}\|^2 = \sum_{j=0}^{d/2-1}\Big(\sin^2(\omega_j pos) + \cos^2(\omega_j pos)\Big)
= \frac{d}{2}
\qquad\text{for every } pos .
$$

So all position vectors live on a sphere of radius $\sqrt{d/2}$ — the
positional signal has *constant energy*, never drowning out the patch
embedding it is added to, regardless of position
(`test_positional.py::TestSinusoidalEncoding::test_exact_norm`).

### 5.5 Dot products depend only on the offset

Using $\sin A \sin B + \cos A \cos B = \cos(A - B)$ pairwise:

$$
\langle PE_p,\, PE_q \rangle
= \sum_{j=0}^{d/2-1} \Big( \sin(\omega_j p)\sin(\omega_j q) + \cos(\omega_j p)\cos(\omega_j q) \Big)
= \sum_{j=0}^{d/2-1} \cos\big(\omega_j (p - q)\big).
$$

The right side is a function of $p - q$ alone — the Gram matrix is
**Toeplitz** (constant along diagonals), symmetric in the offset, and maximal
at $p = q$ where it equals $d/2$. This is the diagonal band you see in
`docs/figures/position_similarity.png`, and it is the raw material from which
attention heads can learn "attend to nearby patches"
(`test_dot_product_depends_only_on_offset`, `test_dot_product_closed_form`).

### 5.6 Relative positions are linear maps (the key design property)

**Claim.** For every fixed offset $k$ there exists a matrix $R_k$,
*independent of* $pos$, such that

$$
PE_{pos+k} = R_k \, PE_{pos} \qquad \text{for all } pos .
$$

**Proof.** Work one frequency pair at a time. The angle-addition formulas give

$$
\begin{pmatrix} \sin(\omega_j(pos+k)) \\ \cos(\omega_j(pos+k)) \end{pmatrix}
=
\underbrace{\begin{pmatrix} \cos(\omega_j k) & \sin(\omega_j k) \\ -\sin(\omega_j k) & \cos(\omega_j k) \end{pmatrix}}_{R_k^{(j)}\ \text{— a rotation by } \omega_j k}
\begin{pmatrix} \sin(\omega_j\, pos) \\ \cos(\omega_j\, pos) \end{pmatrix}.
$$

(First row: $\sin(a+b) = \sin a\cos b + \cos a \sin b$ with $a = \omega_j pos$,
$b = \omega_j k$; second row: $\cos(a+b) = \cos a\cos b - \sin a\sin b$.)
Stacking the $d/2$ pairs, $R_k = \mathrm{diag}\big(R_k^{(0)}, \dots,
R_k^{(d/2-1)}\big)$ is block-diagonal with $2 \times 2$ rotation blocks, and
no entry involves $pos$. $\blacksquare$

This is *why* the encoding is built from interleaved sines and cosines rather
than, say, a single sinusoid or a linear ramp: it makes "shift by $k$" a
**fixed linear transformation** of the encoding, so a learned attention weight
matrix can express relative-position logic ("the patch one row up") uniformly
across the whole image. Each block being a rotation also re-proves section
5.4: rotations preserve norms.
(`test_positional.py::TestSinusoidalEncoding::test_relative_position_is_linear`
constructs $R_k$ explicitly and verifies the identity over the whole table.
Section 10 turns this rotation structure into a positional mechanism of its
own — RoPE applies these same blocks *multiplicatively* inside attention.)

### 5.7 Uniqueness and the interleaving convention

Within any sequence much shorter than the longest wavelength
($\approx 2\pi\theta$ positions), no two positions share an encoding — the
slowest clock hasn't completed a revolution, so the combined reading is
injective (`test_rows_unique` verifies a minimum pairwise distance).

Finally: whether sin/cos are *interleaved* $(s_0, c_0, s_1, c_1, \dots)$ as
here, or *concatenated* $(s_0, \dots, s_{d/2-1}, c_0, \dots, c_{d/2-1})$ as in
some implementations, is immaterial — the two differ by a fixed permutation
of coordinates, which the subsequent learned linear maps absorb. What matters
is consistency.

---

## 6. 2-D positional encodings

The 1-D encoding indexes patches by raster position, so the last patch of row
$r$ and the first patch of row $r+1$ get *adjacent* encodings despite being on
opposite sides of the image, while vertical neighbors ($k = \pm g_w$) appear
$g_w$ apart. The information to recover 2-D geometry is still present (offset
$g_w$ *is* "one row down" — section 5.6 makes it a linear map), but a
factorized 2-D encoding makes it explicit. Split the budget in half and
encode row and column independently:

$$
PE^{2D}_{(r, c)} = \Big[\; PE^{(d/2)}_r \;\big\Vert\; PE^{(d/2)}_c \;\Big] \in \mathbb{R}^d .
$$

Then two patches in the same row agree on their entire first half, two patches
in the same column agree on their second half, and dot products decompose as

$$
\big\langle PE^{2D}_{(r,c)},\, PE^{2D}_{(r',c')} \big\rangle
= f(r - r') + f(c - c'),
$$

with $f$ the 1-D offset kernel of section 5.5 — similarity now decays with
*Manhattan-style* separation rather than raster distance. This is the scheme
used by DETR and MAE. The class token, having no spatial location, receives
the zero vector
(`test_positional.py::TestSinusoidal2D`,
`test_embedding.py::TestPositionalVariants`).

The ViT paper itself ultimately used *learnable* positions
(`positional_encoding="learnable"`), reporting little difference from
sinusoidal in accuracy; sinusoidal remains the zero-parameter,
any-length-generalizing baseline.

---

## 7. Xavier initialization

The projection $z = Wx + b$ is initialized with **Xavier/Glorot uniform**:

$$
W_{mi} \sim \mathcal{U}\left(-\sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}},\; +\sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}}\right),
\qquad n_{\text{in}} = CP^2,\; n_{\text{out}} = d .
$$

**Where this comes from.** Assume inputs $x_i$ and weights $W_{mi}$ are
independent, zero-mean. Each output coordinate $z_m = \sum_{i=1}^{n_{\text{in}}} W_{mi} x_i$
then has variance

$$
\mathrm{Var}(z_m) = n_{\text{in}}\, \mathrm{Var}(W)\, \mathrm{Var}(x).
$$

Keeping the *forward* signal scale constant requires
$\mathrm{Var}(W) = 1/n_{\text{in}}$; the same argument on the backward pass
(gradients flow through $W^\top$, fan $n_{\text{out}}$) requires
$\mathrm{Var}(W) = 1/n_{\text{out}}$. Xavier takes the harmonic compromise

$$
\mathrm{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} ,
$$

and since a uniform $\mathcal{U}(-a, a)$ has variance $a^2/3$, solving
$a^2/3 = 2/(n_{\text{in}}+n_{\text{out}})$ gives the bound
$a = \sqrt{6/(n_{\text{in}}+n_{\text{out}})}$ above. Without such scaling,
$\mathrm{Var}(z)$ is multiplied by $n\,\mathrm{Var}(W)$ at every layer and
activations/gradients grow or shrink **geometrically** with depth.

The derivation assumes a roughly linear (or symmetric, e.g. tanh) regime
around 0; for ReLU networks the dead half-axis halves the variance and
**Kaiming** initialization ($\mathrm{Var}(W) = 2/n_{\text{in}}$) is the
right correction. The patch projection has no nonlinearity, so Xavier's
assumptions hold exactly here.

The bias is initialized to zero: it contributes no variance, and any nonzero
choice would inject an arbitrary constant preference before training has seen
data (`test_embedding.py::TestParameters::test_bias_starts_at_zero`).

---

## 8. Input normalization

Pixels are mapped from $[0, 255]$ to $[0,1]$ and then standardized per channel:

$$
\hat{x}_{\kappa h w} = \frac{x_{\kappa h w} - \mu_\kappa}{\sigma_\kappa},
\qquad
\mu = (0.485,\, 0.456,\, 0.406),\quad
\sigma = (0.229,\, 0.224,\, 0.225),
$$

the channel statistics of ImageNet. After this, each channel of a "typical"
natural image has approximately zero mean and unit variance — which is
precisely the $\mathrm{Var}(x) \approx 1$ assumption the Xavier derivation in
section 7 starts from. Initialization and normalization are two halves of one
contract: *unit-scale signals at the boundary*.

`Normalize.denormalize` inverts the affine map exactly
($x = \hat{x}\sigma + \mu$;
`test_dataset.py::TestNormalize::test_denormalize_round_trip`), which the
visualization utilities use to display tensors as images.

---

## 9. Scaled dot-product attention

What consumes the tokens. Implemented from scratch in
[`attention.py`](../vit_tokenizer/attention.py) and pinned against
`torch.nn.MultiheadAttention` elementwise
(`tests/test_attention.py::TestEquivalenceWithTorch`).

### 9.1 Definition

Given tokens $Z \in \mathbb{R}^{S \times d}$ (here $S = N + 1$), three learned
projections produce queries, keys, and values, and each token's output is a
score-weighted mixture of all value vectors:

$$
Q = Z W_Q^\top,\quad K = Z W_K^\top,\quad V = Z W_V^\top, \qquad
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_h}}\Big)\, V .
$$

The softmax is applied row-wise, so row $m$ of the attention matrix is a
probability distribution over the $S$ tokens — "where token $m$ looks." Rows
summing to 1 is asserted in
`test_attention_weights_shape_and_rows_sum_to_one`.

### 9.2 Why divide by $\sqrt{d_h}$

Suppose the entries of $q, k \in \mathbb{R}^{d_h}$ are independent with zero
mean and unit variance (which sections 7–8 work to make approximately true).
Then

$$
\mathrm{Var}(q \cdot k) = \mathrm{Var}\!\Big(\sum_{i=1}^{d_h} q_i k_i\Big)
= \sum_{i=1}^{d_h} \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = d_h ,
$$

so raw scores have standard deviation $\sqrt{d_h}$ — at $d_h = 256$, typical
scores of magnitude $\pm 16$. Pushed through a softmax, such logits produce
nearly one-hot rows, and the softmax Jacobian
$\mathrm{diag}(p) - p p^\top$ vanishes as $p$ approaches a vertex of the
simplex: **saturated attention stops learning**. Dividing by $\sqrt{d_h}$
restores unit variance. Both halves of this argument are checked
*statistically* in `tests/test_attention.py::TestScaling`: scaled scores have
sample variance $\approx 1$, and at $d_h = 256$ the unscaled softmax rows
place $> 90\%$ of their mass on a single entry while the scaled rows stay
spread out.

### 9.3 Multi-head decomposition

Rather than one attention over $d$ dimensions, split into $H$ heads of
$d_h = d/H$ dimensions, run them in parallel, concatenate, and mix:

$$
\mathrm{MHSA}(Z) = \big[\mathrm{head}_1 \,\Vert\, \cdots \,\Vert\, \mathrm{head}_H\big] W_O^\top + b_O .
$$

Each head can realize a different attention pattern (one tracks the shape,
another the background — visible in `docs/figures/attention_maps.png`).
Parameter count: four $d \times d$ projections plus biases, $4(d^2 + d)$,
*independent of* $H$ — heads partition the width, they don't add capacity.

### 9.4 Permutation equivariance, proved

Section 1 claimed attention has no notion of position. Precisely: for any
permutation matrix $\Pi$,

$$
\mathrm{MHSA}(\Pi Z) = \Pi\, \mathrm{MHSA}(Z).
$$

*Proof.* The projections are per-token: $(\Pi Z) W^\top = \Pi (Z W^\top)$, so
$Q' = \Pi Q$, $K' = \Pi K$, $V' = \Pi V$. The scores become
$Q' K'^\top = \Pi Q K^\top \Pi^\top$. Softmax acts row-wise and a
permutation only relabels rows and columns, so
$\mathrm{softmax}(\Pi A \Pi^\top) = \Pi\, \mathrm{softmax}(A)\, \Pi^\top$.
Then the output is
$\Pi\, \mathrm{softmax}(A)\, \Pi^\top \Pi V = \Pi\, \mathrm{softmax}(A)\, V$,
since $\Pi^\top \Pi = I$. The output projection is again per-token. $\blacksquare$

`tests/test_attention.py::TestPermutationEquivariance` runs this proof
numerically — and verifies that turning RoPE on *breaks* it, which is the
entire point of positional information.

---

## 10. Rotary positional embeddings (RoPE)

Section 5.6 proved that shifting a sinusoidal encoding by $k$ positions is a
fixed block-diagonal rotation $R_k$. RoPE (Su et al. 2021) turns that
observation into the mechanism itself: instead of *adding* a position vector
to the token once at the input, **rotate the queries and keys inside every
attention layer**, pair $j$ of the token at position $m$ by angle
$\omega_j m$, with the same frequencies $\omega_j = \theta^{-2j/d_h}$ as
section 5:

$$
q^{(m)} = R(m)\, q, \qquad k^{(n)} = R(n)\, k,
\qquad
R(m) = \mathrm{diag}\big(R^{(0)}(m), \dots\big),\quad
R^{(j)}(m) = \begin{pmatrix} \cos \omega_j m & -\sin \omega_j m \\ \sin \omega_j m & \cos \omega_j m \end{pmatrix}.
$$

### 10.1 The relative-position theorem

**Claim.** The attention score between a query at position $m$ and a key at
position $n$ depends only on the offset $n - m$:

$$
\big\langle R(m)\, q,\; R(n)\, k \big\rangle = \big\langle q,\; R(n - m)\, k \big\rangle .
$$

*Proof.* Per $2\times 2$ block, plane rotations satisfy
$R^{(j)}(m)^\top = R^{(j)}(-m)$ and compose additively,
$R^{(j)}(a) R^{(j)}(b) = R^{(j)}(a + b)$. Hence

$$
\langle R(m) q, R(n) k \rangle
= q^\top R(m)^\top R(n)\, k
= q^\top R(n - m)\, k . \qquad \blacksquare
$$

Absolute position cancels *exactly* — not approximately, and without any
learned table. Verified in
`tests/test_positional.py::TestRope::test_scores_depend_only_on_relative_position`
by placing identical $q, k$ at shifted positions and asserting equal scores.

### 10.2 Properties

* **Isometry.** Rotations preserve norms: $\|R(m) x\| = \|x\|$ exactly, so
  RoPE never rescales the residual stream (`test_norm_preserved`).
* **Identity at the origin.** $R(0) = I$: the [CLS] token at position 0 is
  untouched (`test_position_zero_is_identity`).
* **Additive vs. multiplicative.** The sinusoidal table of section 5 *adds*
  absolute position to the token content once; attention must then learn to
  extract relative offsets through the linear-map property of 5.6. RoPE
  skips the extraction: relative position appears directly in the scores, at
  every layer. The cost is that position is only visible to attention —
  the MLPs never see it.

In `MiniViT(positional_encoding="rotary")`, the additive table is disabled
and every attention layer rotates its queries and keys
(`tests/test_model.py::test_rotary_disables_additive_table`).

---

## 11. From tokens to a working model: MiniViT

Sections 1–10 build and justify the tokenizer; this section closes the loop
with the smallest model that can *demonstrate* the tokens are trainable
([`model.py`](../vit_tokenizer/model.py)).

### 11.1 The pre-norm encoder block

$$
\begin{aligned}
Z' &= Z + \mathrm{MHSA}\big(\mathrm{LN}(Z)\big) \\
Z'' &= Z' + \mathrm{MLP}\big(\mathrm{LN}(Z')\big),
\qquad \mathrm{MLP}(x) = W_2\, \mathrm{GELU}(W_1 x + b_1) + b_2
\end{aligned}
$$

with $W_1 \in \mathbb{R}^{4d \times d}$, $W_2 \in \mathbb{R}^{d \times 4d}$
(the conventional expansion ratio of 4). The *pre*-norm placement — LayerNorm
on the branch, never on the residual stream — means the identity path from
the loss to any layer is exactly the identity:

$$
\frac{\partial Z''}{\partial Z} = I + (\text{branch terms}),
$$

so gradients reach early layers undiminished regardless of depth. The
original post-norm transformer, $\mathrm{LN}(Z + \mathrm{Sublayer}(Z))$,
rescales the stream at every block and needs learning-rate warmup to train;
pre-norm (ViT's choice) does not.
`tests/test_model.py::TestEncoderBlock::test_residual_path_exists` makes the
identity path tangible: zero every weight and the block *is* the identity.

### 11.2 Readout

After $L$ blocks, the [CLS] state is normalized and classified:

$$
\hat{y} = W_{\text{head}}\; \mathrm{LN}\big(Z_L[0]\big) + b_{\text{head}} \in \mathbb{R}^{\text{classes}} .
$$

By the equivariance of section 9.4, position 0 is only special because the
positional encoding *made* it special — one more reason the tokenizer's
encoding step is load-bearing.

### 11.3 Parameter count

Per block: MHSA $4(d^2 + d)$, MLP $(4d^2 + 4d) + (4d^2 + d) = 8d^2 + 5d$, two
LayerNorms $4d$ — about $12d^2$ for $d \gg 1$. The full model:

$$
\#\text{params} \approx \underbrace{12\, d^2 L}_{\text{blocks}}
+ \underbrace{d\,(CP^2 + 1)}_{\text{tokenizer (§3)}}
+ \underbrace{(N + 2)\, d}_{\text{[CLS] + learnable positions}}
+ \underbrace{d \cdot \text{classes}}_{\text{head}} .
$$

Sanity check at ViT-Base ($d = 768$, $L = 12$, $P = 16$, 1000 classes):
$12 \cdot 12 \cdot 768^2 \approx 84.9\mathrm{M}$, plus $0.59\mathrm{M}$
(tokenizer) $+ 0.15\mathrm{M}$ (positions) $+ 0.77\mathrm{M}$ (head)
$\approx 86\mathrm{M}$ — the published figure.
`tests/test_model.py::test_vit_base_parameter_count` instantiates exactly
this configuration and counts.

### 11.4 Evidence that it learns

Two levels of proof, both runnable:

* **Overfit test** (`tests/test_model.py::test_overfits_tiny_batch`): 150
  Adam steps on 8 fixed samples must drive the cross-entropy below 0.05 and
  classify all 8 correctly. A break anywhere in the gradient path — the PoC's
  parameters-on-a-`Dataset` bug, an in-place op on the graph, a frozen buffer
  that should be a parameter — fails this test.
* **Generalization demo** (`examples/train_shapes.py`): a 355k-parameter
  MiniViT reaches $\approx 96\%$ *held-out* accuracy on the synthetic shapes
  task in about a minute on CPU. The attention overlays it produces
  (`docs/figures/attention_maps.png`) show [CLS] attending to the shape —
  the token pipeline carries the spatial information end to end.

---

## 12. References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — § 3.5 defines the sinusoidal encoding. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021 — the ViT paper. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
3. Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010 — Xavier initialization.
4. He et al., *Delving Deep into Rectifiers*, ICCV 2015 — Kaiming initialization.
5. Carion et al., *End-to-End Object Detection with Transformers (DETR)*, ECCV 2020 — factorized 2-D sinusoidal encodings.
6. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 — the rotation structure of § 5.6, applied multiplicatively. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
7. Kazemnejad, *Transformer Architecture: The Positional Encoding* — the blog post that seeded this project. [link](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
8. Karpathy, *Let's build the GPT Tokenizer* — the video that motivated the original PoC. [YouTube](https://youtu.be/zduSFxRajkE)
