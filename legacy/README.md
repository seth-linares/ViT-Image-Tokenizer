# Legacy PoC scripts

These are the original proof-of-concept files, preserved unchanged for
history. They contain the ideas the current package grew from — and a number
of bugs found while legitimizing it. Each is listed here with the fix and the
test that now guards against regression.

## `1_TORCH_TOKENIZER.py` (PyTorch version)

1. **`__getitem__` crashed unconditionally.**
   `torch.cat([self.class_token_vector.unsqueeze(0), projected_patches], dim=0)`
   concatenates a `(1, 1, d)` tensor with a `(N, d)` tensor — different ranks,
   so `torch.cat` raises. Even past that, the in-place
   `patches_with_token += positional_encodings.unsqueeze(0)` tries to grow a
   `(N+1, d)` tensor to `(1, N+1, d)` in place, which PyTorch forbids.
   *Fixed in* `ViTPatchEmbedding.forward` (shapes verified by
   `tests/test_embedding.py::TestShapes`).

2. **Crash in the most common configuration.** The constructor only assigned
   `self.image_dimensions` in the two *mismatch* branches (image smaller than
   a patch, or not divisible). With the defaults — `224 % 32 == 0` — neither
   branch ran, the attribute was never set, and
   `transforms.Resize(self.image_dimensions, ...)` raised `AttributeError`.
   *Fixed:* the new code validates divisibility explicitly and raises a clear
   `ValueError` telling you to resize
   (`tests/test_patching.py::TestPatchGridShape`).

3. **The patch flattening mixed pixels across patches.**
   `unfold` produces shape `(C, g_h, g_w, P, P)`; calling
   `.view(-1, P*P*3)` directly on that flattens with the *channel* axis
   outermost, so each output row contained $P^2$ pixels from a single channel
   spanning several different patches — not a patch. The channel axis must be
   permuted inside the grid axes first. *Fixed in* `patching.patchify`
   (`tests/test_patching.py::TestPatchify::test_matches_naive_crop_loop`
   compares against an explicit crop loop).

4. **Learnable parameters lived on a `Dataset`.** `nn.Parameter`s registered
   on a `Dataset` never appear in any model's `.parameters()`, so no optimizer
   updates them — and `DataLoader` worker processes would each hold a private
   copy whose gradients are discarded. The projection, bias, and class token
   are *model* weights; they now live on `ViTPatchEmbedding(nn.Module)`
   (`tests/test_embedding.py::TestParameters::test_learnable_parameters_registered`).

5. **`num_patches` used the unadjusted image size.** When the constructor
   shrank the image to make it divisible, `num_patches` was still computed
   from the original `image_dimension`, desynchronizing the positional table
   from the actual patch count.

## `2_GENERAL_TOKENIZER.py` (NumPy version)

6. **Patches were emitted in column-major order** (the outer loop walked
   `i` over the *width*), while everything else — and every reference
   implementation — assumes raster (row-major) order. Harmless only if
   nothing ever assumes an ordering; the positional encoding does.
   *Fixed:* `patchify` is raster-order by construction and tested.

7. **Patch pixels were never scaled:** `np.array(patch)` yields uint8 in
   `[0, 255]`, which was matrix-multiplied directly against a unit-variance
   random projection — outputs of magnitude ~10³ next to positional encodings
   in `[-1, 1]`. *Fixed:* `to_tensor` + `Normalize`
   (`tests/test_dataset.py`).

8. **Positional indexing was inconsistent with the Torch version:** here
   patch *k* received encoding row *k* and the class token received none,
   whereas the Torch version gave the class token position 0 and patches
   1..N. The package standardizes on the latter, with the convention
   documented.

## `test.py`

The one test of the PoC — equivalence of the readable `div_term` and the
conventional `exp`/`log` form — was sound, and survives expanded as
`tests/test_positional.py::TestFrequencyBands::test_matches_exp_log_form`,
with the algebraic identity written out in `docs/MATH.md` § 5.3.
