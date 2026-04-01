# Accelerating GatedDeltaNet Inference by 1.15x

by [Eric Alcaide](/)

This blogpost describes how we accelerated [Gated DeltaNet](https://arxiv.org/abs/2412.06464) by **1.15x** in the forward pass following [Simon Veitner's blogpost](https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/). Here's our [PR](https://github.com/fla-org/flash-linear-attention/pull/797). Once it's merged, practitioners will get the speedup by upgrading [FLA](https://github.com/fla-org/flash-linear-attention) version.

<p align="center">
  <img src="/imgs/gdn_fused_comparison.svg" alt="FLA vs Tricked + Fused Kernels" width="900">
</p>

**Figure 1**: Execution time comparison of [FLA](https://github.com/fla-org/flash-linear-attention) (commit: [f52529e](https://github.com/fla-org/flash-linear-attention/commit/f52529ee8a7b17f4514fd32dbe22632ed9d55c76)) and our improved version, layered on top of FLA's own fused kernels. Benchmarked on NVIDIA H100.

**Table of contents**
1. [What is GDN?](#what-is-gdn)
2. [The trick](#the-trick)
3. [Kernel fusion](#kernel-fusion)
4. [Results](#results)
5. [References](#references)

## What is GDN?

[Gated DeltaNet](https://arxiv.org/abs/2412.06464) (ICLR 2025) is a linear attention mechanism that combines the delta rule with scalar gating in the state transition.
DeltaNet was proposed as an alternative to transformers in 2021 ([Schlag et al., ICML 2021](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf)) and scaled to hardware efficient training in 2024 ([Yang et al., NeurIPS 2024](https://openreview.net/forum?id=y8Rm4VNRPH)). Gated DeltaNet introduces an additional decay, proven to be effective in sequence models ([RetNet](https://arxiv.org/abs/2307.08621), [RWKV6](https://openreview.net/forum?id=soz1SEiPeq), [Mamba2](https://openreview.net/forum?id=ztn8FCR1td)). The state update is:

$$S_t = \alpha_t \left(I - \beta_t \, k_t k_t^\top \right) S_{t-1} + \beta_t \, v_t k_t^\top \quad \in \mathbb{R}^{D_k \times D_v}$$

where $\alpha_t = \exp(g_t)$ is a scalar decay gate and $\beta_t$ is a scalar learning rate. The Householder-like term $(I - \beta_t \, k_t k_t^\top)$ performs a rank-1 correction to the memory state, and the gate controls how much of the old state to retain. The output is then simply $o_t = q_t^\top S_t \in \mathbb{R}^{D_v}$.

<details>
<summary><b>Notation primer</b></summary>

- `kkt`: the outer product $K K^\top$, i.e. the coupling matrix between tokens within a chunk
- `solve`: computing $(I + A)^{-1}$, the inverse of the coupling matrix (done via forward substitution + block merge)
- `WY`: the [WY representation](https://epubs.siam.org/doi/10.1137/0908009) for products of Householder matrices, which enables efficient chunkwise-parallel computation of DeltaNet (see [Appendix B.1, Yang et al.](https://openreview.net/forum?id=y8Rm4VNRPH)). Produces correction vectors `w` (key-path) and `u` (value-path)
- `BT`: chunk size (default 64 tokens). The chunked algorithm processes `BT` tokens at a time
- `BC`: sub-chunk size (default 16). The fused kkt+solve kernel splits each `BT`-chunk into `BT/BC = 4` sub-chunks for the hierarchical solve
- `g_t`: the decay gate in **log space** (output of a `logsigmoid(linear)`). The actual decay is $\alpha_t = \exp(g_t)$, kept in log space for numerical stability (cumulative products become cumulative sums: $ g^{\mathrm{cum}}_t = \sum_{i=1}^{t} g_i $ )

</details>

GDN has quickly become a practical building block for production LLMs from big labs. Alibaba's [Qwen3Next](https://huggingface.co/collections/Qwen/qwen3-next) and [Qwen3.5](https://github.com/QwenLM/Qwen3.5) family uses Gated DeltaNet in 75% of its layers (3:1 hybrid with standard attention). A reference implementation of a high performance, chunkwise-parallel kernel lives in [FLA repo](https://github.com/fla-org/flash-linear-attention), which provides highly optimized Triton kernels for the chunked forward and backward passes of GDN and other alternatives to attention.

## The trick

This blogpost describes the implementation of the optimization proposed by **[Simon Veitner in this blog post](https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/) (we refer the reader there for details on the algorithm)** which was also noted independently in [Comba](https://arxiv.org/abs/2506.02475).

In the chunked algorithm, two coupling matrices appear:

- $M = I + \mathrm{tril}(\beta \cdot K K^\top, -1)$ — without gating
- $N = I + \mathrm{tril}(\beta \cdot \Gamma \odot K K^\top, -1)$ — with gating, where $\Gamma_{ij} = \exp(g^{\mathrm{cum}}_i - g^{\mathrm{cum}}_j)$

FLA computes `N` directly (applying `BT^2` exp operations inside the coupling matrix) and then solves `N^{-1}`.

The key observation is that $N$ is a similarity transform of $M$: $N = GMG^{-1}$ where $G = \mathrm{diag}(\exp(g^{\mathrm{cum}}_1), \ldots, \exp(g^{\mathrm{cum}}_C))$. Therefore: $N^{-1} = GM^{-1}G^{-1}$.

This means we can:
1. Compute `M` (no gating -- skip `BT^2` exp ops)
2. Solve `M^{-1}` (same cost as before)
3. Apply `G` and `G^{-1}` as diagonal scaling in the WY step (only `2*BT` exp ops)

Net savings: `BT^2 - 2*BT` exp operations per chunk. With `BT=64`, that's **3968 fewer exp ops per chunk**.

## Kernel fusion

The math trick alone doesn't beat FLA. The original implementation used **separate** kernels for kkt, solve, and WY -- and the HBM round-trip for the intermediate `A` matrix wiped out the exp savings.

[FLA](https://github.com/fla-org/flash-linear-attention/pull/789) fuses `(kkt+solve)` into a single kernel (the `A` matrix stays in registers, never hits HBM). Our implementation layers the trick **on top of** FLA's existing fused kernels rather than reimplementing them:

**1. `(kkt+solve) + tricked WY`** (training path) -- we call FLA's own fused `(kkt+solve)` kernel with `g=None` (ungated), then run our custom WY kernel that applies G/G_inv scaling. The solved `A` is written to HBM for the backward pass.

**2. `(kkt+solve+WY)`** aka "fusemaxxed" (inference path) -- all three steps in a single kernel. The `A` matrix is computed, solved, AND consumed for the WY computation entirely in registers. Zero HBM traffic for `A`. Used **only** when `torch.is_grad_enabled() == False`.

### Training: `mem_efficient` mode

When gradients are enabled, the training path uses `(kkt+solve) + tricked WY` and saves `A` to HBM for the backward pass. Additionally, when `mem_efficient=False` (default) and `T > 2048`, the `w` and `u` tensors are cached during the forward pass so the backward skips their recomputation. Set `mem_efficient=True` to trade compute for memory on long sequences.

## Results

All benchmarks on NVIDIA H100, B=1, `H * Dh = 2048`, bf16. FLA baseline is [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (commit: [f52529e](https://github.com/fla-org/flash-linear-attention/commit/f52529ee8a7b17f4514fd32dbe22632ed9d55c76)). Geometric mean speedups computed across all sequence lengths (1K-128K).

- **`(kkt+solve+WY)` fusemaxxed** is **1.26-1.40x faster** at short sequences (`T <= 4K`) where kernel launch overhead and HBM traffic dominate. Geo mean: **1.11x** (Dh=128), **1.12x** (Dh=256).

- **`(kkt+solve) + tricked WY`** (training path) is **1.14x faster** at long sequences (`T >= 16K`) where the trick eliminates `BT^2` exp2 ops per chunk from the kkt computation. At short sequences, it runs at parity with FLA since the kkt+solve kernel is FLA's own. Geo mean: **1.13x** (Dh=128), **1.05x** (Dh=256).

- **Auto-dispatch** (inference): automatically picks fusemaxxed for `T <= 8K` and the fused path for `T > 8K`. Never regresses vs FLA. Geo mean: **1.21x** (Dh=128), **1.14x** (Dh=256).

- **Forward + Backward** (with `mem_efficient=False`): When `w` and `u` are cached during the forward pass (for `T >= 2048`), the backward skips their recomputation. Geo mean: **1.02x** (Dh=128), **1.02x** (Dh=256).

We expect this to be relevant for those running inference with OSS LLMs which incorporate GDN in [up to 75% of their layers](https://github.com/QwenLM/Qwen3.5).

<details>
<summary><b>Forward (no_grad) -- detailed tables</b></summary>

#### `Dh=128`, H=16

| SeqLen | FLA (ms) | (kkt+solve)+WY (ms) | speedup | (kkt+solve+WY) (ms) | speedup |
|-------:|---------:|--------------------:|--------:|--------------------:|--------:|
| 1K     | 0.452 | 0.393 | **1.15x** | 0.324 | **1.39x** |
| 2K     | 0.454 | 0.396 | **1.15x** | 0.324 | **1.40x** |
| 4K     | 0.459 | 0.419 | **1.10x** | 0.347 | **1.32x** |
| 8K     | 0.572 | 0.534 | **1.07x** | 0.529 | **1.08x** |
| 16K    | 0.899 | 0.790 | **1.14x** | 0.911 | 0.99x |
| 32K    | 1.611 | 1.416 | **1.14x** | 1.676 | 0.96x |
| 64K    | 3.014 | 2.648 | **1.14x** | 3.196 | 0.94x |
| 128K   | 5.844 | 5.128 | **1.14x** | 6.230 | 0.94x |

Geo mean: fused **1.13x**, fusemaxxed **1.11x**, best-of-both **1.21x**

#### `Dh=256`, H=8

| SeqLen | FLA (ms) | (kkt+solve)+WY (ms) | speedup | (kkt+solve+WY) (ms) | speedup |
|-------:|---------:|--------------------:|--------:|--------------------:|--------:|
| 1K     | 0.453 | 0.395 | **1.15x** | 0.323 | **1.40x** |
| 2K     | 0.460 | 0.403 | **1.14x** | 0.332 | **1.39x** |
| 4K     | 0.538 | 0.501 | **1.08x** | 0.428 | **1.26x** |
| 8K     | 0.717 | 0.687 | **1.04x** | 0.641 | **1.12x** |
| 16K    | 1.092 | 1.084 | 1.01x | 1.123 | 0.97x |
| 32K    | 1.974 | 1.971 | 1.00x | 2.071 | 0.95x |
| 64K    | 3.726 | 3.722 | 1.00x | 3.931 | 0.95x |
| 128K   | 7.201 | 7.177 | 1.00x | 7.624 | 0.94x |

Geo mean: fused **1.05x**, fusemaxxed **1.12x**, best-of-both **1.14x**

</details>

<details>
<summary><b>Forward + Backward -- detailed tables</b></summary>

#### `Dh=128`, H=16

| SeqLen | FLA (ms) | Tricked (ms) | speedup |
|-------:|---------:|-------------:|--------:|
| 1K     | 1.487 | 1.741 | 0.85x |
| 2K     | 1.807 | 1.696 | **1.07x** |
| 4K     | 1.835 | 1.698 | **1.08x** |
| 8K     | 2.121 | 2.049 | **1.04x** |
| 16K    | 3.272 | 3.157 | **1.04x** |
| 32K    | 5.840 | 5.629 | **1.04x** |
| 64K    | 10.992 | 10.524 | **1.04x** |
| 128K   | 21.373 | 20.393 | **1.05x** |

Geo mean: **1.02x**

#### `Dh=256`, H=8

| SeqLen | FLA (ms) | Tricked (ms) | speedup |
|-------:|---------:|-------------:|--------:|
| 1K     | 1.611 | 1.521 | **1.06x** |
| 2K     | 1.582 | 1.449 | **1.09x** |
| 4K     | 1.911 | 1.886 | 1.01x |
| 8K     | 2.858 | 2.820 | 1.01x |
| 16K    | 4.773 | 4.727 | 1.01x |
| 32K    | 8.783 | 8.762 | 1.00x |
| 64K    | 16.881 | 16.789 | 1.01x |
| 128K   | 33.133 | 32.878 | 1.01x |

Geo mean: **1.02x**

</details>

## References

1. Songlin Yang, Jan Kautz, Ali Hatamizadeh. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://openreview.net/forum?id=r8H7xhYPwz). ICLR 2025.
2. Simon Veitner. [Simple Math to Speed Up GDN Prefill](https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/). 2026.
3. Songlin Yang, Yu Zhang et al. [flash-linear-attention](https://github.com/fla-org/flash-linear-attention). Reference Triton kernels for GDN and other linear attention models.
4. Imanol Schlag et al. [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf). ICML 2021.
5. Songlin Yang et al. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://openreview.net/forum?id=y8Rm4VNRPH). NeurIPS 2024.
6. Yutao Sun et al. [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621). 2023.
7. Bo Peng et al. [Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence](https://openreview.net/forum?id=soz1SEiPeq). COLM 2024.
8. Tri Dao, Albert Gu. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://openreview.net/forum?id=ztn8FCR1td). ICML 2024.
9. Qwen Team. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). 2025. See also [Qwen3.5](https://github.com/QwenLM/Qwen3.5).
10. Jiaxi Hu, Yongqi Pan, Jusen Du et al. [Comba: Improving Bilinear RNNs with Closed-loop Control](https://arxiv.org/abs/2506.02475). 2025.

## Cite this

```bibtex
@misc{alcaide2026accelerating,
  title   = {Accelerating GatedDeltaNet Inference by 1.15x},
  author  = {Eric Alcaide},
  month   = {March},
  year    = {2026},
  url     = {https://hypnopump.github.io/post.html?slug=accelerating-gdn-inference}
}

@misc{veitner2026speedup,
  title   = {Simple Math to Speed Up GDN Prefill},
  author  = {Simon Veitner},
  month   = {March},
  year    = {2026},
  url     = {https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/}
}
```
