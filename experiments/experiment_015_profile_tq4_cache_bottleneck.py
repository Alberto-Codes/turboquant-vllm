r"""Experiment 015 -- Profile TQ4 cache bottleneck: compress/dequant vs Flash Attention.

P9 Phase 3c.7: Determines whether the bottleneck in the TQ4 vLLM backend is in
PyTorch compress/dequant operations or Flash Attention itself.

Decision matrix:
  - If compress/dequant dominates → implement Triton kernels (3c.8-3c.9)
  - If Flash Attention dominates → PyTorch path is acceptable, skip to 3d benchmark

Profiles three phases of TQ4 attention per decode step:
  1. Compress: K+V normalize → rotate → bucketize → nibble-pack (new tokens)
  2. Decompress: K+V unpack → centroid lookup → unrotate → scale (full cache)
  3. Attention: flash_attn_varlen_func on decompressed FP16 tensors

Examples:
    Default profiling with Molmo2-8B dimensions:

    ```bash
    uv run python experiments/experiment_015_profile_tq4_cache_bottleneck.py
    ```

    Custom cache lengths:

    ```bash
    uv run python experiments/experiment_015_profile_tq4_cache_bottleneck.py \
        --cache-lens 256,1024,4096
    ```

    More measurement iterations:

    ```bash
    uv run python experiments/experiment_015_profile_tq4_cache_bottleneck.py \
        --iters 200 --warmup 20
    ```

See Also:
    :mod:`turboquant_consumer.vllm.tq4_backend`: TQ4 attention backend.
    ``docs/ROADMAP.md``: Phase 3c.7.
    ``experiments/experiment_014_full_episode_benchmark.py``: Prior vLLM benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from turboquant_consumer.quantizer import TurboQuantMSE

# ---------------------------------------------------------------------------
# TQ4 constants (must match tq4_backend.py)
# ---------------------------------------------------------------------------
TQ4_BITS = 4
TQ4_SEED = 42
TQ4_NORM_BYTES = 4  # fp32

# Molmo2-8B defaults
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_NUM_QUERY_HEADS = 32
DEFAULT_CACHE_LENS = [128, 256, 512, 1024, 2048, 4096]


# ---------------------------------------------------------------------------
# VRAM helpers (same pattern as experiments 013/014)
# ---------------------------------------------------------------------------


def _get_vram_mb() -> float:
    """Return peak GPU memory in MiB.

    Returns:
        Peak VRAM usage in MiB, or 0.0 if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_vram() -> None:
    """Reset peak memory stats and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GPU timing
# ---------------------------------------------------------------------------


def _benchmark_op(
    fn: callable,
    warmup: int = 10,
    iters: int = 100,
) -> dict[str, float]:
    """Benchmark a GPU operation using CUDA events.

    Returns:
        Dict with median_ms, mean_ms, min_ms, max_ms, p95_ms.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    n = len(times)
    return {
        "median_ms": round(times[n // 2], 4),
        "mean_ms": round(sum(times) / n, 4),
        "min_ms": round(times[0], 4),
        "max_ms": round(times[-1], 4),
        "p95_ms": round(times[int(n * 0.95)], 4),
    }


# ---------------------------------------------------------------------------
# Standalone TQ4 operations (replicated from tq4_backend.py for isolation)
# ---------------------------------------------------------------------------


def _compress_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    rotation_t: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress K and V tensors to packed TQ4 uint8.

    Replicates the exact operations from ``TQ4AttentionImpl._compress``,
    applied to both K and V, without class instantiation overhead.

    Args:
        key: ``(N, H, D)`` float16.
        value: ``(N, H, D)`` float16.
        rotation_t: ``(D, D)`` float32 -- transposed rotation matrix.
        boundaries: ``(2**bits - 1,)`` float32 -- quantization boundaries.

    Returns:
        k_packed, k_norms, v_packed, v_norms
    """

    def _compress_one(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, H, D = x.shape
        flat = x.reshape(N * H, D).float()
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ rotation_t
        indices = torch.bucketize(rotated, boundaries)
        indices = indices.clamp(0, (1 << TQ4_BITS) - 1)
        idx_u8 = indices.to(torch.uint8)
        packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]
        return packed.reshape(N, H, D // 2), norms.reshape(N, H, 1)

    k_packed, k_norms = _compress_one(key)
    v_packed, v_norms = _compress_one(value)
    return k_packed, k_norms, v_packed, v_norms


def _decompress_kv(
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompress packed TQ4 K and V back to float.

    Replicates ``TQ4AttentionImpl._decompress`` for both K and V.

    Args:
        k_packed: ``(N, H, D//2)`` uint8.
        k_norms: ``(N, H, 1)`` float32.
        v_packed: ``(N, H, D//2)`` uint8.
        v_norms: ``(N, H, 1)`` float32.
        centroids: ``(2**bits,)`` float32 -- centroid values.
        rotation: ``(D, D)`` float32 -- rotation matrix.
        dtype: Output dtype.

    Returns:
        key, value -- both ``(N, H, D)`` in ``dtype``.
    """

    def _decompress_one(
        packed: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        N, H, half_D = packed.shape
        D = half_D * 2
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(N * H, D)
        flat_norms = norms.reshape(N * H, 1)
        reconstructed = centroids[indices]
        unrotated = reconstructed @ rotation
        result = unrotated * flat_norms
        return result.reshape(N, H, D).to(dtype)

    key = _decompress_one(k_packed, k_norms)
    value = _decompress_one(v_packed, v_norms)
    return key, value


# ---------------------------------------------------------------------------
# Profile scenarios
# ---------------------------------------------------------------------------


def _profile_decode_step(
    cache_len: int,
    head_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    rotation: torch.Tensor,
    rotation_t: torch.Tensor,
    boundaries: torch.Tensor,
    centroids: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Profile one decode step at a given cache depth.

    One decode step = compress(1 token K+V) + decompress(cache_len K+V)
    + attention(1 query vs cache_len KV).

    Returns:
        Dict with per-phase timing stats and percentage breakdown.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    # --- synthetic data ---
    new_k = torch.randn(
        1,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    new_v = torch.randn(
        1,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    # Pre-compressed cache (simulates already-stored data)
    cache_k_packed = torch.randint(
        0,
        255,
        (cache_len, num_kv_heads, head_dim // 2),
        device=device,
        dtype=torch.uint8,
    )
    cache_k_norms = (
        torch.randn(
            cache_len,
            num_kv_heads,
            1,
            device=device,
            dtype=torch.float32,
        )
        .abs_()
        .clamp_(min=0.1)
    )
    cache_v_packed = torch.randint(
        0,
        255,
        (cache_len, num_kv_heads, head_dim // 2),
        device=device,
        dtype=torch.uint8,
    )
    cache_v_norms = (
        torch.randn(
            cache_len,
            num_kv_heads,
            1,
            device=device,
            dtype=torch.float32,
        )
        .abs_()
        .clamp_(min=0.1)
    )

    query = torch.randn(
        1,
        num_query_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    # --- profile compress (1 token K+V) ---
    compress_stats = _benchmark_op(
        lambda: _compress_kv(new_k, new_v, rotation_t, boundaries),
        warmup,
        iters,
    )

    # --- profile decompress (full cache K+V) ---
    decompress_stats = _benchmark_op(
        lambda: _decompress_kv(
            cache_k_packed,
            cache_k_norms,
            cache_v_packed,
            cache_v_norms,
            centroids,
            rotation,
        ),
        warmup,
        iters,
    )

    # --- profile attention (1 query vs cache_len KV) ---
    # Pre-decompress for the attention benchmark (isolate attention cost)
    k_full, v_full = _decompress_kv(
        cache_k_packed,
        cache_k_norms,
        cache_v_packed,
        cache_v_norms,
        centroids,
        rotation,
    )
    cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, cache_len], device=device, dtype=torch.int32)

    attention_stats = _benchmark_op(
        lambda: flash_attn_varlen_func(
            q=query,
            k=k_full,
            v=v_full,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=cache_len,
            cu_seqlens_k=cu_k,
        ),
        warmup,
        iters,
    )

    # --- summary ---
    total_median = (
        compress_stats["median_ms"]
        + decompress_stats["median_ms"]
        + attention_stats["median_ms"]
    )

    return {
        "cache_len": cache_len,
        "compress": compress_stats,
        "decompress": decompress_stats,
        "attention": attention_stats,
        "total_median_ms": round(total_median, 4),
        "pct_compress": round(100 * compress_stats["median_ms"] / total_median, 1),
        "pct_decompress": round(
            100 * decompress_stats["median_ms"] / total_median,
            1,
        ),
        "pct_attention": round(
            100 * attention_stats["median_ms"] / total_median,
            1,
        ),
    }


def _profile_prefill(
    seq_len: int,
    head_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    rotation: torch.Tensor,
    rotation_t: torch.Tensor,
    boundaries: torch.Tensor,
    centroids: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Profile prefill: compress(seq_len K+V) + attention(seq_len vs seq_len).

    Returns:
        Dict with per-phase timing stats and percentage breakdown.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    key = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    value = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    query = torch.randn(
        seq_len,
        num_query_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    # --- profile compress (all tokens K+V) ---
    compress_stats = _benchmark_op(
        lambda: _compress_kv(key, value, rotation_t, boundaries),
        warmup,
        iters,
    )

    # --- profile attention (self-attention, causal) ---
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)

    attention_stats = _benchmark_op(
        lambda: flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            max_seqlen_q=seq_len,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=seq_len,
            cu_seqlens_k=cu_seqlens,
            causal=True,
        ),
        warmup,
        iters,
    )

    total_median = compress_stats["median_ms"] + attention_stats["median_ms"]

    return {
        "seq_len": seq_len,
        "compress": compress_stats,
        "attention": attention_stats,
        "total_median_ms": round(total_median, 4),
        "pct_compress": round(100 * compress_stats["median_ms"] / total_median, 1),
        "pct_attention": round(
            100 * attention_stats["median_ms"] / total_median,
            1,
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    head_dim: int = DEFAULT_HEAD_DIM,
    num_query_heads: int = DEFAULT_NUM_QUERY_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    cache_lens: list[int] | None = None,
    warmup: int = 10,
    iters: int = 100,
) -> dict[str, Any]:
    """Run the TQ4 cache bottleneck profiling experiment.

    Returns:
        Dict with decode/prefill profiling results, conclusion, and GPU info.
    """
    if cache_lens is None:
        cache_lens = list(DEFAULT_CACHE_LENS)

    device = torch.device("cuda")

    # --- Initialize TQ4 quantizer to get rotation/boundaries/centroids ---
    quantizer = TurboQuantMSE(head_dim, TQ4_BITS, seed=TQ4_SEED)
    rotation = quantizer.rotation.to(device)  # (D, D) fp32
    rotation_t = rotation.T.contiguous()
    boundaries = quantizer.codebook.boundaries.to(device)  # (15,) fp32
    centroids = quantizer.codebook.centroids.to(device)  # (16,) fp32

    gpu_name = torch.cuda.get_device_name(0)
    _reset_vram()

    results: dict[str, Any] = {
        "experiment": "015-profile-tq4-cache-bottleneck",
        "phase": "P9-3c.7",
        "gpu": gpu_name,
        "config": {
            "head_dim": head_dim,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "tq4_bits": TQ4_BITS,
            "warmup": warmup,
            "iters": iters,
        },
        "decode": [],
        "prefill": [],
    }

    # --- Decode profiling ---
    print(f"\n{'=' * 60}")
    print(f"TQ4 Cache Bottleneck Profiling -- {gpu_name}")
    print(f"Config: D={head_dim}, Qh={num_query_heads}, KVh={num_kv_heads}")
    print(f"Timing: {warmup} warmup + {iters} measured iterations")
    print(f"{'=' * 60}")

    print("\n--- Decode (1 new token per step) ---")
    header = (
        f"{'Cache':>8} {'Compress':>10} {'Decompress':>12} "
        f"{'Attention':>10} {'Total':>8} {'%C':>5} {'%D':>5} {'%A':>5}"
    )
    print(header)
    print("-" * len(header))

    for cache_len in cache_lens:
        r = _profile_decode_step(
            cache_len,
            head_dim,
            num_query_heads,
            num_kv_heads,
            rotation,
            rotation_t,
            boundaries,
            centroids,
            device,
            warmup,
            iters,
        )
        results["decode"].append(r)
        print(
            f"{cache_len:>8} "
            f"{r['compress']['median_ms']:>9.3f}ms "
            f"{r['decompress']['median_ms']:>11.3f}ms "
            f"{r['attention']['median_ms']:>9.3f}ms "
            f"{r['total_median_ms']:>7.3f}ms "
            f"{r['pct_compress']:>4.1f}% "
            f"{r['pct_decompress']:>4.1f}% "
            f"{r['pct_attention']:>4.1f}%"
        )

    # --- Prefill profiling ---
    print("\n--- Prefill (compress + self-attention, causal) ---")
    header_pf = (
        f"{'SeqLen':>8} {'Compress':>10} {'Attention':>10} "
        f"{'Total':>8} {'%C':>5} {'%A':>5}"
    )
    print(header_pf)
    print("-" * len(header_pf))

    for seq_len in cache_lens:
        r = _profile_prefill(
            seq_len,
            head_dim,
            num_query_heads,
            num_kv_heads,
            rotation,
            rotation_t,
            boundaries,
            centroids,
            device,
            warmup,
            iters,
        )
        results["prefill"].append(r)
        print(
            f"{seq_len:>8} "
            f"{r['compress']['median_ms']:>9.3f}ms "
            f"{r['attention']['median_ms']:>9.3f}ms "
            f"{r['total_median_ms']:>7.3f}ms "
            f"{r['pct_compress']:>4.1f}% "
            f"{r['pct_attention']:>4.1f}%"
        )

    # --- Decision ---
    last_decode = results["decode"][-1]
    pct_cd = last_decode["pct_compress"] + last_decode["pct_decompress"]
    bottleneck = "compress/decompress" if pct_cd > 60 else "attention"

    recommendation = (
        "Triton kernels (3c.8-3c.9) likely worth it -- "
        "fuse compress+write and read+dequant"
        if bottleneck == "compress/decompress"
        else "PyTorch compress/dequant path is acceptable -- "
        "skip to 3d production benchmark"
    )

    results["conclusion"] = {
        "bottleneck": bottleneck,
        "pct_compress_decompress": round(pct_cd, 1),
        "pct_attention": last_decode["pct_attention"],
        "recommendation": recommendation,
        "at_cache_len": last_decode["cache_len"],
    }

    print(f"\n{'=' * 60}")
    print(f"CONCLUSION at cache_len={last_decode['cache_len']}:")
    print(f"  Compress:    {last_decode['pct_compress']:.1f}%")
    print(f"  Decompress:  {last_decode['pct_decompress']:.1f}%")
    print(f"  Attention:   {last_decode['pct_attention']:.1f}%")
    print(f"  Bottleneck:  {bottleneck}")
    print(f"  -> {recommendation}")
    print(f"{'=' * 60}")

    results["vram_peak_mib"] = round(_get_vram_mb(), 1)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=("Experiment 015: Profile TQ4 cache bottleneck (P9 Phase 3c.7)"),
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=DEFAULT_HEAD_DIM,
        help=f"Head dimension (default: {DEFAULT_HEAD_DIM})",
    )
    parser.add_argument(
        "--num-query-heads",
        type=int,
        default=DEFAULT_NUM_QUERY_HEADS,
        help=f"Number of query heads (default: {DEFAULT_NUM_QUERY_HEADS})",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=DEFAULT_NUM_KV_HEADS,
        help=f"Number of KV heads (default: {DEFAULT_NUM_KV_HEADS})",
    )
    parser.add_argument(
        "--cache-lens",
        type=str,
        default=",".join(map(str, DEFAULT_CACHE_LENS)),
        help="Comma-separated cache/sequence lengths to profile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per benchmark (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Measurement iterations per benchmark (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/logs/experiment-015-profile-tq4-cache-bottleneck.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    cache_lens = [int(x) for x in args.cache_lens.split(",")]

    results = run_experiment(
        head_dim=args.head_dim,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        cache_lens=cache_lens,
        warmup=args.warmup,
        iters=args.iters,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
