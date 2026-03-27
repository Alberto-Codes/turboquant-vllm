r"""Experiment 016 -- Benchmark Triton TQ4 compress/decompress kernels vs PyTorch.

P9 Phases 3c.8-3c.9: Measures the speedup from fused Triton kernels for
TQ4 compress and decompress, plus the pre/post-rotation optimization that
moves O(cache_len) matmuls to O(1) Q/output rotations.

Profiles the full decode step breakdown: compress + pre-rotate Q +
decompress + Flash Attention + post-rotate output.

Examples:
    ```bash
    uv run python experiments/experiment_016_triton_kernel_benchmark.py
    uv run python experiments/experiment_016_triton_kernel_benchmark.py \
        --cache-lens 256,1024,4096
    ```

See Also:
    :mod:`turboquant_consumer.triton.tq4_compress`: Phase 3c.9 fused compress.
    :mod:`turboquant_consumer.triton.tq4_decompress`: Phase 3c.8 fused decompress.
    ``experiments/experiment_015_profile_tq4_cache_bottleneck.py``: Pre-Triton baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from turboquant_consumer.quantizer import TurboQuantMSE
from turboquant_consumer.triton.tq4_compress import tq4_compress
from turboquant_consumer.triton.tq4_decompress import tq4_decompress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TQ4_BITS = 4
TQ4_SEED = 42
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_NUM_QUERY_HEADS = 32
DEFAULT_CACHE_LENS = [128, 256, 512, 1024, 2048, 4096]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_vram_mb() -> float:
    """Return peak GPU memory in MiB.

    Returns:
        Peak VRAM in MiB, or 0.0 if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_vram() -> None:
    """Reset peak memory stats and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


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
# PyTorch baseline (old path)
# ---------------------------------------------------------------------------


def _pytorch_compress_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    rotation_t: torch.Tensor,
    boundaries: torch.Tensor,
) -> None:
    """Old PyTorch compress path for K+V (no Triton)."""
    for x in [key, value]:
        N, H, D = x.shape
        flat = x.reshape(N * H, D).float()
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ rotation_t
        indices = torch.bucketize(rotated, boundaries).clamp(0, 15)
        idx_u8 = indices.to(torch.uint8)
        _packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]


def _pytorch_decompress_kv(
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
) -> None:
    """Old PyTorch decompress path for K+V (with rotation)."""
    for packed, norms in [(k_packed, k_norms), (v_packed, v_norms)]:
        N, H, half_D = packed.shape
        D = half_D * 2
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(N * H, D)
        flat_norms = norms.reshape(N * H, 1)
        reconstructed = centroids[indices]
        _unrotated = (reconstructed @ rotation) * flat_norms


# ---------------------------------------------------------------------------
# Benchmark
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
    rot_T_even: torch.Tensor,
    rot_T_odd: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Profile one decode step: old PyTorch vs new Triton path.

    Returns:
        Dict with old/new timing breakdown and speedups.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    half_D = head_dim // 2
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
    cache_k_packed = torch.randint(
        0,
        255,
        (cache_len, num_kv_heads, half_D),
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
        (cache_len, num_kv_heads, half_D),
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

    # --- Old path ---
    old_compress = _benchmark_op(
        lambda: _pytorch_compress_kv(new_k, new_v, rotation_t, boundaries),
        warmup,
        iters,
    )
    old_decompress = _benchmark_op(
        lambda: _pytorch_decompress_kv(
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

    # Pre-decompress for attention benchmark
    kv_full_old = []
    for packed, norms in [
        (cache_k_packed, cache_k_norms),
        (cache_v_packed, cache_v_norms),
    ]:
        N, H, hD = packed.shape
        D = hD * 2
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(N * H, D)
        flat_norms = norms.reshape(N * H, 1)
        reconstructed = centroids[indices]
        unrotated = (reconstructed @ rotation) * flat_norms
        kv_full_old.append(unrotated.reshape(N, H, D).to(torch.float16))

    cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, cache_len], device=device, dtype=torch.int32)

    old_attn = _benchmark_op(
        lambda: flash_attn_varlen_func(
            q=query,
            k=kv_full_old[0],
            v=kv_full_old[1],
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=cache_len,
            cu_seqlens_k=cu_k,
        ),
        warmup,
        iters,
    )
    old_total = (
        old_compress["median_ms"] + old_decompress["median_ms"] + old_attn["median_ms"]
    )

    # --- New path (Triton + pre/post-rotation) ---
    new_compress = _benchmark_op(
        lambda: (
            tq4_compress(new_k, rot_T_even, rot_T_odd, boundaries),
            tq4_compress(new_v, rot_T_even, rot_T_odd, boundaries),
        ),
        warmup,
        iters,
    )
    new_prerot = _benchmark_op(
        lambda: (query.float() @ rotation_t).to(torch.float16),
        warmup,
        iters,
    )
    new_decompress = _benchmark_op(
        lambda: (
            tq4_decompress(
                cache_k_packed,
                cache_k_norms,
                centroids,
                torch.float16,
            ),
            tq4_decompress(
                cache_v_packed,
                cache_v_norms,
                centroids,
                torch.float16,
            ),
        ),
        warmup,
        iters,
    )

    q_rot = (query.float() @ rotation_t).to(torch.float16)
    k_rot = tq4_decompress(
        cache_k_packed,
        cache_k_norms,
        centroids,
        torch.float16,
    )
    v_rot = tq4_decompress(
        cache_v_packed,
        cache_v_norms,
        centroids,
        torch.float16,
    )
    new_attn = _benchmark_op(
        lambda: flash_attn_varlen_func(
            q=q_rot,
            k=k_rot,
            v=v_rot,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=cache_len,
            cu_seqlens_k=cu_k,
        ),
        warmup,
        iters,
    )

    out = torch.randn(
        1,
        num_query_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    new_postrot = _benchmark_op(
        lambda: (out.float() @ rotation).to(torch.float16),
        warmup,
        iters,
    )

    new_total = (
        new_compress["median_ms"]
        + new_prerot["median_ms"]
        + new_decompress["median_ms"]
        + new_attn["median_ms"]
        + new_postrot["median_ms"]
    )

    return {
        "cache_len": cache_len,
        "old": {
            "compress": old_compress,
            "decompress": old_decompress,
            "attention": old_attn,
            "total_median_ms": round(old_total, 4),
        },
        "new": {
            "compress": new_compress,
            "prerotate_q": new_prerot,
            "decompress": new_decompress,
            "attention": new_attn,
            "postrotate_out": new_postrot,
            "total_median_ms": round(new_total, 4),
        },
        "speedup": round(old_total / new_total, 2),
        "compress_speedup": round(
            old_compress["median_ms"] / new_compress["median_ms"],
            2,
        ),
        "decompress_speedup": round(
            old_decompress["median_ms"] / new_decompress["median_ms"],
            2,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(
    head_dim: int = DEFAULT_HEAD_DIM,
    num_query_heads: int = DEFAULT_NUM_QUERY_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    cache_lens: list[int] | None = None,
    warmup: int = 10,
    iters: int = 100,
) -> dict[str, Any]:
    """Run the Triton kernel benchmark experiment.

    Returns:
        Dict with per-cache-length results and GPU info.
    """
    if cache_lens is None:
        cache_lens = list(DEFAULT_CACHE_LENS)

    device = torch.device("cuda")

    quantizer = TurboQuantMSE(head_dim, TQ4_BITS, seed=TQ4_SEED)
    rotation = quantizer.rotation.to(device)
    rotation_t = rotation.T.contiguous()
    boundaries = quantizer.codebook.boundaries.to(device)
    centroids = quantizer.codebook.centroids.to(device)
    rot_T_even = rotation_t[:, 0::2].contiguous()
    rot_T_odd = rotation_t[:, 1::2].contiguous()

    gpu_name = torch.cuda.get_device_name(0)
    _reset_vram()

    results: dict[str, Any] = {
        "experiment": "016-triton-kernel-benchmark",
        "phase": "P9-3c.8-3c.9",
        "gpu": gpu_name,
        "config": {
            "head_dim": head_dim,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "tq4_bits": TQ4_BITS,
            "warmup": warmup,
            "iters": iters,
        },
        "decode_steps": [],
    }

    print(f"\n{'=' * 72}")
    print(f"Triton TQ4 Kernel Benchmark -- {gpu_name}")
    print(f"Config: D={head_dim}, Qh={num_query_heads}, KVh={num_kv_heads}")
    print(f"{'=' * 72}")

    header = (
        f"{'Cache':>8} {'Old(ms)':>10} {'New(ms)':>10} "
        f"{'Speedup':>8} {'C-speed':>8} {'D-speed':>8}"
    )
    print(f"\n{header}")
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
            rot_T_even,
            rot_T_odd,
            device,
            warmup,
            iters,
        )
        results["decode_steps"].append(r)
        print(
            f"{cache_len:>8} "
            f"{r['old']['total_median_ms']:>9.3f}ms "
            f"{r['new']['total_median_ms']:>9.3f}ms "
            f"{r['speedup']:>7.1f}x "
            f"{r['compress_speedup']:>7.1f}x "
            f"{r['decompress_speedup']:>7.1f}x"
        )

    last = results["decode_steps"][-1]
    print(f"\n{'=' * 72}")
    print(f"At cache_len={last['cache_len']}:")
    print(f"  Old total:  {last['old']['total_median_ms']:.3f}ms")
    print(f"  New total:  {last['new']['total_median_ms']:.3f}ms")
    print(f"  Speedup:    {last['speedup']:.1f}x")
    print(f"{'=' * 72}")

    results["vram_peak_mib"] = round(_get_vram_mb(), 1)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment 016: Triton TQ4 kernel benchmark (P9 3c.8-3c.9)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=DEFAULT_HEAD_DIM,
    )
    parser.add_argument(
        "--num-query-heads",
        type=int,
        default=DEFAULT_NUM_QUERY_HEADS,
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=DEFAULT_NUM_KV_HEADS,
    )
    parser.add_argument(
        "--cache-lens",
        type=str,
        default=",".join(map(str, DEFAULT_CACHE_LENS)),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/logs/experiment-016-triton-kernel-benchmark.json",
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
