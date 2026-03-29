r"""Experiment 020 -- Fused Paged TQ4 Decode Benchmark.

Hypothesis: The fused paged TQ4 decode kernel achieves 1.5-2x speedup at 4K
context and 2.5-3x at 32K vs decompress-all + attention baseline, due to
8.5x HBM traffic reduction (1160 → 136 bytes/token, Experiment 019).

Three paths measured:

    (a) Decompress-all baseline:  gather pages → tq4_decompress → SDPA
    (b) Fused paged decode:       fused_paged_tq4_decode() (in-tile decompress)
    (c) End-to-end TPOT via vLLM  with ``TQ4_USE_FUSED_PAGED=1`` (optional)

Parameters: context lengths [1K, 4K, 8K, 16K, 32K], single-sequence decode,
Molmo2 config (28Q/4KV, D=128, block_size=16).

Examples:
    ```bash
    # Full kernel benchmark (paths a+b)
    uv run python experiments/experiment_020_fused_paged_decode_benchmark.py

    # Quick smoke test
    uv run python experiments/experiment_020_fused_paged_decode_benchmark.py \
        --context-lens 1024,4096 --warmup 20 --timed 200

    # Include vLLM end-to-end (path c)
    uv run python experiments/experiment_020_fused_paged_decode_benchmark.py --e2e
    ```

See Also:
    ``experiments/experiment_019_fused_paged_hbm_traffic.py``:
        HBM traffic analysis (8.5x reduction confirmed).
    ``experiments/experiment_018_cuda_graph_decode_latency.py``:
        CUDA graph decode latency baseline methodology.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

# ── Molmo2 Config ──────────────────────────────────────────────────────────

H_Q = 28
H_KV = 4
HEAD_DIM = 128
HALF_D = HEAD_DIM // 2
BLOCK_SIZE = 16
SEED = 42
BITS = 4

# Per-token byte layout offsets
K_IDX_END = H_KV * HALF_D
K_NORM_END = K_IDX_END + H_KV * 4
V_IDX_END = K_NORM_END + H_KV * HALF_D
TOTAL_BYTES = V_IDX_END + H_KV * 4

DEFAULT_CONTEXT_LENS = [1024, 4096, 8192, 16384, 32768]
DEFAULT_WARMUP = 100
DEFAULT_TIMED = 1000

# Research projections from consolidated kernel design doc
PROJECTIONS = {
    1024: (1.2, 1.5),
    4096: (1.5, 2.0),
    8192: (1.8, 2.5),
    16384: (2.0, 2.8),
    32768: (2.5, 3.0),
}

MODEL_ID = "allenai/Molmo2-4B"

# ── Setup ──────────────────────────────────────────────────────────────────


def _init_quantizer() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Create TQ4 quantizer and extract GPU-resident primitives."""
    from turboquant_vllm.quantizer import TurboQuantMSE

    q = TurboQuantMSE(HEAD_DIM, bits=BITS, seed=SEED)
    centroids = q.codebook.centroids.cuda()
    rotation = q.rotation.cuda()
    boundaries = q.codebook.boundaries.cuda()
    rot_t = rotation.T.contiguous()
    rot_t_even = rot_t[:, 0::2].contiguous()
    rot_t_odd = rot_t[:, 1::2].contiguous()
    return centroids, rotation, boundaries, rot_t_even, rot_t_odd


def _build_paged_cache(
    seq_len: int,
    boundaries: torch.Tensor,
    rot_t_even: torch.Tensor,
    rot_t_odd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a compressed paged KV cache with random data."""
    from turboquant_vllm.triton.tq4_compress import tq4_compress

    torch.manual_seed(SEED)
    keys = torch.randn(seq_len, H_KV, HEAD_DIM, dtype=torch.float16, device="cuda")
    vals = torch.randn(seq_len, H_KV, HEAD_DIM, dtype=torch.float16, device="cuda")

    k_packed, k_norms = tq4_compress(keys, rot_t_even, rot_t_odd, boundaries)
    v_packed, v_norms = tq4_compress(vals, rot_t_even, rot_t_odd, boundaries)

    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, TOTAL_BYTES, dtype=torch.uint8, device="cuda"
    )
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32, device="cuda")

    phys_order = list(range(num_blocks))
    random.Random(SEED).shuffle(phys_order)

    for b in range(num_blocks):
        phys = phys_order[b]
        block_table[0, b] = phys
        s = b * BLOCK_SIZE
        e = min(s + BLOCK_SIZE, seq_len)
        n = e - s
        kv_cache[phys, :n, :K_IDX_END] = k_packed[s:e].reshape(n, -1)
        kv_cache[phys, :n, K_IDX_END:K_NORM_END] = (
            k_norms[s:e].reshape(n, H_KV).contiguous().view(torch.uint8)
        )
        kv_cache[phys, :n, K_NORM_END:V_IDX_END] = v_packed[s:e].reshape(n, -1)
        kv_cache[phys, :n, V_IDX_END:] = (
            v_norms[s:e].reshape(n, H_KV).contiguous().view(torch.uint8)
        )

    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    return kv_cache, block_table, seq_lens_t


# ── Path (a): Decompress-all + SDPA ───────────────────────────────────────


def _decompress_all_attend(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Decompress-all + attention baseline (replicates production fallback)."""
    from turboquant_vllm.triton.tq4_decompress import tq4_decompress

    sl = seq_lens[0].item()
    num_blocks = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE
    phys_blocks = block_table[0, :num_blocks]

    gathered = kv_cache[phys_blocks].reshape(-1, TOTAL_BYTES)[:sl]

    k_packed = gathered[:, :K_IDX_END].contiguous().reshape(sl, H_KV, HALF_D)
    k_norms = (
        gathered[:, K_IDX_END:K_NORM_END]
        .contiguous()
        .view(torch.float32)
        .reshape(sl, H_KV, 1)
    )
    v_packed = gathered[:, K_NORM_END:V_IDX_END].contiguous().reshape(sl, H_KV, HALF_D)
    v_norms = (
        gathered[:, V_IDX_END:].contiguous().view(torch.float32).reshape(sl, H_KV, 1)
    )

    k_dec = tq4_decompress(k_packed, k_norms, centroids, q.dtype)
    v_dec = tq4_decompress(v_packed, v_norms, centroids, q.dtype)

    q_rot = (q.float() @ rotation.T).to(q.dtype)

    # SDPA with GQA: (batch, heads, seqlen, dim)
    q_sdpa = q_rot.unsqueeze(2)  # [1, H_Q, 1, D]
    k_sdpa = k_dec.permute(1, 0, 2).unsqueeze(0)  # [1, H_KV, sl, D]
    v_sdpa = v_dec.permute(1, 0, 2).unsqueeze(0)  # [1, H_KV, sl, D]

    out_sdpa = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, scale=sm_scale, is_causal=False, enable_gqa=True
    )
    out_rot = out_sdpa.squeeze(2)  # [1, H_Q, D]
    return (out_rot.float() @ rotation).to(q.dtype)


# ── Path (b): Fused paged decode ──────────────────────────────────────────


def _fused_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Fused paged TQ4 decode (single kernel, in-tile decompress)."""
    from turboquant_vllm.triton.fused_paged_tq4_attention import (
        fused_paged_tq4_decode,
    )

    return fused_paged_tq4_decode(
        q,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        rotation,
        num_kv_heads=H_KV,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        sm_scale=sm_scale,
    )


# ── Timing ─────────────────────────────────────────────────────────────────


def _benchmark_fn(fn: object, warmup: int, timed: int) -> dict[str, float]:
    """Time a GPU function using CUDA events, return stats in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_us: list[float] = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000)  # ms → μs

    times_us.sort()
    return {
        "mean_us": round(statistics.mean(times_us), 2),
        "std_us": round(statistics.stdev(times_us) if len(times_us) > 1 else 0, 2),
        "p50_us": round(statistics.median(times_us), 2),
        "p99_us": round(times_us[int(len(times_us) * 0.99)], 2),
        "min_us": round(min(times_us), 2),
        "max_us": round(max(times_us), 2),
    }


# ── Correctness Gate ───────────────────────────────────────────────────────


def _correctness_gate(
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    rot_t_even: torch.Tensor,
    rot_t_odd: torch.Tensor,
) -> dict[str, float]:
    """Assert fused output matches decompress-all reference (cosine > 0.999)."""
    print("\n[Correctness Gate] Comparing fused vs decompress-all...")

    kv_cache, block_table, seq_lens = _build_paged_cache(
        256, boundaries, rot_t_even, rot_t_odd
    )
    torch.manual_seed(SEED + 1)
    q = torch.randn(1, H_Q, HEAD_DIM, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    with torch.inference_mode():
        ref = _decompress_all_attend(
            q, kv_cache, block_table, seq_lens, centroids, rotation, sm_scale
        )
        fused = _fused_decode(
            q, kv_cache, block_table, seq_lens, centroids, rotation, sm_scale
        )

    cos = F.cosine_similarity(
        ref.reshape(-1, HEAD_DIM).float(),
        fused.reshape(-1, HEAD_DIM).float(),
        dim=1,
    )
    mean_cos = cos.mean().item()
    min_cos = cos.min().item()

    print(f"  Mean cosine: {mean_cos:.6f}  Min cosine: {min_cos:.6f}")
    if min_cos < 0.999:
        print(f"  ABORT: Min cosine {min_cos:.6f} < 0.999 threshold")
        sys.exit(1)
    print("  PASS: cosine > 0.999")
    return {"mean_cosine": round(mean_cos, 6), "min_cosine": round(min_cos, 6)}


# ── Kernel Benchmark ───────────────────────────────────────────────────────


def _benchmark_kernels(
    context_lens: list[int],
    warmup: int,
    timed: int,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    rot_t_even: torch.Tensor,
    rot_t_odd: torch.Tensor,
) -> list[dict]:
    """Benchmark fused vs decompress-all at each context length."""
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    results = []

    for ctx_len in context_lens:
        print(f"\n  Context: {ctx_len}")
        kv_cache, block_table, seq_lens = _build_paged_cache(
            ctx_len, boundaries, rot_t_even, rot_t_odd
        )
        torch.manual_seed(SEED + 1)
        q = torch.randn(1, H_Q, HEAD_DIM, dtype=torch.float16, device="cuda")

        print(f"    (a) decompress-all: warmup={warmup}...", end="", flush=True)
        with torch.inference_mode():
            stats_a = _benchmark_fn(
                lambda _kv=kv_cache, _bt=block_table, _sl=seq_lens: (
                    _decompress_all_attend(
                        q, _kv, _bt, _sl, centroids, rotation, sm_scale
                    )
                ),
                warmup,
                timed,
            )
        print(f" p50={stats_a['p50_us']:.0f}μs")

        print(f"    (b) fused decode:   warmup={warmup}...", end="", flush=True)
        with torch.inference_mode():
            stats_b = _benchmark_fn(
                lambda _kv=kv_cache, _bt=block_table, _sl=seq_lens: _fused_decode(
                    q, _kv, _bt, _sl, centroids, rotation, sm_scale
                ),
                warmup,
                timed,
            )
        print(f" p50={stats_b['p50_us']:.0f}μs")

        speedup = stats_a["p50_us"] / stats_b["p50_us"] if stats_b["p50_us"] > 0 else 0
        proj_lo, proj_hi = PROJECTIONS.get(ctx_len, (0, 0))
        meets = (
            "within"
            if proj_lo <= speedup <= proj_hi
            else ("above" if speedup > proj_hi else "below")
        )
        print(
            f"    Speedup: {speedup:.2f}x  (projection: {proj_lo}-{proj_hi}x, {meets})"
        )

        results.append(
            {
                "context_length": ctx_len,
                "decompress_all": stats_a,
                "fused_decode": stats_b,
                "speedup": round(speedup, 3),
                "projection_range": [proj_lo, proj_hi],
                "meets_projection": meets,
            }
        )

        del kv_cache, block_table, seq_lens
        torch.cuda.empty_cache()

    return results


# ── Autotune Config Profiling ──────────────────────────────────────────────

# Decode configs: BLOCK_N in {16, 32, 64} × num_stages in {2, 3} × num_warps in {4, 8}
_DECODE_PROFILE_CONFIGS = [
    {"BLOCK_N": bn, "num_stages": s, "num_warps": w}
    for bn in [16, 32, 64]
    for s in [2, 3]
    for w in [4, 8]
]


def _profile_decode_autotune(
    context_lens: list[int],
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    rot_t_even: torch.Tensor,
    rot_t_odd: torch.Tensor,
    warmup: int = 20,
    timed: int = 100,
) -> dict:
    """Profile each autotune config at multiple context lengths."""
    import triton

    from turboquant_vllm.triton.fused_paged_tq4_attention import (
        _fused_paged_tq4_decode_kernel,
        fused_paged_tq4_decode,
    )

    autotuner = _fused_paged_tq4_decode_kernel
    original_configs = list(autotuner.configs)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    results: dict[str, dict[str, float]] = {}

    print("\n[Autotune Profiling] Decode kernel — 12 configs")

    try:
        for ctx_len in context_lens:
            kv_cache, block_table, seq_lens = _build_paged_cache(
                ctx_len, boundaries, rot_t_even, rot_t_odd
            )
            torch.manual_seed(SEED + 1)
            q = torch.randn(1, H_Q, HEAD_DIM, dtype=torch.float16, device="cuda")
            ctx_key = str(ctx_len)
            results[ctx_key] = {}
            print(f"\n  Context: {ctx_len}")

            for cfg_dict in _DECODE_PROFILE_CONFIGS:
                cfg = triton.Config(
                    {"BLOCK_N": cfg_dict["BLOCK_N"]},
                    num_stages=cfg_dict["num_stages"],
                    num_warps=cfg_dict["num_warps"],
                )
                autotuner.configs = [cfg]
                autotuner.cache.clear()

                cfg_label = (
                    f"BN={cfg_dict['BLOCK_N']}_s={cfg_dict['num_stages']}"
                    f"_w={cfg_dict['num_warps']}"
                )

                with torch.inference_mode():
                    stats = _benchmark_fn(
                        lambda _kv=kv_cache, _bt=block_table, _sl=seq_lens: (
                            fused_paged_tq4_decode(
                                q,
                                _kv,
                                _bt,
                                _sl,
                                centroids,
                                rotation,
                                num_kv_heads=H_KV,
                                head_dim=HEAD_DIM,
                                block_size=BLOCK_SIZE,
                                sm_scale=sm_scale,
                            )
                        ),
                        warmup,
                        timed,
                    )
                results[ctx_key][cfg_label] = stats["p50_us"]
                print(f"    {cfg_label}: {stats['p50_us']:.0f}μs")

            # Report winner for this context length
            winner = min(results[ctx_key], key=results[ctx_key].get)
            print(f"    → Winner: {winner} ({results[ctx_key][winner]:.0f}μs)")

            del kv_cache, block_table, seq_lens
            torch.cuda.empty_cache()
    finally:
        autotuner.configs = original_configs
        autotuner.cache.clear()

    # Analyze consistency
    winners = {ctx: min(cfgs, key=cfgs.get) for ctx, cfgs in results.items()}
    unique_winners = set(winners.values())
    consistent = len(unique_winners) == 1
    print(f"\n  Consistent winner: {'YES' if consistent else 'NO'} ({unique_winners})")

    return {
        "per_context_results": results,
        "winners": winners,
        "consistent": consistent,
    }


# ── Path (c): vLLM End-to-End ──────────────────────────────────────────────

_FILLER_PARAGRAPH = (
    "The transformer architecture processes sequences through self-attention "
    "mechanisms that compute pairwise interactions between all tokens in the "
    "input sequence. Key-value caches store intermediate attention states to "
    "avoid redundant computation during autoregressive generation, trading "
    "memory for compute. Quantization techniques like TQ4 compress these "
    "caches from sixteen-bit floating point to four-bit indices plus scalar "
    "norms, reducing memory footprint by approximately three point seven six "
    "times while preserving cosine similarity above zero point nine three. "
)


def _benchmark_e2e(
    context_lens: list[int],
    num_trials: int = 3,
    max_new_tokens: int = 32,
) -> list[dict]:
    """Path (c): end-to-end TPOT via vLLM with TQ4_USE_FUSED_PAGED=1."""
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig

    os.environ["TQ4_USE_FUSED_PAGED"] = "1"

    print("\n[Path (c)] vLLM end-to-end with TQ4_USE_FUSED_PAGED=1")
    print(f"  Model: {MODEL_ID}, trials={num_trials}, max_new={max_new_tokens}")

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=max(context_lens) + max_new_tokens + 64,
        gpu_memory_utilization=0.88,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        attention_config=AttentionConfig(backend="CUSTOM"),
    )
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    results = []

    for ctx_len in context_lens:
        repeats = (ctx_len // 80) + 5
        text = f"Context {ctx_len}. " + _FILLER_PARAGRAPH * repeats
        tokens = tokenizer.encode(text)[:ctx_len]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        actual_len = len(tokenizer.encode(prompt))
        print(f"\n  Context: {actual_len} tokens (target {ctx_len})")

        # Warmup
        llm.generate([prompt], params)

        trial_data = []
        for trial in range(num_trials):
            t_start = time.perf_counter()
            outputs = llm.generate([prompt], params)
            t_end = time.perf_counter()
            wall_s = t_end - t_start
            out_tokens = len(outputs[0].outputs[0].token_ids)
            # Wall time includes prefill + decode; metric is e2e average per token
            e2e_tpot_ms = wall_s / out_tokens * 1000 if out_tokens > 0 else 0
            trial_data.append(
                {
                    "trial": trial + 1,
                    "wall_s": round(wall_s, 4),
                    "output_tokens": out_tokens,
                    "e2e_tpot_ms": round(e2e_tpot_ms, 3),
                }
            )

        median_tpot = statistics.median([t["e2e_tpot_ms"] for t in trial_data])
        print(f"    e2e TPOT: {median_tpot:.2f}ms ({max_new_tokens} tokens/trial)")
        results.append(
            {
                "context_length": actual_len,
                "target_context_length": ctx_len,
                "max_new_tokens": max_new_tokens,
                "median_e2e_tpot_ms": round(median_tpot, 3),
                "trials": trial_data,
            }
        )

    del llm
    torch.cuda.empty_cache()
    return results


# ── Summary ────────────────────────────────────────────────────────────────


def _print_summary(kernel_results: list[dict]) -> None:
    """Print formatted comparison table."""
    print(f"\n\n{'#' * 70}")
    print("SUMMARY — Experiment 020: Fused Paged TQ4 Decode Benchmark")
    print(f"{'#' * 70}")

    header = (
        f"{'Ctx':>8} {'Decomp(μs)':>12} {'Fused(μs)':>12} "
        f"{'Speedup':>8} {'Proj':>10} {'Status':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for r in kernel_results:
        ctx = r["context_length"]
        da_p50 = r["decompress_all"]["p50_us"]
        f_p50 = r["fused_decode"]["p50_us"]
        sp = r["speedup"]
        proj = f"{r['projection_range'][0]}-{r['projection_range'][1]}x"
        status = r["meets_projection"]
        print(
            f"{ctx:>8} {da_p50:>12.0f} {f_p50:>12.0f} "
            f"{sp:>7.2f}x {proj:>10} {status:>8}"
        )


def _generate_markdown(experiment: dict) -> str:
    """Generate markdown summary from experiment results."""
    kr = experiment.get("kernel_results", [])
    lines = [
        "# Experiment 020: Fused Paged TQ4 Decode Benchmark",
        "",
        f"**Date:** {experiment['timestamp']}",
        f"**GPU:** {experiment['gpu']}",
        "**Config:** Molmo2 (28Q/4KV, D=128, block_size=16)",
        "",
        "## Correctness Gate",
        "",
    ]
    cg = experiment.get("correctness_gate", {})
    min_cos = cg.get("min_cosine")
    gate_label = (
        "**PASS**" if min_cos is not None and min_cos >= 0.999 else "**UNKNOWN**"
    )
    lines.append(
        f"Mean cosine: {cg.get('mean_cosine', 'N/A')}, "
        f"min cosine: {min_cos if min_cos is not None else 'N/A'} — {gate_label}"
    )
    lines.extend(
        [
            "",
            "## Kernel Benchmark (paths a vs b)",
            "",
            "| Context | Decompress-all (μs) | Fused (μs) | Speedup | Projection | Status |",
            "|--------:|--------------------:|-----------:|--------:|-----------:|-------:|",
        ]
    )
    for r in kr:
        ctx = r["context_length"]
        da = r["decompress_all"]["p50_us"]
        fu = r["fused_decode"]["p50_us"]
        sp = r["speedup"]
        proj = f"{r['projection_range'][0]}-{r['projection_range'][1]}x"
        status = r["meets_projection"]
        lines.append(
            f"| {ctx:,} | {da:,.0f} | {fu:,.0f} | {sp:.2f}x | {proj} | {status} |"
        )

    e2e = experiment.get("e2e_results")
    if e2e:
        lines.extend(
            [
                "",
                "## End-to-End TPOT (path c, TQ4_USE_FUSED_PAGED=1, includes prefill)",
                "",
                "| Context | e2e TPOT (ms) |",
                "|--------:|----------:|",
            ]
        )
        for r in e2e:
            lines.append(f"| {r['context_length']:,} | {r['median_e2e_tpot_ms']:.2f} |")

    # VRAM savings analysis (static calculation)
    # Without fused: decompress buffers = max_cache_tokens × H_KV × D × 2B × 2
    # With fused: decompress buffers = max_prefill_len × H_KV × D × 2B × 2
    # Molmo2: vLLM default ~32K cache, max_prefill ~2048
    max_cache = 32768
    max_prefill = 2048
    per_buf = H_KV * HEAD_DIM * 2  # bytes per token per buffer (fp16)
    without_mb = max_cache * per_buf * 2 / (1024 * 1024)
    with_mb = max_prefill * per_buf * 2 / (1024 * 1024)

    lines.extend(
        [
            "",
            "## Buffer Downsizing VRAM Savings",
            "",
            "With fused decode enabled, decompress buffers (pre-allocated for "
            "CUDA graph compatibility) are downsized from full-cache capacity "
            "to max-prefill capacity, since decode no longer decompresses.",
            "",
            "| Metric | Without fused | With fused |",
            "|--------|--------------|------------|",
            f"| Buffer sizing | {max_cache:,} tokens (full cache) "
            f"| {max_prefill:,} tokens (max prefill) |",
            f"| K+V decompress buffers | {without_mb:.1f} MiB | {with_mb:.1f} MiB |",
            f"| Savings | — | {without_mb - with_mb:.1f} MiB "
            f"({(1 - with_mb / without_mb) * 100:.0f}%) |",
            "",
            "## Analysis",
            "",
            "Values are p50 (median) from CUDA event timing.",
            "Decompress-all path: gather pages → tq4_decompress → SDPA (enable_gqa).",
            "Fused path: fused_paged_tq4_decode() (pre-rotate → in-tile decompress + "
            "attention → post-rotate).",
            "",
            f"Warmup: {experiment['config']['warmup']}, "
            f"timed: {experiment['config']['timed']} iterations per condition.",
        ]
    )
    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────


def run_experiment(
    context_lens: list[int],
    warmup: int,
    timed: int,
    include_e2e: bool = False,
    profile_autotune: bool = False,
) -> dict:
    """Run the full experiment."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    experiment: dict = {
        "experiment": "020-fused-paged-decode-benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "model_config": {
            "H_Q": H_Q,
            "H_KV": H_KV,
            "HEAD_DIM": HEAD_DIM,
            "BLOCK_SIZE": BLOCK_SIZE,
        },
        "config": {
            "context_lens": context_lens,
            "warmup": warmup,
            "timed": timed,
        },
        "hardware": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda or "N/A",
        },
    }

    print(f"\n{'#' * 70}")
    print("Experiment 020: Fused Paged TQ4 Decode Benchmark")
    print(f"GPU: {gpu_name}")
    print(f"Context lengths: {context_lens}")
    print(f"Warmup: {warmup}, Timed: {timed}")
    print(f"{'#' * 70}")

    # Init
    print("\n[Setup] Initializing quantizer...")
    centroids, rotation, boundaries, rot_t_even, rot_t_odd = _init_quantizer()

    # Correctness gate
    experiment["correctness_gate"] = _correctness_gate(
        centroids, rotation, boundaries, rot_t_even, rot_t_odd
    )

    # Kernel benchmark
    print("\n[Kernel Benchmark] Paths (a) decompress-all vs (b) fused decode")
    experiment["kernel_results"] = _benchmark_kernels(
        context_lens,
        warmup,
        timed,
        centroids,
        rotation,
        boundaries,
        rot_t_even,
        rot_t_odd,
    )

    _print_summary(experiment["kernel_results"])

    # Autotune profiling
    if profile_autotune:
        autotune_lens = [1024, 8192, 32768]
        experiment["autotune_profiling"] = _profile_decode_autotune(
            autotune_lens,
            centroids,
            rotation,
            boundaries,
            rot_t_even,
            rot_t_odd,
        )

    # Optional e2e
    if include_e2e:
        experiment["e2e_results"] = _benchmark_e2e(context_lens)

    return experiment


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment 020: Fused Paged TQ4 Decode Benchmark",
    )
    parser.add_argument(
        "--context-lens",
        type=str,
        default=",".join(map(str, DEFAULT_CONTEXT_LENS)),
        help="Comma-separated context lengths",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--timed", type=int, default=DEFAULT_TIMED)
    parser.add_argument(
        "--e2e", action="store_true", help="Include vLLM end-to-end (path c)"
    )
    parser.add_argument(
        "--profile-autotune",
        action="store_true",
        help="Profile all 12 decode autotune configs at 1K/8K/32K",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/logs/experiment-020-fused-paged-decode-benchmark.json",
    )
    args = parser.parse_args()

    context_lens = [int(x) for x in args.context_lens.split(",")]
    results = run_experiment(
        context_lens, args.warmup, args.timed, args.e2e, args.profile_autotune
    )

    # Write JSON
    json_path = Path(args.output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nJSON saved to {json_path}")

    # Write markdown
    md_path = json_path.with_suffix(".md")
    md_path.write_text(_generate_markdown(results))
    print(f"Markdown saved to {md_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
