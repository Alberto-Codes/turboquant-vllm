r"""Experiment 021 -- INT8 Fused Paged TQ4 Prefill Benchmark.

Hypothesis: The INT8 fused paged TQ4 prefill kernel achieves 1.3-2x speedup
vs decompress-all + attention baseline by using IMMA tensor cores for Q@K^T.

Caveat: Experiment 017 showed INT8/FP16 ratio of ~0.99x with unoptimized
single-stage kernels (48 TOPS, 7.3% of 660 TOPS peak). This benchmark
measures the current kernel as a pre-optimization baseline.  Task 3
(autotune config profiling) now focuses on refining single-stage kernel
configs (num_stages=1); deeper pipeline staging was explored but is
currently slower on RTX 4090 and not the primary path to INT8 speedup.

Three paths measured:

    (a) Decompress-all baseline:  gather pages → tq4_decompress → SDPA
    (b) INT8 fused prefill:       fused_paged_tq4_int8_prefill() (IMMA Q@K^T)
    (c) End-to-end via vLLM       with both feature gates enabled (optional)

Parameters: prefill lengths [128, 512, 1024, 2048], single-sequence,
Molmo2 config (28Q/4KV, D=128, block_size=16).

Examples:
    ```bash
    # Full kernel benchmark (paths a+b)
    uv run python experiments/experiment_021_int8_prefill_benchmark.py

    # Quick smoke test
    uv run python experiments/experiment_021_int8_prefill_benchmark.py \
        --prefill-lens 128,512 --warmup 10 --timed 100

    # Include vLLM end-to-end (path c)
    uv run python experiments/experiment_021_int8_prefill_benchmark.py --e2e
    ```

See Also:
    ``experiments/experiment_017_int8_tensor_core_dispatch.py``:
        Phase 0 INT8 tensor core validation (IMMA confirmed, 48 TOPS).
    ``experiments/experiment_020_fused_paged_decode_benchmark.py``:
        Companion decode benchmark (FP16 path).
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

K_IDX_END = H_KV * HALF_D
K_NORM_END = K_IDX_END + H_KV * 4
V_IDX_END = K_NORM_END + H_KV * HALF_D
TOTAL_BYTES = V_IDX_END + H_KV * 4

DEFAULT_PREFILL_LENS = [128, 512, 1024, 2048]
DEFAULT_WARMUP = 50
DEFAULT_TIMED = 500

PROJECTIONS = {
    128: (1.0, 1.3),
    512: (1.1, 1.5),
    1024: (1.3, 1.8),
    2048: (1.3, 2.0),
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
    """Build a compressed paged KV cache with random data (single sequence)."""
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
    """Decompress-all + SDPA baseline for prefill (causal masking)."""
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

    # SDPA with GQA + causal: (batch, heads, seqlen, dim)
    q_sdpa = q_rot.permute(1, 0, 2).unsqueeze(0)  # [1, H_Q, num_tokens, D]
    k_sdpa = k_dec.permute(1, 0, 2).unsqueeze(0)  # [1, H_KV, sl, D]
    v_sdpa = v_dec.permute(1, 0, 2).unsqueeze(0)  # [1, H_KV, sl, D]

    out_sdpa = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, scale=sm_scale, is_causal=True, enable_gqa=True
    )
    out_rot = out_sdpa.squeeze(0).permute(1, 0, 2)  # [num_tokens, H_Q, D]
    return (out_rot.float() @ rotation).to(q.dtype)


# ── Path (b): INT8 fused prefill ──────────────────────────────────────────


def _int8_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """INT8 fused paged TQ4 prefill (IMMA Q@K^T, FP16 P@V)."""
    from turboquant_vllm.triton.fused_paged_tq4_int8_prefill import (
        fused_paged_tq4_int8_prefill,
    )

    return fused_paged_tq4_int8_prefill(
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
        times_us.append(start.elapsed_time(end) * 1000)

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
    """Assert INT8 prefill output matches decompress-all reference (cosine > 0.998)."""
    print("\n[Correctness Gate] Comparing INT8 prefill vs decompress-all...")

    ctx_len = 128
    kv_cache, block_table, seq_lens = _build_paged_cache(
        ctx_len, boundaries, rot_t_even, rot_t_odd
    )
    torch.manual_seed(SEED + 1)
    q = torch.randn(ctx_len, H_Q, HEAD_DIM, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    with torch.inference_mode():
        ref = _decompress_all_attend(
            q, kv_cache, block_table, seq_lens, centroids, rotation, sm_scale
        )
        int8 = _int8_prefill(
            q, kv_cache, block_table, seq_lens, centroids, rotation, sm_scale
        )

    cos = F.cosine_similarity(
        ref.reshape(-1, HEAD_DIM).float(),
        int8.reshape(-1, HEAD_DIM).float(),
        dim=1,
    )
    mean_cos = cos.mean().item()
    min_cos = cos.min().item()

    print(f"  Mean cosine: {mean_cos:.6f}  Min cosine: {min_cos:.6f}")
    if min_cos < 0.998:
        print(f"  ABORT: Min cosine {min_cos:.6f} < 0.998 threshold")
        sys.exit(1)
    print("  PASS: cosine > 0.998")
    return {"mean_cosine": round(mean_cos, 6), "min_cosine": round(min_cos, 6)}


# ── Kernel Benchmark ───────────────────────────────────────────────────────


def _benchmark_kernels(
    prefill_lens: list[int],
    warmup: int,
    timed: int,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    rot_t_even: torch.Tensor,
    rot_t_odd: torch.Tensor,
) -> list[dict]:
    """Benchmark INT8 prefill vs decompress-all at each prefill length."""
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    results = []

    for pf_len in prefill_lens:
        print(f"\n  Prefill length: {pf_len}")
        kv_cache, block_table, seq_lens = _build_paged_cache(
            pf_len, boundaries, rot_t_even, rot_t_odd
        )
        torch.manual_seed(SEED + 1)
        q = torch.randn(pf_len, H_Q, HEAD_DIM, dtype=torch.float16, device="cuda")

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

        print(f"    (b) INT8 prefill:   warmup={warmup}...", end="", flush=True)
        with torch.inference_mode():
            stats_b = _benchmark_fn(
                lambda _kv=kv_cache, _bt=block_table, _sl=seq_lens: _int8_prefill(
                    q, _kv, _bt, _sl, centroids, rotation, sm_scale
                ),
                warmup,
                timed,
            )
        print(f" p50={stats_b['p50_us']:.0f}μs")

        speedup = stats_a["p50_us"] / stats_b["p50_us"] if stats_b["p50_us"] > 0 else 0
        proj_lo, proj_hi = PROJECTIONS.get(pf_len, (0, 0))
        meets = (
            "within"
            if proj_lo <= speedup <= proj_hi
            else ("above" if speedup > proj_hi else "below")
        )

        # Throughput (tokens/sec) from p50 latency
        tput_a = pf_len / (stats_a["p50_us"] / 1e6) if stats_a["p50_us"] > 0 else 0
        tput_b = pf_len / (stats_b["p50_us"] / 1e6) if stats_b["p50_us"] > 0 else 0

        print(
            f"    Speedup: {speedup:.2f}x  (projection: {proj_lo}-{proj_hi}x, {meets})"
        )
        print(f"    Throughput: baseline={tput_a:,.0f} tok/s, INT8={tput_b:,.0f} tok/s")

        results.append(
            {
                "prefill_length": pf_len,
                "decompress_all": stats_a,
                "int8_prefill": stats_b,
                "speedup": round(speedup, 3),
                "throughput_baseline_tok_s": round(tput_a, 0),
                "throughput_int8_tok_s": round(tput_b, 0),
                "projection_range": [proj_lo, proj_hi],
                "meets_projection": meets,
            }
        )

        del kv_cache, block_table, seq_lens
        torch.cuda.empty_cache()

    return results


# ── Autotune Config Profiling ──────────────────────────────────────────────

# INT8 prefill configs: BLOCK_N in {16, 32} × num_stages in {1, 2, 3} × num_warps in {4, 8}
_PREFILL_PROFILE_CONFIGS = [
    {"BLOCK_N": bn, "num_stages": s, "num_warps": w}
    for bn in [16, 32]
    for s in [1, 2, 3]
    for w in [4, 8]
]


def _profile_prefill_autotune(
    prefill_lens: list[int],
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    rot_t_even: torch.Tensor,
    rot_t_odd: torch.Tensor,
    warmup: int = 20,
    timed: int = 100,
) -> dict:
    """Profile each INT8 prefill autotune config at multiple lengths."""
    import triton

    from turboquant_vllm.triton.fused_paged_tq4_int8_prefill import (
        _fused_paged_tq4_int8_prefill_kernel,
        fused_paged_tq4_int8_prefill,
    )

    autotuner = _fused_paged_tq4_int8_prefill_kernel
    original_configs = list(autotuner.configs)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    results: dict[str, dict[str, float]] = {}

    print("\n[Autotune Profiling] INT8 prefill kernel — 12 configs")

    try:
        for pf_len in prefill_lens:
            kv_cache, block_table, seq_lens = _build_paged_cache(
                pf_len, boundaries, rot_t_even, rot_t_odd
            )
            torch.manual_seed(SEED + 1)
            q = torch.randn(pf_len, H_Q, HEAD_DIM, dtype=torch.float16, device="cuda")
            pf_key = str(pf_len)
            results[pf_key] = {}
            print(f"\n  Prefill length: {pf_len}")

            for cfg_dict in _PREFILL_PROFILE_CONFIGS:
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
                            fused_paged_tq4_int8_prefill(
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
                results[pf_key][cfg_label] = stats["p50_us"]
                print(f"    {cfg_label}: {stats['p50_us']:.0f}μs")

            winner = min(results[pf_key], key=results[pf_key].get)
            print(f"    → Winner: {winner} ({results[pf_key][winner]:.0f}μs)")

            del kv_cache, block_table, seq_lens
            torch.cuda.empty_cache()
    finally:
        autotuner.configs = original_configs
        autotuner.cache.clear()

    winners = {pf: min(cfgs, key=cfgs.get) for pf, cfgs in results.items()}
    unique_winners = set(winners.values())
    consistent = len(unique_winners) == 1
    print(f"\n  Consistent winner: {'YES' if consistent else 'NO'} ({unique_winners})")

    return {
        "per_length_results": results,
        "winners": winners,
        "consistent": consistent,
    }


# ── Path (c): vLLM End-to-End ──────────────────────────────────────────────

_FILLER_PARAGRAPH = (
    "The transformer architecture processes sequences through self-attention "
    "mechanisms that compute pairwise interactions between all tokens in the "
    "input sequence. Key-value caches store intermediate attention states to "
    "avoid redundant computation during autoregressive generation, trading "
    "memory for compute. "
)


def _benchmark_e2e(
    prefill_lens: list[int],
    num_trials: int = 3,
    max_new_tokens: int = 8,
) -> list[dict]:
    """Path (c): end-to-end via vLLM with both feature gates enabled."""
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig

    os.environ["TQ4_USE_FUSED_PAGED"] = "1"
    os.environ["TQ4_USE_INT8_PREFILL"] = "1"

    print("\n[Path (c)] vLLM e2e with TQ4_USE_FUSED_PAGED=1 TQ4_USE_INT8_PREFILL=1")
    print(f"  Model: {MODEL_ID}, trials={num_trials}")

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=max(prefill_lens) + max_new_tokens + 64,
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

    for pf_len in prefill_lens:
        repeats = (pf_len // 40) + 5
        text = f"Prefill {pf_len}. " + _FILLER_PARAGRAPH * repeats
        tokens = tokenizer.encode(text)[:pf_len]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        actual_len = len(tokenizer.encode(prompt))
        print(f"\n  Prefill: {actual_len} tokens (target {pf_len})")

        llm.generate([prompt], params)  # warmup

        trial_data = []
        for trial in range(num_trials):
            t_start = time.perf_counter()
            outputs = llm.generate([prompt], params)
            t_end = time.perf_counter()
            wall_s = t_end - t_start
            out_tokens = len(outputs[0].outputs[0].token_ids)
            # Wall time includes prefill + decode (8 tokens); metric is e2e throughput
            e2e_prefill_tput = actual_len / wall_s if wall_s > 0 else 0
            trial_data.append(
                {
                    "trial": trial + 1,
                    "wall_s": round(wall_s, 4),
                    "output_tokens": out_tokens,
                    "e2e_prefill_throughput_tok_s": round(e2e_prefill_tput, 0),
                }
            )

        median_tput = statistics.median(
            [t["e2e_prefill_throughput_tok_s"] for t in trial_data]
        )
        print(f"    e2e throughput: {median_tput:,.0f} tok/s")
        results.append(
            {
                "prefill_length": actual_len,
                "target_prefill_length": pf_len,
                "median_e2e_throughput_tok_s": round(median_tput, 0),
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
    print("SUMMARY — Experiment 021: INT8 Prefill Benchmark")
    print(f"{'#' * 70}")

    header = (
        f"{'PfLen':>8} {'Decomp(μs)':>12} {'INT8(μs)':>12} "
        f"{'Speedup':>8} {'Proj':>10} {'Status':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for r in kernel_results:
        pf = r["prefill_length"]
        da = r["decompress_all"]["p50_us"]
        i8 = r["int8_prefill"]["p50_us"]
        sp = r["speedup"]
        proj = f"{r['projection_range'][0]}-{r['projection_range'][1]}x"
        status = r["meets_projection"]
        print(f"{pf:>8} {da:>12.0f} {i8:>12.0f} {sp:>7.2f}x {proj:>10} {status:>8}")


def _generate_markdown(experiment: dict) -> str:
    """Generate markdown summary from experiment results."""
    kr = experiment.get("kernel_results", [])
    lines = [
        "# Experiment 021: INT8 Fused Paged TQ4 Prefill Benchmark",
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
        "**PASS**" if min_cos is not None and min_cos >= 0.998 else "**UNKNOWN**"
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
            "| Prefill | Decomp (μs) | INT8 (μs) | Speedup "
            "| Base tok/s | INT8 tok/s | Proj | Status |",
            "|--------:|-----------:|----------:|--------:"
            "|-----------:|-----------:|-----:|-------:|",
        ]
    )
    for r in kr:
        pf = r["prefill_length"]
        da = r["decompress_all"]["p50_us"]
        i8 = r["int8_prefill"]["p50_us"]
        sp = r["speedup"]
        tb = r["throughput_baseline_tok_s"]
        ti = r["throughput_int8_tok_s"]
        proj = f"{r['projection_range'][0]}-{r['projection_range'][1]}x"
        status = r["meets_projection"]
        lines.append(
            f"| {pf:,} | {da:,.0f} | {i8:,.0f} | {sp:.2f}x "
            f"| {tb:,.0f} | {ti:,.0f} | {proj} | {status} |"
        )

    e2e = experiment.get("e2e_results")
    if e2e:
        lines.extend(
            [
                "",
                "## End-to-End Throughput (path c, both gates enabled, includes decode)",
                "",
                "| Prefill | e2e Throughput (tok/s) |",
                "|--------:|-------------------:|",
            ]
        )
        for r in e2e:
            lines.append(
                f"| {r['prefill_length']:,} | {r['median_e2e_throughput_tok_s']:,.0f} |"
            )

    # INT8 prefill gate recommendation — use min/median, not max
    speedups = [r["speedup"] for r in kr]
    min_speedup = min(speedups) if speedups else 0
    median_speedup = statistics.median(speedups) if speedups else 0
    max_speedup = max(speedups) if speedups else 0
    if min_speedup >= 1.0 and median_speedup >= 1.05:
        rec = "enable-by-default"
        rec_detail = (
            f"INT8 prefill shows {median_speedup:.2f}x median speedup (min "
            f"{min_speedup:.2f}x, max {max_speedup:.2f}x) — all lengths at or above "
            "parity. Recommend enabling TQ4_USE_INT8_PREFILL by default."
        )
    elif median_speedup >= 1.0:
        rec = "disable-by-default"
        rec_detail = (
            f"INT8 prefill shows {median_speedup:.2f}x median speedup (min "
            f"{min_speedup:.2f}x, max {max_speedup:.2f}x) — marginal overall. "
            "Keep TQ4_USE_INT8_PREFILL disabled by default; users can opt in."
        )
    else:
        rec = "disable-by-default"
        rec_detail = (
            f"INT8 prefill shows {median_speedup:.2f}x median speedup (min "
            f"{min_speedup:.2f}x, max {max_speedup:.2f}x). "
            "Per-tile INT8 quantization overhead negates IMMA throughput advantage "
            "at longer sequences. Keep gate disabled. Revisit after per-row quantization."
        )

    lines.extend(
        [
            "",
            "## INT8 Prefill Gate Recommendation",
            "",
            f"**Recommendation:** `{rec}`",
            "",
            rec_detail,
            "",
            "## Notes",
            "",
            "- Values are p50 (median) from CUDA event timing.",
            "- Decompress-all: gather pages → tq4_decompress → SDPA (causal, enable_gqa).",
            "- INT8 prefill: fused_paged_tq4_int8_prefill() (IMMA Q@K^T + FP16 P@V).",
            "- Experiment 017 showed ~0.99x INT8/FP16 ratio with naive single-stage "
            "kernels. Speedup depends on autotune optimization (Task 3).",
            f"- Warmup: {experiment['config']['warmup']}, "
            f"timed: {experiment['config']['timed']} iterations per condition.",
        ]
    )
    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────


def run_experiment(
    prefill_lens: list[int],
    warmup: int,
    timed: int,
    include_e2e: bool = False,
    profile_autotune: bool = False,
) -> dict:
    """Run the full INT8 prefill experiment."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    experiment: dict = {
        "experiment": "021-int8-prefill-benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "model_config": {
            "H_Q": H_Q,
            "H_KV": H_KV,
            "HEAD_DIM": HEAD_DIM,
            "BLOCK_SIZE": BLOCK_SIZE,
        },
        "config": {
            "prefill_lens": prefill_lens,
            "warmup": warmup,
            "timed": timed,
        },
        "hardware": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda or "N/A",
        },
    }

    print(f"\n{'#' * 70}")
    print("Experiment 021: INT8 Fused Paged TQ4 Prefill Benchmark")
    print(f"GPU: {gpu_name}")
    print(f"Prefill lengths: {prefill_lens}")
    print(f"Warmup: {warmup}, Timed: {timed}")
    print(f"{'#' * 70}")

    print("\n[Setup] Initializing quantizer...")
    centroids, rotation, boundaries, rot_t_even, rot_t_odd = _init_quantizer()

    experiment["correctness_gate"] = _correctness_gate(
        centroids, rotation, boundaries, rot_t_even, rot_t_odd
    )

    print("\n[Kernel Benchmark] Paths (a) decompress-all vs (b) INT8 prefill")
    experiment["kernel_results"] = _benchmark_kernels(
        prefill_lens,
        warmup,
        timed,
        centroids,
        rotation,
        boundaries,
        rot_t_even,
        rot_t_odd,
    )

    _print_summary(experiment["kernel_results"])

    if profile_autotune:
        autotune_lens = [128, 1024, 2048]
        experiment["autotune_profiling"] = _profile_prefill_autotune(
            autotune_lens,
            centroids,
            rotation,
            boundaries,
            rot_t_even,
            rot_t_odd,
        )

    if include_e2e:
        experiment["e2e_results"] = _benchmark_e2e(prefill_lens)

    return experiment


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment 021: INT8 Fused Paged TQ4 Prefill Benchmark",
    )
    parser.add_argument(
        "--prefill-lens",
        type=str,
        default=",".join(map(str, DEFAULT_PREFILL_LENS)),
        help="Comma-separated prefill lengths",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--timed", type=int, default=DEFAULT_TIMED)
    parser.add_argument(
        "--e2e", action="store_true", help="Include vLLM end-to-end (path c)"
    )
    parser.add_argument(
        "--profile-autotune",
        action="store_true",
        help="Profile all 12 INT8 prefill autotune configs at 128/1024/2048",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/logs/experiment-021-int8-prefill-benchmark.json",
    )
    args = parser.parse_args()

    prefill_lens = [int(x) for x in args.prefill_lens.split(",")]
    results = run_experiment(
        prefill_lens, args.warmup, args.timed, args.e2e, args.profile_autotune
    )

    json_path = Path(args.output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nJSON saved to {json_path}")

    md_path = json_path.with_suffix(".md")
    md_path.write_text(_generate_markdown(results))
    print(f"Markdown saved to {md_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
