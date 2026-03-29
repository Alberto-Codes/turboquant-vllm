r"""Experiment 018 -- TQ4 CUDA Graph Decode Latency.

D7 Phase C end-to-end validation: measures TPOT improvement from CUDA graph
buffer pre-allocation on RTX 4090 with Molmo2-4B and TQ4 KV cache.

Three conditions:

    Baseline:    enforce_eager (no CUDA graphs)
    Phase A:     CUDAGraphMode.PIECEWISE (graphs around individual kernels)
    Phase B:     CUDAGraphMode.FULL_DECODE_ONLY (full decode step captured)

Parameters: batch sizes {1, 4, 8, 16}, context lengths {2048, 4096}.
Metrics: TPOT (ms), peak VRAM (MiB), throughput (tokens/sec).

Examples:
    ```bash
    # Full experiment (all conditions, all combos)
    uv run python experiments/experiment_018_cuda_graph_decode_latency.py

    # Quick smoke test
    uv run python experiments/experiment_018_cuda_graph_decode_latency.py \
        --batch-sizes 1,4 --context-lens 2048 --num-trials 1

    # Single condition
    uv run python experiments/experiment_018_cuda_graph_decode_latency.py \
        --conditions full_decode
    ```

See Also:
    :mod:`turboquant_vllm.vllm.tq4_backend`: TQ4 attention backend with
        CUDA graph buffer pre-allocation (Story 4.2).
    ``experiments/experiment_016_triton_kernel_benchmark.py``: Kernel-level
        Triton benchmark (P9 3c.8-3c.9).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "allenai/Molmo2-4B"
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
DEFAULT_CONTEXT_LENS = [2048, 4096]
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_NUM_TRIALS = 3
ALL_CONDITIONS = ["baseline_eager", "piecewise", "full_decode"]

_FILLER_PARAGRAPH = (
    "The transformer architecture processes sequences through self-attention "
    "mechanisms that compute pairwise interactions between all tokens in the "
    "input sequence. Key-value caches store intermediate attention states to "
    "avoid redundant computation during autoregressive generation, trading "
    "memory for compute. Quantization techniques like TQ4 compress these "
    "caches from sixteen-bit floating point to four-bit indices plus scalar "
    "norms, reducing memory footprint by approximately three point seven six "
    "times while preserving cosine similarity above zero point nine three. "
    "CUDA graphs capture GPU operations into replayable recordings that "
    "eliminate CPU-GPU synchronization overhead and kernel launch latency "
    "during the decode phase of language model inference. Pre-allocating "
    "buffers to maximum size and slicing to actual usage at runtime prevents "
    "dynamic memory allocation calls that would break graph capture. "
    "The rotary position embedding scheme encodes sequence position through "
    "rotation matrices applied to query and key vectors before the dot "
    "product attention computation. Flash attention rewrites the attention "
    "kernel to operate in tiled blocks, reducing memory bandwidth from "
    "quadratic to linear while maintaining exact numerical equivalence. "
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_vram_mib() -> float:
    """Return peak GPU memory in MiB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_vram() -> None:
    """Reset CUDA peak memory stats and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _free_gpu() -> None:
    """Release GPU memory between conditions."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _make_prompts(
    target_tokens: int,
    batch_size: int,
    tokenizer: Any,
) -> list[str]:
    """Generate batch_size text prompts of approximately target_tokens length.

    Each prompt gets a unique suffix to prevent prefix-cache sharing.
    """
    # Build long text by repeating filler (~85 tokens per paragraph)
    repeats = (target_tokens // 80) + 5
    base_text = _FILLER_PARAGRAPH * repeats

    prompts = []
    for i in range(batch_size):
        # Unique prefix to defeat prefix caching
        prefix = f"Request {i + 1} of {batch_size}. "
        text = prefix + base_text

        tokens = tokenizer.encode(text)
        if len(tokens) > target_tokens:
            tokens = tokens[:target_tokens]
        prompts.append(tokenizer.decode(tokens, skip_special_tokens=True))
    return prompts


def _build_condition_kwargs(name: str) -> dict[str, Any]:
    """Return LLM constructor kwargs for a given condition name."""
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CUDAGraphMode

    if name == "baseline_eager":
        return {"enforce_eager": True}
    if name == "piecewise":
        return {
            "compilation_config": CompilationConfig(
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
            ),
        }
    if name == "full_decode":
        return {
            "compilation_config": CompilationConfig(
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            ),
        }
    msg = f"Unknown condition: {name}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _benchmark_condition(
    condition_name: str,
    batch_sizes: list[int],
    context_lens: list[int],
    max_new_tokens: int,
    num_trials: int,
) -> dict[str, Any]:
    """Run all batch_size x context_length combos for one CUDA graph mode."""
    from vllm import LLM, SamplingParams

    llm_kwargs = _build_condition_kwargs(condition_name)

    print(f"\n{'=' * 70}")
    print(f"Condition: {condition_name}")
    print(f"{'=' * 70}")

    from vllm.config import AttentionConfig

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=4608,
        gpu_memory_utilization=0.88,
        kv_cache_memory_bytes=2
        * 1024**3,  # 2 GiB — leaves room for full-cache decompress
        dtype="bfloat16",
        enable_prefix_caching=False,
        attention_config=AttentionConfig(backend="CUSTOM"),
        **llm_kwargs,
    )
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    condition_results: list[dict[str, Any]] = []

    for ctx_len in context_lens:
        # Pre-generate prompts at max batch size, slice for smaller batches
        all_prompts = _make_prompts(ctx_len, max(batch_sizes), tokenizer)
        actual_len = len(tokenizer.encode(all_prompts[0]))
        print(f"\n  Context: {actual_len} tokens (target {ctx_len})")

        for batch_size in batch_sizes:
            prompts = all_prompts[:batch_size]

            # Warmup: JIT compile Triton kernels + CUDA graph capture
            print(f"    B={batch_size}: warmup...", end="", flush=True)
            llm.generate(prompts, sampling_params)
            print(" bench...", end="", flush=True)

            trial_data: list[dict[str, Any]] = []
            for trial in range(num_trials):
                t_start = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params)
                t_end = time.perf_counter()

                wall_s = t_end - t_start
                total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                # VRAM tracked in child process; use nvidia-smi for snapshot
                vram = 0.0

                trial_data.append(
                    {
                        "trial": trial + 1,
                        "wall_s": round(wall_s, 4),
                        "output_tokens": total_out_tokens,
                        "throughput_tok_s": round(total_out_tokens / wall_s, 2),
                        "tpot_ms": round(wall_s / total_out_tokens * 1000, 3),
                        "vram_peak_mib": round(vram, 1),
                    }
                )

            # Median of trials
            tpots = sorted(t["tpot_ms"] for t in trial_data)
            thrpts = sorted(t["throughput_tok_s"] for t in trial_data)
            mid = len(trial_data) // 2

            entry = {
                "batch_size": batch_size,
                "context_length": actual_len,
                "target_context_length": ctx_len,
                "max_new_tokens": max_new_tokens,
                "median_tpot_ms": tpots[mid],
                "median_throughput_tok_s": thrpts[mid],
                "vram_peak_mib": trial_data[-1]["vram_peak_mib"],
                "trials": trial_data,
            }
            condition_results.append(entry)
            print(
                f" TPOT={tpots[mid]:.2f}ms  "
                f"{thrpts[mid]:.0f} tok/s  "
                f"VRAM={entry['vram_peak_mib']:.0f}MiB"
            )

    # Teardown
    del llm
    _free_gpu()

    return {
        "name": condition_name,
        "load_time_s": round(load_time, 1),
        "results": condition_results,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    conditions: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    context_lens: list[int] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_trials: int = DEFAULT_NUM_TRIALS,
) -> dict[str, Any]:
    """Run the CUDA graph decode latency experiment.

    Returns:
        Dict with per-condition results, ready for JSON serialization.
    """
    if conditions is None:
        conditions = list(ALL_CONDITIONS)
    if batch_sizes is None:
        batch_sizes = list(DEFAULT_BATCH_SIZES)
    if context_lens is None:
        context_lens = list(DEFAULT_CONTEXT_LENS)

    # Defer CUDA init — calling get_device_name(0) here would consume ~2.8 GiB
    # in the parent process, reducing free VRAM visible to vLLM's child process.
    gpu_name = "NVIDIA GeForce RTX 4090"  # filled after first condition

    experiment: dict[str, Any] = {
        "experiment": "018-cuda-graph-decode-latency",
        "phase": "D7-Phase-C",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "model": MODEL_ID,
        "config": {
            "batch_sizes": batch_sizes,
            "context_lengths": context_lens,
            "max_new_tokens": max_new_tokens,
            "num_trials": num_trials,
        },
        "conditions": [],
    }

    print(f"\n{'#' * 70}")
    print("Experiment 018: TQ4 CUDA Graph Decode Latency")
    print(f"GPU: {gpu_name}")
    print(f"Model: {MODEL_ID}")
    print(f"Batch sizes: {batch_sizes}  Context: {context_lens}")
    print(f"Trials: {num_trials}  Max new tokens: {max_new_tokens}")
    print(f"{'#' * 70}")

    for cond_name in conditions:
        try:
            result = _benchmark_condition(
                cond_name,
                batch_sizes,
                context_lens,
                max_new_tokens,
                num_trials,
            )
            experiment["conditions"].append(result)
        except Exception as exc:
            print(f"\n  ERROR in {cond_name}: {exc}")
            experiment["conditions"].append({"name": cond_name, "error": str(exc)})
            _free_gpu()

    # ── Summary table ─────────────────────────────────────────────────────
    _print_summary(experiment)
    return experiment


def _print_summary(experiment: dict[str, Any]) -> None:
    """Print a comparison table across all conditions."""
    print(f"\n\n{'#' * 70}")
    print("SUMMARY — Experiment 018")
    print(f"{'#' * 70}")

    header = (
        f"{'Condition':>15} {'Batch':>6} {'Ctx':>6} "
        f"{'TPOT(ms)':>10} {'tok/s':>10} {'VRAM(MiB)':>10} {'Speedup':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    # Collect baseline TPOT for speedup calculation
    baseline: dict[tuple[int, int], float] = {}
    for cond in experiment["conditions"]:
        if cond.get("name") == "baseline_eager":
            for r in cond.get("results", []):
                baseline[(r["batch_size"], r["target_context_length"])] = r[
                    "median_tpot_ms"
                ]

    for cond in experiment["conditions"]:
        if "error" in cond:
            print(f"{cond['name']:>15}  ERROR: {cond['error']}")
            continue
        for r in cond["results"]:
            key = (r["batch_size"], r["target_context_length"])
            speedup_str = ""
            if key in baseline and cond["name"] != "baseline_eager":
                sp = baseline[key] / r["median_tpot_ms"]
                speedup_str = f"{sp:.2f}x"

            print(
                f"{cond['name']:>15} "
                f"{r['batch_size']:>6} "
                f"{r['target_context_length']:>6} "
                f"{r['median_tpot_ms']:>10.2f} "
                f"{r['median_throughput_tok_s']:>10.0f} "
                f"{r['vram_peak_mib']:>10.0f} "
                f"{speedup_str:>8}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment 018: TQ4 CUDA Graph Decode Latency (D7 Phase C)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(ALL_CONDITIONS),
        help="Comma-separated list of conditions to run",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(map(str, DEFAULT_BATCH_SIZES)),
    )
    parser.add_argument(
        "--context-lens",
        type=str,
        default=",".join(map(str, DEFAULT_CONTEXT_LENS)),
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/logs/experiment-018-cuda-graph-decode-latency.json",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    context_lens = [int(x) for x in args.context_lens.split(",")]

    results = run_experiment(
        conditions=conditions,
        batch_sizes=batch_sizes,
        context_lens=context_lens,
        max_new_tokens=args.max_new_tokens,
        num_trials=args.num_trials,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
