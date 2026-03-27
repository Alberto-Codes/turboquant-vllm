r"""Experiment 007 — End-to-end Molmo2-4B validation on AMD Radeon 890M.

Phase 3 of P7 (ROCm platform support): validates that TurboQuant
compressed inference produces correct output on AMD iGPU hardware.

Five validation steps:
    3.1  Verify model weights are accessible (fail-fast).
    3.2  GPU baseline inference (no compression), greedy decode.
    3.3  GPU TQ4 compressed inference, compare output to baseline.
    3.4  CPU vs GPU cross-validation (short generation).
    3.5  Throughput report (tok/s, VRAM, compression stats).

Usage (inside ROCm container via run-rocm.sh):
    ```bash
    ./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py
    ./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py \
        --model allenai/Molmo2-4B --bits 4 --max-new-tokens 64
    ```

Outputs JSON results to experiments/logs/experiment-007-e2e-amd-validation.json
and a human-readable summary to stdout.

Examples:
    ```bash
    ./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py
    ```

See Also:
    :mod:`turboquant_vllm.benchmark`: Production benchmark harness.
    :mod:`turboquant_vllm.kv_cache`: CompressedDynamicCache implementation.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_vram_tracking() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / (1024**3), 1)
        info["gcn_arch"] = getattr(props, "gcnArchName", "unknown")
    return info


def _detect_model_config(model: Any) -> dict[str, int]:
    config = model.config
    text_config = getattr(config, "text_config", config)
    hidden_size = text_config.hidden_size
    num_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", hidden_size // num_heads)
    num_kv_heads = getattr(text_config, "num_key_value_heads", num_heads)
    num_layers = text_config.num_hidden_layers
    return {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
    }


# ---------------------------------------------------------------------------
# Step 3.1 — Verify model weights
# ---------------------------------------------------------------------------


def step_3_1_verify_weights(model_id: str) -> dict[str, Any]:
    """Check that model weights are downloadable / cached locally.

    Returns:
        Dict with 'status' ('ok' or 'fail') and model metadata.
    """
    from huggingface_hub import model_info

    print("\n" + "=" * 60)
    print("STEP 3.1 — Verify model weights")
    print("=" * 60)

    try:
        info = model_info(model_id)
        size_gb = (info.safetensors.total if info.safetensors else 0) / (1024**3)
        print(f"  Model: {model_id}")
        print(f"  Size:  {size_gb:.1f} GB")
        print("  OK — model accessible")
        return {"status": "ok", "model_id": model_id, "size_gb": round(size_gb, 1)}
    except Exception as e:
        print(f"  FAIL — {e}")
        print("  Hint: run `huggingface-cli login` inside the container")
        return {"status": "fail", "error": str(e)}


# ---------------------------------------------------------------------------
# Step 3.2 — GPU baseline inference
# ---------------------------------------------------------------------------


def _load_model(model_id: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if device == "cpu":
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
    else:
        _reset_vram_tracking()
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, processor


def _run_inference(
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int,
    label: str,
) -> dict[str, Any]:
    content = [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }
    input_len = inputs["input_ids"].shape[-1]

    _reset_vram_tracking()
    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    elapsed = time.perf_counter() - start
    vram_peak = _get_vram_mb()

    generated_ids = output_ids[0, input_len:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)
    output_len = len(generated_ids)
    tok_s = output_len / elapsed if elapsed > 0 else 0

    print(
        f"  [{label}] {output_len} tokens, {tok_s:.1f} tok/s, "
        f"VRAM peak: {vram_peak:.0f} MiB, time: {elapsed:.1f}s"
    )

    return {
        "label": label,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "output_text": output_text,
        "output_token_ids": generated_ids.tolist(),
        "vram_peak_mib": round(vram_peak, 1),
        "elapsed_s": round(elapsed, 2),
        "tok_per_s": round(tok_s, 2),
    }


def step_3_2_gpu_baseline(
    model: Any, processor: Any, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    """Run baseline (uncompressed) inference on GPU.

    Returns:
        Dict with output text, token IDs, VRAM peak, and timing.
    """
    print("\n" + "=" * 60)
    print("STEP 3.2 — GPU baseline inference (no compression)")
    print("=" * 60)
    return _run_inference(model, processor, prompt, max_new_tokens, "GPU-baseline")


# ---------------------------------------------------------------------------
# Step 3.3 — GPU TQ4 compressed inference
# ---------------------------------------------------------------------------


def step_3_3_gpu_compressed(
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int,
    head_dim: int,
    bits: int,
) -> dict[str, Any]:
    """Run TQ-compressed inference on GPU and report compression stats.

    Returns:
        Dict with output text, token IDs, VRAM peak, timing, and compression stats.
    """
    from transformers import DynamicCache

    from turboquant_vllm.kv_cache import CompressedDynamicCache

    print("\n" + "=" * 60)
    print(f"STEP 3.3 — GPU TQ{bits} compressed inference")
    print("=" * 60)

    # Monkey-patch DynamicCache to inject CompressedDynamicCache
    original_init = DynamicCache.__init__
    wrappers: list[CompressedDynamicCache] = []

    def patched_init(self_cache: Any, *args: Any, **kwargs: Any) -> None:
        """Wrap new DynamicCache instances with TurboQuant compression.

        Other Parameters:
            **kwargs: Forwarded to the original ``DynamicCache.__init__``.
        """
        original_init(self_cache, *args, **kwargs)
        wrapper = CompressedDynamicCache(self_cache, head_dim=head_dim, bits=bits)
        wrappers.append(wrapper)

    DynamicCache.__init__ = patched_init  # type: ignore[method-assign]

    try:
        result = _run_inference(
            model, processor, prompt, max_new_tokens, f"GPU-TQ{bits}"
        )
    finally:
        DynamicCache.__init__ = original_init  # type: ignore[method-assign]

    if wrappers:
        stats = wrappers[-1].compression_stats()
        result["compression_stats"] = stats
        if stats:
            print(
                f"  Compression: {stats['compression_ratio']}x "
                f"({stats['compressed_mib']:.1f} MiB vs "
                f"{stats['baseline_mib']:.1f} MiB baseline)"
            )

    return result


# ---------------------------------------------------------------------------
# Step 3.4 — CPU vs GPU cross-validation
# ---------------------------------------------------------------------------


def step_3_4_cross_validate(
    model_id: str,
    prompt: str,
    cross_validate_tokens: int,
    gpu_result: dict[str, Any],
) -> dict[str, Any]:
    """Run short CPU inference and compare token outputs to GPU baseline.

    Returns:
        Dict with CPU result, token comparison, and match rate.
    """
    print("\n" + "=" * 60)
    print(f"STEP 3.4 — CPU vs GPU cross-validation ({cross_validate_tokens} tokens)")
    print("=" * 60)

    print("  Loading model on CPU (float32)... this will be slow")
    cpu_model, cpu_processor = _load_model(model_id, device="cpu")

    cpu_result = _run_inference(
        cpu_model, cpu_processor, prompt, cross_validate_tokens, "CPU-baseline"
    )

    # Free CPU model to reclaim RAM
    del cpu_model
    import gc

    gc.collect()

    # Compare token IDs
    gpu_ids = gpu_result["output_token_ids"][:cross_validate_tokens]
    cpu_ids = cpu_result["output_token_ids"][:cross_validate_tokens]

    min_len = min(len(gpu_ids), len(cpu_ids))
    matching = sum(1 for a, b in zip(gpu_ids[:min_len], cpu_ids[:min_len]) if a == b)
    match_rate = matching / min_len if min_len > 0 else 0.0

    print(f"\n  Token comparison ({min_len} tokens):")
    print(f"    Matching: {matching}/{min_len} ({match_rate:.1%})")

    if match_rate < 1.0:
        # Find first divergence point
        for i, (g, c) in enumerate(zip(gpu_ids[:min_len], cpu_ids[:min_len])):
            if g != c:
                print(f"    First divergence at token {i}: GPU={g}, CPU={c}")
                break
        print("    Note: CPU/GPU divergence is expected for autoregressive models")
        print("    (bf16 vs fp32 accumulation differences compound across tokens)")

    cross_result: dict[str, Any] = {
        "cpu_result": cpu_result,
        "gpu_tokens_compared": gpu_ids[:min_len],
        "cpu_tokens_compared": cpu_ids[:min_len],
        "tokens_compared": min_len,
        "tokens_matching": matching,
        "match_rate": round(match_rate, 4),
    }

    # Qualitative check: are both outputs coherent/similar?
    print(f"\n  GPU output: {gpu_result['output_text'][:120]}...")
    print(f"  CPU output: {cpu_result['output_text'][:120]}...")

    return cross_result


# ---------------------------------------------------------------------------
# Step 3.5 — Throughput report
# ---------------------------------------------------------------------------


def step_3_5_throughput_report(
    device_info: dict,
    model_config: dict,
    gpu_baseline: dict,
    gpu_compressed: dict,
    cross_validation: dict | None,
) -> dict[str, Any]:
    """Compile final throughput and quality report.

    Returns:
        Dict with throughput, VRAM, overhead, and quality metrics.
    """
    print("\n" + "=" * 60)
    print("STEP 3.5 — Throughput report")
    print("=" * 60)

    baseline_tok_s = gpu_baseline["tok_per_s"]
    compressed_tok_s = gpu_compressed["tok_per_s"]
    overhead = compressed_tok_s / baseline_tok_s if baseline_tok_s > 0 else 0

    texts_match = (
        gpu_baseline["output_text"].strip() == gpu_compressed["output_text"].strip()
    )

    report = {
        "device": device_info,
        "model_config": model_config,
        "baseline_tok_s": baseline_tok_s,
        "compressed_tok_s": compressed_tok_s,
        "compression_overhead": round(overhead, 3),
        "baseline_vram_peak_mib": gpu_baseline["vram_peak_mib"],
        "compressed_vram_peak_mib": gpu_compressed["vram_peak_mib"],
        "texts_identical": texts_match,
    }

    if cross_validation:
        report["cpu_gpu_match_rate"] = cross_validation["match_rate"]

    print(f"\n  Device:              {device_info.get('device_name', 'N/A')}")
    print(
        f"  Model:               {model_config['num_layers']}L / "
        f"{model_config['num_heads']}H / head_dim={model_config['head_dim']}"
    )
    print(
        f"  Baseline:            {baseline_tok_s:.1f} tok/s, "
        f"{gpu_baseline['vram_peak_mib']:.0f} MiB peak"
    )
    print(
        f"  Compressed:          {compressed_tok_s:.1f} tok/s, "
        f"{gpu_compressed['vram_peak_mib']:.0f} MiB peak"
    )
    print(f"  Overhead:            {overhead:.2f}x")
    print(f"  Texts identical:     {texts_match}")

    if not texts_match:
        print(f"  Baseline output:     {gpu_baseline['output_text'][:80]}...")
        print(f"  Compressed output:   {gpu_compressed['output_text'][:80]}...")

    comp_stats = gpu_compressed.get("compression_stats", {})
    if comp_stats:
        print(
            f"  KV compression:      {comp_stats['compression_ratio']}x "
            f"({comp_stats['bits']}-bit"
            f"{', nibble-packed' if comp_stats.get('nibble_packed') else ''})"
        )

    if cross_validation:
        print(f"  CPU/GPU match rate:  {cross_validation['match_rate']:.1%}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(
    model_id: str,
    prompt: str,
    bits: int = 4,
    max_new_tokens: int = 64,
    cross_validate_tokens: int = 16,
    skip_cross_validate: bool = False,
) -> dict[str, Any]:
    """Run the full Phase 3 end-to-end validation experiment.

    Returns:
        Dict with all step results keyed as step_3_1 through step_3_5.
    """
    results: dict[str, Any] = {
        "experiment": "007-e2e-amd-validation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "bits": bits,
        "max_new_tokens": max_new_tokens,
        "prompt": prompt,
    }

    # Device info
    results["device"] = _device_info()
    if not results["device"]["cuda_available"]:
        print("WARNING: CUDA not available — running CPU-only (no GPU validation)")

    # Step 3.1 — Verify weights
    results["step_3_1"] = step_3_1_verify_weights(model_id)
    if results["step_3_1"]["status"] == "fail":
        print("\nABORT: Cannot access model weights. Fix auth and retry.")
        return results

    # Load model on GPU
    print("\nLoading model on GPU...")
    model, processor = _load_model(model_id, device="cuda")
    model_config = _detect_model_config(model)
    results["model_config"] = model_config
    head_dim = model_config["head_dim"]

    print(
        f"  {model_config['num_layers']} layers, "
        f"{model_config['num_heads']} heads, "
        f"{model_config['num_kv_heads']} KV heads, head_dim={head_dim}"
    )

    # Step 3.2 — GPU baseline
    results["step_3_2"] = step_3_2_gpu_baseline(
        model, processor, prompt, max_new_tokens
    )

    # Step 3.3 — GPU compressed
    results["step_3_3"] = step_3_3_gpu_compressed(
        model, processor, prompt, max_new_tokens, head_dim, bits
    )

    # Free GPU model before CPU loading
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3.4 — CPU vs GPU cross-validation
    if skip_cross_validate:
        print("\n  Skipping CPU cross-validation (--skip-cross-validate)")
        results["step_3_4"] = {"status": "skipped"}
    else:
        results["step_3_4"] = step_3_4_cross_validate(
            model_id, prompt, cross_validate_tokens, results["step_3_2"]
        )

    # Step 3.5 — Throughput report
    cross_val = results["step_3_4"] if not skip_cross_validate else None
    results["step_3_5"] = step_3_5_throughput_report(
        results["device"],
        model_config,
        results["step_3_2"],
        results["step_3_3"],
        cross_val,
    )

    return results


def main() -> None:
    """CLI entry point for the Phase 3 validation experiment."""
    parser = argparse.ArgumentParser(
        description="Experiment 007: End-to-end Molmo2 validation on AMD ROCm"
    )
    parser.add_argument(
        "--model",
        default="allenai/Molmo2-4B",
        help="HuggingFace model ID (default: allenai/Molmo2-4B)",
    )
    parser.add_argument(
        "--prompt",
        default="Describe the main character of Seinfeld in one paragraph.",
        help="Text prompt for inference",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="TurboQuant bits per coordinate (default: 4)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate for GPU runs (default: 64)",
    )
    parser.add_argument(
        "--cross-validate-tokens",
        type=int,
        default=16,
        help="Max tokens for CPU cross-validation (default: 16)",
    )
    parser.add_argument(
        "--skip-cross-validate",
        action="store_true",
        help="Skip the slow CPU vs GPU cross-validation step",
    )
    parser.add_argument(
        "--output",
        default="experiments/logs/experiment-007-e2e-amd-validation.json",
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    results = run_experiment(
        model_id=args.model,
        prompt=args.prompt,
        bits=args.bits,
        max_new_tokens=args.max_new_tokens,
        cross_validate_tokens=args.cross_validate_tokens,
        skip_cross_validate=args.skip_cross_validate,
    )

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    # Exit code: 0 if baseline ran, 1 if aborted
    sys.exit(0 if "step_3_2" in results else 1)


if __name__ == "__main__":
    main()
