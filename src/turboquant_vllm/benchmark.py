r"""Benchmark harness for TurboQuant KV cache compression on Molmo2.

Loads Molmo2 via HuggingFace transformers and runs inference with:
1. Baseline: standard DynamicCache (no compression)
2. TurboQuant: accuracy-only or compressed mode

Two modes:

- **Accuracy-only** (default): TurboQuantKVCache compresses then
  immediately decompresses. Measures quality impact, no VRAM savings.
- **Compressed** (``--compressed``): CompressedDynamicCache stores uint8
  indices + fp16 norms. Measures real VRAM savings.

Measures output text diff, VRAM peak, generation time, and (in compressed
mode) KV cache compression statistics.

Usage:
    ```bash
    # Accuracy-only mode
    python -m turboquant_vllm.benchmark \
        --model allenai/Molmo2-4B --bits 3

    # Compressed mode (real VRAM savings)
    python -m turboquant_vllm.benchmark \
        --model allenai/Molmo2-4B --bits 3 --compressed \
        --video /path/to/clip.mp4
    ```

Requires GPU with sufficient VRAM for the chosen model.

Examples:
    ```python
    from turboquant_vllm.benchmark import run_benchmark
    results = run_benchmark("allenai/Molmo2-4B", "Describe the scene.", bits=3)
    ```

See Also:
    :class:`turboquant_vllm.TurboQuantKVCache`: Accuracy-only cache wrapper.
    :class:`turboquant_vllm.CompressedDynamicCache`: Compressed cache with VRAM savings.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


def _get_vram_mb() -> float:
    """Get current GPU VRAM usage in MiB.

    Returns:
        VRAM allocated in MiB, or 0.0 if CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_vram_tracking() -> None:
    """Reset CUDA peak memory tracking for a fresh measurement."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def load_model(
    model_id: str,
) -> tuple[Any, Any]:
    """Load a Molmo2 model and processor from HuggingFace.

    Args:
        model_id: HuggingFace model identifier (e.g., 'allenai/Molmo2-8B').

    Returns:
        Tuple of (model, processor) ready for inference.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading model from {model_id} (bfloat16)...")
    _reset_vram_tracking()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_vram = _get_vram_mb()
    print(f"Model loaded: {model_vram:.0f} MiB VRAM")

    return model, processor


def run_inference(
    model: Any,
    processor: Any,
    prompt: str,
    video_path: str | None = None,
    max_new_tokens: int = 256,
) -> tuple[str, float, float]:
    """Run a single inference pass and measure performance.

    Args:
        model: Loaded Molmo2 model.
        processor: Loaded Molmo2 processor.
        prompt: Text prompt for the model.
        video_path: Optional path to a video file.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (output_text, vram_peak_mib, elapsed_seconds).
    """
    # Build messages in chat format
    content = [{"type": "text", "text": prompt}]
    if video_path:
        content.insert(0, {"type": "video", "video": video_path})

    messages = [{"role": "user", "content": content}]

    # Tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]
    print(f"  Input tokens: {input_len}")

    # Generate
    _reset_vram_tracking()
    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    elapsed = time.perf_counter() - start
    vram_peak = _get_vram_mb()

    # Decode output (skip input tokens)
    generated_ids = output_ids[0, input_len:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)

    output_len = len(generated_ids)
    tokens_per_sec = output_len / elapsed if elapsed > 0 else 0
    print(
        f"  Output tokens: {output_len}, {tokens_per_sec:.1f} tok/s, "
        f"VRAM peak: {vram_peak:.0f} MiB, time: {elapsed:.1f}s"
    )

    return output_text, vram_peak, elapsed


def _detect_model_config(model: Any) -> dict[str, int]:
    """Extract KV cache parameters from a model's config.

    Args:
        model: A loaded HuggingFace model.

    Returns:
        Dict with head_dim, num_heads, num_kv_heads, num_layers.
    """
    config = model.config
    # Molmo2 wraps a text config inside the main config
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


def _patch_cache(
    cache_cls: type,
    wrapper_cls: type,
    head_dim: int,
    bits: int,
) -> tuple[Any, list]:
    """Monkey-patch DynamicCache.__init__ to inject a TurboQuant wrapper.

    Args:
        cache_cls: The DynamicCache class to patch.
        wrapper_cls: The wrapper class (TurboQuantKVCache or
            CompressedDynamicCache).
        head_dim: Model head dimension.
        bits: Quantization bits per coordinate.

    Returns:
        Tuple of (original_init, wrappers_list) where wrappers_list
        collects wrapper instances for post-run inspection.
    """
    original_init = cache_cls.__init__
    wrappers: list = []

    def patched_init(self_cache: Any, *args: Any, **kwargs: Any) -> None:
        """Wrap a new DynamicCache with TurboQuant compression.

        Other Parameters:
            **kwargs: Forwarded to the original ``DynamicCache.__init__``.
        """
        original_init(self_cache, *args, **kwargs)
        wrapper = wrapper_cls(self_cache, head_dim=head_dim, bits=bits)
        wrappers.append(wrapper)

    setattr(cache_cls, "__init__", patched_init)  # noqa: B010
    return original_init, wrappers


def run_benchmark(
    model_id: str,
    prompt: str,
    video_path: str | None = None,
    bits: int = 3,
    max_new_tokens: int = 256,
    *,
    compressed: bool = False,
) -> dict:
    """Run baseline vs TurboQuant comparison benchmark.

    Args:
        model_id: HuggingFace model identifier.
        prompt: Text prompt for inference.
        video_path: Optional path to a video file.
        bits: TurboQuant bits per coordinate.
        max_new_tokens: Maximum tokens to generate.
        compressed: If True, benchmark CompressedDynamicCache (real VRAM
            savings). If False, benchmark TurboQuantKVCache (accuracy only).

    Returns:
        Dict with benchmark results including both runs and comparison metrics.
    """
    from transformers import DynamicCache

    from turboquant_vllm.kv_cache import (
        CompressedDynamicCache,
        TurboQuantKVCache,
    )

    model, processor = load_model(model_id)
    model_cfg = _detect_model_config(model)
    head_dim = model_cfg["head_dim"]

    print(
        f"Model config: {model_cfg['num_layers']} layers, "
        f"{model_cfg['num_heads']} heads, "
        f"{model_cfg['num_kv_heads']} KV heads, head_dim={head_dim}"
    )

    results: dict = {
        "model_id": model_id,
        "bits": bits,
        "mode": "compressed" if compressed else "accuracy_only",
        **model_cfg,
        "prompt": prompt,
        "video_path": video_path,
    }

    # --- Baseline run (no compression) ---
    print("\n=== BASELINE (no compression) ===")
    baseline_text, baseline_vram, baseline_time = run_inference(
        model, processor, prompt, video_path, max_new_tokens
    )
    results["baseline"] = {
        "output_text": baseline_text,
        "vram_peak_mib": baseline_vram,
        "elapsed_s": baseline_time,
    }

    # --- TurboQuant run ---
    wrapper_cls = CompressedDynamicCache if compressed else TurboQuantKVCache
    mode_label = "COMPRESSED" if compressed else "ACCURACY-ONLY"
    print(f"\n=== TURBOQUANT TQ{bits} K+V ({mode_label}) ===")

    original_init, wrappers = _patch_cache(DynamicCache, wrapper_cls, head_dim, bits)

    try:
        tq_text, tq_vram, tq_time = run_inference(
            model, processor, prompt, video_path, max_new_tokens
        )
    finally:
        setattr(DynamicCache, "__init__", original_init)  # noqa: B010

    tq_results: dict[str, Any] = {
        "output_text": tq_text,
        "vram_peak_mib": tq_vram,
        "elapsed_s": tq_time,
    }

    # Report compression stats if using CompressedDynamicCache
    if compressed and wrappers:
        stats = wrappers[-1].compression_stats()
        tq_results["compression_stats"] = stats
        if stats:
            print(f"  KV cache compression: {stats['compression_ratio']}x")
            print(
                f"  Compressed: {stats['compressed_mib']:.1f} MiB vs "
                f"baseline: {stats['baseline_mib']:.1f} MiB "
                f"(saved {stats['savings_mib']:.1f} MiB)"
            )

    results["turboquant"] = tq_results

    # --- Comparison ---
    texts_match = baseline_text.strip() == tq_text.strip()
    time_ratio = tq_time / baseline_time if baseline_time > 0 else 0
    vram_delta = baseline_vram - tq_vram

    results["comparison"] = {
        "texts_identical": texts_match,
        "time_ratio": round(time_ratio, 3),
        "vram_delta_mib": round(vram_delta, 1),
        "baseline_text_preview": baseline_text[:200],
        "turboquant_text_preview": tq_text[:200],
    }

    print("\n=== COMPARISON ===")
    print(f"  Texts identical: {texts_match}")
    print(f"  Time ratio (TQ/baseline): {time_ratio:.2f}x")
    print(f"  VRAM delta: {vram_delta:+.0f} MiB")
    if not texts_match:
        print(f"  Baseline: {baseline_text[:100]}...")
        print(f"  TQ{bits}:     {tq_text[:100]}...")

    return results


def main() -> None:
    """CLI entry point for the benchmark harness."""
    parser = argparse.ArgumentParser(
        description="Benchmark TurboQuant KV cache compression on Molmo2"
    )
    parser.add_argument(
        "--model",
        default="allenai/Molmo2-8B",
        help="HuggingFace model ID (default: allenai/Molmo2-8B)",
    )
    parser.add_argument(
        "--prompt",
        default="Describe what you see in detail.",
        help="Text prompt for inference",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to video file (optional — text-only if omitted)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=3,
        help="TurboQuant bits per coordinate (default: 3)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        default=False,
        help="Use CompressedDynamicCache for real VRAM savings (default: accuracy-only mode)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results (default: stdout)",
    )

    args = parser.parse_args()

    results = run_benchmark(
        model_id=args.model,
        prompt=args.prompt,
        video_path=args.video,
        bits=args.bits,
        max_new_tokens=args.max_new_tokens,
        compressed=args.compressed,
    )

    output_json = json.dumps(results, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output_json)
        print(f"\nResults saved to {args.output}")
    else:
        print(f"\n{output_json}")


if __name__ == "__main__":
    main()
