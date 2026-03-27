r"""Experiment 010 -- Fused TQ4 K+V kernel E2E Molmo2-4B validation.

Phase 4 gate: validates the fused TQ4 kernel across all 36 transformer
layers via the AttentionInterface + cache side-channel integration.
Tests the existential risk: does 36-layer composition stay above 0.93
cosine similarity, or does precision drift accumulate like the Q@K^T-only
kernel (0.43 over 36 layers)?

Three comparison paths:
    1. SDPA baseline (fp16 KV) — ground truth.
    2. Unfused TQ4 (CompressedDynamicCache + SDPA) — quantization quality.
    3. Fused TQ4 K+V (CompressedDynamicCache + fused kernel) — kernel + quant.

Usage:
    ```bash
    uv run python experiments/experiment_010_fused_tq4_kv_e2e.py
    uv run python experiments/experiment_010_fused_tq4_kv_e2e.py --skip-image
    ```

Outputs JSON results to ``experiments/logs/experiment-010-fused-tq4-kv-e2e.json``.

Examples:
    ```bash
    uv run python experiments/experiment_010_fused_tq4_kv_e2e.py
    ```

See Also:
    :mod:`turboquant_vllm.triton.attention_interface`: install_fused_tq4_kv.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

_CLIP_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "molmo-video-analyzer"
    / "data"
    / "tv"
    / "clip01.mp4"
)


def _get_vram_mb() -> float:
    """Return peak GPU memory in MiB.

    Returns:
        Peak VRAM in MiB, or 0 if no CUDA.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_vram() -> None:
    """Reset CUDA peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _extract_frame(video_path: Path) -> Any:
    """Extract first frame from video as PIL Image.

    Returns:
        PIL Image of the first frame.

    Raises:
        RuntimeError: If no frames found.
    """
    import av
    from PIL import Image

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        img: Image.Image = frame.to_image()
        container.close()
        return img
    container.close()
    msg = f"No frames in {video_path}"
    raise RuntimeError(msg)


def _run_inference(
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int,
    label: str,
    image: Any = None,
) -> dict[str, Any]:
    """Run inference and collect metrics.

    Returns:
        Dict with output text, token IDs, VRAM, timing.
    """
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }
    input_len = inputs["input_ids"].shape[-1]

    _reset_vram()
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
        f"  [{label}] {input_len} in, {output_len} out, "
        f"{tok_s:.1f} tok/s, {vram_peak:.0f} MiB, {elapsed:.1f}s"
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


def _compare(base: dict[str, Any], exp: dict[str, Any], label: str) -> dict[str, Any]:
    """Compare two inference results.

    Returns:
        Dict with match rate, text comparison, throughput ratio.
    """
    b_ids = base["output_token_ids"]
    e_ids = exp["output_token_ids"]
    n = min(len(b_ids), len(e_ids))
    matching = sum(1 for a, b in zip(b_ids[:n], e_ids[:n]) if a == b)
    match_rate = matching / n if n > 0 else 0.0
    texts_match = base["output_text"].strip() == exp["output_text"].strip()
    throughput = exp["tok_per_s"] / base["tok_per_s"] if base["tok_per_s"] > 0 else 0.0

    print(
        f"  [{label}] {matching}/{n} tokens ({match_rate:.1%}), "
        f"text={'same' if texts_match else 'DIFF'}, "
        f"throughput {throughput:.2f}x"
    )

    if not texts_match:
        print(f"    base:  {base['output_text'][:80]}...")
        print(f"    exp:   {exp['output_text'][:80]}...")

    return {
        "label": label,
        "tokens_compared": n,
        "tokens_matching": matching,
        "match_rate": round(match_rate, 4),
        "texts_identical": texts_match,
        "throughput_ratio": round(throughput, 3),
    }


def run_experiment(
    model_id: str,
    text_prompt: str,
    image_prompt: str,
    max_new_tokens: int,
    bits: int,
    clip_path: Path,
    skip_image: bool,
) -> dict[str, Any]:
    """Run the full 3-path comparison experiment.

    Returns:
        Dict with all results and gate assessment.
    """
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        DynamicCache,
    )

    from turboquant_vllm.kv_cache import CompressedDynamicCache
    from turboquant_vllm.triton.attention_interface import (
        install_fused_tq4_kv,
        uninstall_fused_tq4_kv,
    )

    results: dict[str, Any] = {
        "experiment": "010-fused-tq4-kv-e2e",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "bits": bits,
        "max_new_tokens": max_new_tokens,
    }

    # Load model
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _reset_vram()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    text_config = getattr(model.config, "text_config", model.config)
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    results["model_config"] = {
        "num_layers": text_config.num_hidden_layers,
        "num_heads": text_config.num_attention_heads,
        "num_kv_heads": getattr(
            text_config,
            "num_key_value_heads",
            text_config.num_attention_heads,
        ),
        "head_dim": head_dim,
        "attn_impl": model.config._attn_implementation,
    }
    print(f"  {results['model_config']}")

    # ---- Helper to run all 3 paths on one prompt ----
    def run_3_paths(prompt: str, label: str, image: Any = None) -> dict[str, Any]:
        """Run SDPA, unfused TQ4, and fused TQ4 on one prompt.

        Returns:
            Dict with results for each path and comparisons.
        """
        # Path 1: SDPA baseline
        sdpa = _run_inference(
            model, processor, prompt, max_new_tokens, f"{label}-SDPA", image
        )

        # Path 2: Unfused TQ4 (CompressedDynamicCache + SDPA)
        original_init = DynamicCache.__init__
        wrappers: list[CompressedDynamicCache] = []

        def patched_init(self_cache: Any, *a: Any, **kw: Any) -> None:
            """Wrap DynamicCache with TQ4 compression.

            Other Parameters:
                **kw: Forwarded to original init.
            """
            original_init(self_cache, *a, **kw)
            w = CompressedDynamicCache(self_cache, head_dim=head_dim, bits=bits)
            wrappers.append(w)

        DynamicCache.__init__ = patched_init  # type: ignore[method-assign]
        try:
            unfused = _run_inference(
                model,
                processor,
                prompt,
                max_new_tokens,
                f"{label}-UnfusedTQ4",
                image,
            )
        finally:
            DynamicCache.__init__ = original_init  # type: ignore[method-assign]

        # Path 3: Fused TQ4 K+V (cache side-channel)
        fused_wrappers: list[CompressedDynamicCache] = []

        def fused_patched_init(self_cache: Any, *a: Any, **kw: Any) -> None:
            """Wrap DynamicCache and install fused kernel.

            Other Parameters:
                **kw: Forwarded to original init.
            """
            original_init(self_cache, *a, **kw)
            w = CompressedDynamicCache(self_cache, head_dim=head_dim, bits=bits)
            fused_wrappers.append(w)
            install_fused_tq4_kv(model, w)

        DynamicCache.__init__ = fused_patched_init  # type: ignore[method-assign]
        try:
            fused = _run_inference(
                model,
                processor,
                prompt,
                max_new_tokens,
                f"{label}-FusedTQ4KV",
                image,
            )
        finally:
            DynamicCache.__init__ = original_init  # type: ignore[method-assign]
            uninstall_fused_tq4_kv(model)

        # Comparisons
        print()
        c_unfused = _compare(sdpa, unfused, f"{label} SDPA→Unfused")
        c_fused = _compare(sdpa, fused, f"{label} SDPA→Fused")
        c_fused_vs_unfused = _compare(unfused, fused, f"{label} Unfused→Fused")

        return {
            "sdpa": sdpa,
            "unfused_tq4": unfused,
            "fused_tq4_kv": fused,
            "sdpa_vs_unfused": c_unfused,
            "sdpa_vs_fused": c_fused,
            "unfused_vs_fused": c_fused_vs_unfused,
        }

    # ---- Run experiments ----
    print("\n" + "=" * 60)
    print("TEXT-ONLY")
    print("=" * 60)
    results["text"] = run_3_paths(text_prompt, "text")

    if not skip_image and clip_path.exists():
        print("\n" + "=" * 60)
        print("IMAGE (Seinfeld clip01)")
        print("=" * 60)
        image = _extract_frame(clip_path)
        print(f"  Frame: {image.size[0]}x{image.size[1]}")
        results["image"] = run_3_paths(image_prompt, "image", image)
    else:
        results["image"] = {"status": "skipped"}

    # ---- Gate assessment ----
    print("\n" + "=" * 60)
    print("P5 PHASE 4 GATE ASSESSMENT")
    print("=" * 60)

    text_fvsu = results["text"]["unfused_vs_fused"]
    gate: dict[str, Any] = {
        "fused_vs_unfused_text_match": text_fvsu["match_rate"],
        "fused_vs_unfused_text_identical": text_fvsu["texts_identical"],
    }

    if "unfused_vs_fused" in results.get("image", {}):
        img_fvsu = results["image"]["unfused_vs_fused"]
        gate["fused_vs_unfused_image_match"] = img_fvsu["match_rate"]

    # The decisive gate: fused must match unfused (same quantization,
    # different attention path). Divergence = kernel bug.
    text_pass = text_fvsu["texts_identical"]
    gate["overall"] = "PASS" if text_pass else "FAIL"

    for k, v in gate.items():
        print(f"  {k}: {v}")

    results["gate"] = gate
    return results


def main() -> None:
    """CLI entry point for Experiment 010."""
    parser = argparse.ArgumentParser(
        description="Experiment 010: Fused TQ4 K+V E2E Molmo2 validation"
    )
    parser.add_argument(
        "--model",
        default="allenai/Molmo2-4B",
    )
    parser.add_argument(
        "--text-prompt",
        default="Describe the main character of Seinfeld in one paragraph.",
    )
    parser.add_argument(
        "--image-prompt",
        default="Describe what is happening in this image in detail.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--clip",
        type=Path,
        default=_CLIP_PATH,
    )
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument(
        "--output",
        default="experiments/logs/experiment-010-fused-tq4-kv-e2e.json",
    )
    args = parser.parse_args()

    results = run_experiment(
        model_id=args.model,
        text_prompt=args.text_prompt,
        image_prompt=args.image_prompt,
        max_new_tokens=args.max_new_tokens,
        bits=args.bits,
        clip_path=args.clip,
        skip_image=args.skip_image,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    sys.exit(0 if results.get("gate", {}).get("overall") == "PASS" else 1)


if __name__ == "__main__":
    main()
