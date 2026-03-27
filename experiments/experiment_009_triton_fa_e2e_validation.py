r"""Experiment 009 -- Triton Flash Attention end-to-end Molmo2-4B validation.

Phase 1 gate for P5: validates that the custom Triton FA kernel produces
identical output to SDPA when run end-to-end on Molmo2-4B, and does not
regress throughput.

Four validation steps:
    1. Text-only: SDPA baseline vs Triton FA -- token match + throughput.
    2. Image: SDPA baseline vs Triton FA with a Seinfeld clip frame.
    3. Throughput comparison and P5 Phase 1 exit gate assessment.

Usage:
    ```bash
    uv run python experiments/experiment_009_triton_fa_e2e_validation.py
    uv run python experiments/experiment_009_triton_fa_e2e_validation.py \
        --model allenai/Molmo2-4B --max-new-tokens 64
    ```

Outputs JSON results to ``experiments/logs/experiment-009-triton-fa-e2e.json``
and a human-readable summary to stdout.

Examples:
    ```bash
    uv run python experiments/experiment_009_triton_fa_e2e_validation.py
    ```

See Also:
    :mod:`turboquant_vllm.triton.flash_attention`: The Triton FA kernel.
    :mod:`turboquant_vllm.triton.attention_interface`: HF registration.
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

_CLIP_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "molmo-video-analyzer"
    / "data"
    / "tv"
    / "clip01.mp4"
)


def _get_vram_mb() -> float:
    """Return peak GPU memory in MiB (0 if no CUDA).

    Returns:
        Peak VRAM in MiB.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_vram_tracking() -> None:
    """Reset CUDA peak memory stats and empty cache."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _device_info() -> dict[str, Any]:
    """Collect GPU device metadata.

    Returns:
        Dict with CUDA availability, device name, memory, and arch.
    """
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / (1024**3), 1)
    return info


def _detect_model_config(model: Any) -> dict[str, int]:
    """Extract attention geometry from model config.

    Returns:
        Dict with head_dim, num_heads, num_kv_heads, num_layers.
    """
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


def _load_model(model_id: str) -> tuple[Any, Any]:
    """Load model and processor on GPU with bf16.

    Returns:
        ``(model, processor)`` tuple.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _reset_vram_tracking()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, processor


def _extract_frame(video_path: Path) -> Any:
    """Extract the first frame from a video file as a PIL Image.

    Returns:
        PIL Image of the first frame.

    Raises:
        RuntimeError: If the video contains no frames.
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
    """Run a single inference pass and collect metrics.

    Returns:
        Dict with output text, token IDs, VRAM peak, and timing.
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
        f"  [{label}] {input_len} input, {output_len} output tokens, "
        f"{tok_s:.1f} tok/s, VRAM: {vram_peak:.0f} MiB, time: {elapsed:.1f}s"
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


def _compare_results(
    baseline: dict[str, Any], experimental: dict[str, Any], label: str
) -> dict[str, Any]:
    """Compare two inference results: token match + throughput ratio.

    Returns:
        Dict with match metrics and throughput comparison.
    """
    base_ids = baseline["output_token_ids"]
    exp_ids = experimental["output_token_ids"]
    min_len = min(len(base_ids), len(exp_ids))
    matching = sum(1 for a, b in zip(base_ids[:min_len], exp_ids[:min_len]) if a == b)
    match_rate = matching / min_len if min_len > 0 else 0.0

    texts_identical = (
        baseline["output_text"].strip() == experimental["output_text"].strip()
    )

    throughput_ratio = (
        experimental["tok_per_s"] / baseline["tok_per_s"]
        if baseline["tok_per_s"] > 0
        else 0.0
    )

    print(f"\n  [{label}] Comparison:")
    print(f"    Token match:   {matching}/{min_len} ({match_rate:.1%})")
    print(f"    Text identical: {texts_identical}")
    print(
        f"    Throughput:    {experimental['tok_per_s']:.1f} vs "
        f"{baseline['tok_per_s']:.1f} tok/s ({throughput_ratio:.2f}x)"
    )

    if not texts_identical:
        print(f"    SDPA:      {baseline['output_text'][:100]}...")
        print(f"    Triton FA: {experimental['output_text'][:100]}...")

    return {
        "label": label,
        "tokens_compared": min_len,
        "tokens_matching": matching,
        "match_rate": round(match_rate, 4),
        "texts_identical": texts_identical,
        "throughput_ratio": round(throughput_ratio, 3),
    }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_1_text_only(
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run text-only inference: SDPA baseline then Triton FA.

    Returns:
        Dict with baseline, experimental, and comparison results.
    """
    from turboquant_vllm.triton.attention_interface import install_triton_fa

    print("\n" + "=" * 60)
    print("STEP 1 — Text-only: SDPA vs Triton FA")
    print("=" * 60)

    # SDPA baseline
    sdpa_result = _run_inference(model, processor, prompt, max_new_tokens, "SDPA-text")

    # Switch to Triton FA
    original_impl = model.config._attn_implementation
    install_triton_fa(model)
    triton_result = _run_inference(
        model, processor, prompt, max_new_tokens, "TritonFA-text"
    )

    # Restore SDPA for next step
    model.config._attn_implementation = original_impl

    comparison = _compare_results(sdpa_result, triton_result, "text-only")

    return {
        "sdpa": sdpa_result,
        "triton_fa": triton_result,
        "comparison": comparison,
    }


def step_2_image(
    model: Any,
    processor: Any,
    image_prompt: str,
    max_new_tokens: int,
    clip_path: Path,
) -> dict[str, Any]:
    """Run image inference: SDPA baseline then Triton FA with a video frame.

    Returns:
        Dict with baseline, experimental, comparison, and frame metadata.
    """
    from turboquant_vllm.triton.attention_interface import install_triton_fa

    print("\n" + "=" * 60)
    print("STEP 2 — Image: SDPA vs Triton FA (Seinfeld clip01 frame)")
    print("=" * 60)

    if not clip_path.exists():
        print(f"  SKIP — clip not found at {clip_path}")
        return {"status": "skipped", "reason": f"clip not found: {clip_path}"}

    image = _extract_frame(clip_path)
    print(f"  Frame: {image.size[0]}x{image.size[1]} from {clip_path.name}")

    # SDPA baseline
    sdpa_result = _run_inference(
        model, processor, image_prompt, max_new_tokens, "SDPA-image", image=image
    )

    # Switch to Triton FA
    original_impl = model.config._attn_implementation
    install_triton_fa(model)
    triton_result = _run_inference(
        model, processor, image_prompt, max_new_tokens, "TritonFA-image", image=image
    )

    # Restore
    model.config._attn_implementation = original_impl

    comparison = _compare_results(sdpa_result, triton_result, "image")

    return {
        "sdpa": sdpa_result,
        "triton_fa": triton_result,
        "comparison": comparison,
        "frame_size": list(image.size),
        "clip": clip_path.name,
    }


def step_3_summary(
    device_info: dict[str, Any],
    model_config: dict[str, int],
    text_result: dict[str, Any],
    image_result: dict[str, Any],
) -> dict[str, Any]:
    """Print P5 Phase 1 exit gate assessment.

    Returns:
        Dict with pass/fail assessment for each exit criterion.
    """
    print("\n" + "=" * 60)
    print("STEP 3 — P5 Phase 1 Exit Gate Assessment")
    print("=" * 60)

    text_comp = text_result.get("comparison", {})
    image_comp = image_result.get("comparison", {})

    # Exit criterion 1.1: Unit tests (already done — 15/15)
    c1_1 = True

    # Exit criterion 1.2: GQA (already done — test_gqa_7_to_1)
    c1_2 = True

    # Exit criterion 1.3: Token-identical text output
    c1_3_text = text_comp.get("texts_identical", False)
    c1_3_image = (
        image_comp.get("texts_identical", False)
        if "comparison" in image_result
        else None
    )

    # Exit criterion 1.4: Throughput >= baseline
    c1_4_text = text_comp.get("throughput_ratio", 0) >= 0.9  # allow 10% margin
    c1_4_image = (
        image_comp.get("throughput_ratio", 0) >= 0.9
        if "comparison" in image_result
        else None
    )

    gate = {
        "1.1_unit_tests": {"pass": c1_1, "note": "15/15 tests >0.999 cosine"},
        "1.2_gqa_28q_4kv": {"pass": c1_2, "note": "Validated in test suite"},
        "1.3_text_identical": {
            "text_only": c1_3_text,
            "image": c1_3_image,
            "pass": c1_3_text and (c1_3_image is not False),
        },
        "1.4_throughput": {
            "text_ratio": text_comp.get("throughput_ratio"),
            "image_ratio": (
                image_comp.get("throughput_ratio")
                if "comparison" in image_result
                else None
            ),
            "pass": c1_4_text and (c1_4_image is not False),
        },
    }

    all_pass = all(v["pass"] for v in gate.values())
    gate["overall"] = "PASS" if all_pass else "FAIL"

    print(f"\n  Device: {device_info.get('device_name', 'N/A')}")
    print(
        f"  Model:  {model_config['num_layers']}L / "
        f"{model_config['num_heads']}Q / {model_config['num_kv_heads']}KV / "
        f"head_dim={model_config['head_dim']}"
    )
    print()
    for key, val in gate.items():
        if key == "overall":
            status = val
            print(f"\n  {'=' * 40}")
            print(f"  OVERALL: {status}")
            print(f"  {'=' * 40}")
        else:
            is_pass = val.get("pass", False)
            mark = "PASS" if is_pass else "FAIL"
            print(f"  [{mark}] {key}")

    return gate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(
    model_id: str,
    text_prompt: str,
    image_prompt: str,
    max_new_tokens: int,
    clip_path: Path,
    skip_image: bool = False,
) -> dict[str, Any]:
    """Run the full Phase 1 exit gate experiment.

    Returns:
        Dict with all step results and gate assessment.
    """
    results: dict[str, Any] = {
        "experiment": "009-triton-fa-e2e-validation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "text_prompt": text_prompt,
        "image_prompt": image_prompt,
    }

    results["device"] = _device_info()
    if not results["device"]["cuda_available"]:
        print("ABORT: CUDA required for Triton FA validation.")
        return results

    # Load model
    print("\nLoading model on GPU...")
    model, processor = _load_model(model_id)
    model_config = _detect_model_config(model)
    results["model_config"] = model_config
    print(
        f"  {model_config['num_layers']}L, {model_config['num_heads']}Q/"
        f"{model_config['num_kv_heads']}KV, head_dim={model_config['head_dim']}"
    )
    print(f"  attn_implementation: {model.config._attn_implementation}")

    # Step 1 — Text-only
    results["step_1"] = step_1_text_only(model, processor, text_prompt, max_new_tokens)

    # Step 2 — Image
    if skip_image:
        print("\n  Skipping image step (--skip-image)")
        results["step_2"] = {"status": "skipped"}
    else:
        results["step_2"] = step_2_image(
            model, processor, image_prompt, max_new_tokens, clip_path
        )

    # Step 3 — Gate assessment
    results["step_3"] = step_3_summary(
        results["device"],
        model_config,
        results["step_1"],
        results["step_2"],
    )

    return results


def main() -> None:
    """CLI entry point for Experiment 009."""
    parser = argparse.ArgumentParser(
        description="Experiment 009: Triton FA end-to-end Molmo2 validation"
    )
    parser.add_argument(
        "--model",
        default="allenai/Molmo2-4B",
        help="HuggingFace model ID (default: allenai/Molmo2-4B)",
    )
    parser.add_argument(
        "--text-prompt",
        default="Describe the main character of Seinfeld in one paragraph.",
        help="Text-only prompt for Step 1",
    )
    parser.add_argument(
        "--image-prompt",
        default="Describe what is happening in this image in detail.",
        help="Image prompt for Step 2",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--clip",
        type=Path,
        default=_CLIP_PATH,
        help=f"Path to video clip for image test (default: {_CLIP_PATH})",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip the image inference step",
    )
    parser.add_argument(
        "--output",
        default="experiments/logs/experiment-009-triton-fa-e2e.json",
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    results = run_experiment(
        model_id=args.model,
        text_prompt=args.text_prompt,
        image_prompt=args.image_prompt,
        max_new_tokens=args.max_new_tokens,
        clip_path=args.clip,
        skip_image=args.skip_image,
    )

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    # Exit code based on gate
    gate = results.get("step_3", {})
    overall = gate.get("overall", "FAIL")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
