r"""Experiment 013 -- FP32 eager attention baseline vs P5 fused TQ4.

P9 Phase 1: establishes the paper's actual speed baseline. The paper's
"8x speedup" is vs FP32 attention logit computation, NOT vs cuDNN Flash
Attention. This experiment measures our P5 fused TQ4 kernel against
FP32 eager (no SDPA, no Flash Attention) to see if we already match.

Four comparison paths:
    1. bf16 SDPA (cuDNN Flash Attention) — our "impossible to beat" reference.
    2. FP32 eager attention — the paper's actual baseline.
    3. bf16 eager attention — intermediate reference.
    4. P5 fused TQ4 K+V — our kernel.

Usage:
    ```bash
    uv run python experiments/experiment_013_fp32_eager_baseline.py
    ```

Examples:
    ```bash
    uv run python experiments/experiment_013_fp32_eager_baseline.py --skip-image
    ```

See Also:
    :mod:`turboquant_vllm.triton.flash_attention_tq4_kv`: P5 fused kernel.
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
        Peak VRAM in MiB.
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
    """Extract first frame from video.

    Returns:
        PIL Image.

    Raises:
        RuntimeError: If no frames found.
    """
    import av
    from PIL import Image

    container = av.open(str(video_path))
    for frame in container.decode(container.streams.video[0]):
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
        Dict with output text, token IDs, timing.
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
        "vram_peak_mib": round(vram_peak, 1),
        "elapsed_s": round(elapsed, 2),
        "tok_per_s": round(tok_s, 2),
    }


def run_experiment(
    model_id: str,
    prompt: str,
    image_prompt: str,
    max_new_tokens: int,
    clip_path: Path,
    skip_image: bool,
) -> dict[str, Any]:
    """Run the 4-path baseline comparison.

    Returns:
        Dict with all results and speedup analysis.
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
        "experiment": "013-fp32-eager-baseline",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
    }

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def run_all_paths(
        prompt_text: str, label: str, image: Any = None
    ) -> dict[str, Any]:
        """Run 4 paths on one prompt, reloading model for dtype changes.

        Returns:
            Dict with results for each path and speedup analysis.
        """
        path_results: dict[str, Any] = {}

        # Path 1: bf16 SDPA (cuDNN Flash Attention)
        print("\n  Loading model (bf16 SDPA)...")
        _reset_vram()
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        path_results["bf16_sdpa"] = _run_inference(
            model,
            processor,
            prompt_text,
            max_new_tokens,
            f"{label}-bf16-SDPA",
            image,
        )
        del model
        torch.cuda.empty_cache()

        # Path 2: bf16 eager (no SDPA, no Flash Attention)
        print("\n  Loading model (bf16 eager)...")
        _reset_vram()
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        path_results["bf16_eager"] = _run_inference(
            model,
            processor,
            prompt_text,
            max_new_tokens,
            f"{label}-bf16-eager",
            image,
        )
        del model
        torch.cuda.empty_cache()

        # Path 3: FP32 eager (the paper's actual baseline)
        print("\n  Loading model (fp32 eager)...")
        _reset_vram()
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        path_results["fp32_eager"] = _run_inference(
            model,
            processor,
            prompt_text,
            max_new_tokens,
            f"{label}-fp32-eager",
            image,
        )
        del model
        torch.cuda.empty_cache()

        # Path 4: Fused TQ4 K+V (our kernel)
        print("\n  Loading model (bf16 + fused TQ4 K+V)...")
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

        original_init = DynamicCache.__init__
        fused_wrappers: list[CompressedDynamicCache] = []
        model_ref = [model]  # capture for closure

        def fused_patched_init(self_cache: Any, *a: Any, **kw: Any) -> None:
            """Wrap and install fused kernel.

            Other Parameters:
                **kw: Forwarded to original init.
            """
            original_init(self_cache, *a, **kw)
            w = CompressedDynamicCache(self_cache, head_dim=head_dim, bits=4)
            fused_wrappers.append(w)
            install_fused_tq4_kv(model_ref[0], w)

        DynamicCache.__init__ = fused_patched_init  # type: ignore[method-assign]
        try:
            path_results["fused_tq4"] = _run_inference(
                model,
                processor,
                prompt_text,
                max_new_tokens,
                f"{label}-fusedTQ4",
                image,
            )
        finally:
            DynamicCache.__init__ = original_init  # type: ignore[method-assign]
            uninstall_fused_tq4_kv(model)
        del model
        torch.cuda.empty_cache()

        # Speedup analysis
        fp32_tok = path_results["fp32_eager"]["tok_per_s"]
        fused_tok = path_results["fused_tq4"]["tok_per_s"]
        sdpa_tok = path_results["bf16_sdpa"]["tok_per_s"]
        eager_tok = path_results["bf16_eager"]["tok_per_s"]

        analysis = {
            "fused_vs_fp32_eager": round(fused_tok / fp32_tok, 2)
            if fp32_tok > 0
            else 0,
            "fused_vs_bf16_eager": round(fused_tok / eager_tok, 2)
            if eager_tok > 0
            else 0,
            "fused_vs_bf16_sdpa": round(fused_tok / sdpa_tok, 2) if sdpa_tok > 0 else 0,
            "sdpa_vs_fp32_eager": round(sdpa_tok / fp32_tok, 2) if fp32_tok > 0 else 0,
        }
        path_results["analysis"] = analysis

        print(f"\n  [{label}] Speedup analysis:")
        print(f"    Fused TQ4 vs FP32 eager:  {analysis['fused_vs_fp32_eager']}x")
        print(f"    Fused TQ4 vs bf16 eager:  {analysis['fused_vs_bf16_eager']}x")
        print(f"    Fused TQ4 vs bf16 SDPA:   {analysis['fused_vs_bf16_sdpa']}x")
        print(f"    bf16 SDPA vs FP32 eager:  {analysis['sdpa_vs_fp32_eager']}x")

        return path_results

    # Text-only
    print("\n" + "=" * 60)
    print("TEXT-ONLY")
    print("=" * 60)
    results["text"] = run_all_paths(prompt, "text")

    # Image
    if not skip_image and clip_path.exists():
        print("\n" + "=" * 60)
        print("IMAGE (Seinfeld clip01)")
        print("=" * 60)
        image = _extract_frame(clip_path)
        print(f"  Frame: {image.size[0]}x{image.size[1]}")
        results["image"] = run_all_paths(image_prompt, "image", image)
    else:
        results["image"] = {"status": "skipped"}

    return results


def main() -> None:
    """CLI entry point for Experiment 013."""
    parser = argparse.ArgumentParser(
        description="Experiment 013: FP32 eager baseline vs P5 fused TQ4"
    )
    parser.add_argument("--model", default="allenai/Molmo2-4B")
    parser.add_argument(
        "--prompt",
        default="Describe the main character of Seinfeld in one paragraph.",
    )
    parser.add_argument(
        "--image-prompt",
        default="Describe what is happening in this image in detail.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--clip", type=Path, default=_CLIP_PATH)
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument(
        "--output",
        default="experiments/logs/experiment-013-fp32-eager-baseline.json",
    )
    args = parser.parse_args()

    results = run_experiment(
        model_id=args.model,
        prompt=args.prompt,
        image_prompt=args.image_prompt,
        max_new_tokens=args.max_new_tokens,
        clip_path=args.clip,
        skip_image=args.skip_image,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
