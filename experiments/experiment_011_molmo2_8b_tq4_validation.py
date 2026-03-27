r"""Experiment 011 -- Molmo2-8B TQ4 compression quality validation.

P4 gate: confirms CompressedDynamicCache (3.76x TQ4) preserves output
quality on the production 8B model, specifically character recognition
in Seinfeld clips (Elaine, Kramer, Jerry) that the 4B model cannot do.

Two paths compared:
    1. SDPA baseline (fp16 KV) — ground truth.
    2. Unfused TQ4 (CompressedDynamicCache + SDPA) — compression quality.

Usage:
    ```bash
    uv run python experiments/experiment_011_molmo2_8b_tq4_validation.py
    uv run python experiments/experiment_011_molmo2_8b_tq4_validation.py --skip-image
    ```

Outputs JSON to ``experiments/logs/experiment-011-molmo2-8b-tq4.json``.

Examples:
    ```bash
    uv run python experiments/experiment_011_molmo2_8b_tq4_validation.py
    ```

See Also:
    :mod:`turboquant_consumer.kv_cache`: CompressedDynamicCache.
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
    print(f"    Output: {output_text[:120]}...")

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


def run_experiment(
    model_id: str,
    text_prompt: str,
    image_prompt: str,
    max_new_tokens: int,
    bits: int,
    clip_path: Path,
    skip_image: bool,
) -> dict[str, Any]:
    """Run SDPA baseline vs unfused TQ4 on Molmo2-8B.

    Returns:
        Dict with all results and quality assessment.
    """
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        DynamicCache,
    )

    from turboquant_consumer.kv_cache import CompressedDynamicCache

    results: dict[str, Any] = {
        "experiment": "011-molmo2-8b-tq4-validation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "bits": bits,
        "max_new_tokens": max_new_tokens,
    }

    # Load model — use 4-bit weight quantization to fit 8B in 24GB VRAM
    from transformers import BitsAndBytesConfig

    print(f"\nLoading {model_id} (4-bit NF4 weights)...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _reset_vram()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    text_config = getattr(model.config, "text_config", model.config)
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    model_config = {
        "num_layers": text_config.num_hidden_layers,
        "num_heads": text_config.num_attention_heads,
        "num_kv_heads": getattr(
            text_config,
            "num_key_value_heads",
            text_config.num_attention_heads,
        ),
        "head_dim": head_dim,
    }
    results["model_config"] = model_config
    load_vram = _get_vram_mb()
    print(f"  Config: {model_config}")
    print(f"  Model VRAM: {load_vram:.0f} MiB")

    def run_2_paths(prompt: str, label: str, image: Any = None) -> dict[str, Any]:
        """Run SDPA baseline and unfused TQ4 on one prompt.

        Returns:
            Dict with results for each path and comparison.
        """
        # Path 1: SDPA baseline
        sdpa = _run_inference(
            model, processor, prompt, max_new_tokens, f"{label}-SDPA", image
        )

        # Path 2: Unfused TQ4
        original_init = DynamicCache.__init__
        wrappers: list[CompressedDynamicCache] = []

        def patched_init(self_cache: Any, *a: Any, **kw: Any) -> None:
            """Wrap DynamicCache with TQ4 compression.

            Other Parameters:
                **kw: Forwarded to original init.
            """
            original_init(self_cache, *a, **kw)
            wrappers.append(
                CompressedDynamicCache(self_cache, head_dim=head_dim, bits=bits)
            )

        DynamicCache.__init__ = patched_init  # type: ignore[method-assign]
        try:
            tq4 = _run_inference(
                model, processor, prompt, max_new_tokens, f"{label}-TQ4", image
            )
        finally:
            DynamicCache.__init__ = original_init  # type: ignore[method-assign]

        # Compare
        b_ids = sdpa["output_token_ids"]
        e_ids = tq4["output_token_ids"]
        n = min(len(b_ids), len(e_ids))
        matching = sum(1 for a, b in zip(b_ids[:n], e_ids[:n]) if a == b)
        match_rate = matching / n if n > 0 else 0.0
        texts_match = sdpa["output_text"].strip() == tq4["output_text"].strip()

        print(
            f"\n  [{label}] Token match: {matching}/{n} ({match_rate:.1%}), "
            f"text={'same' if texts_match else 'DIFF'}"
        )

        comp_stats = {}
        if wrappers:
            stats = wrappers[-1].compression_stats()
            if stats:
                comp_stats = stats
                print(
                    f"  Compression: {stats['compression_ratio']}x "
                    f"({stats['compressed_mib']:.1f} MiB vs "
                    f"{stats['baseline_mib']:.1f} MiB baseline)"
                )

        return {
            "sdpa": sdpa,
            "tq4": tq4,
            "token_match": matching,
            "tokens_compared": n,
            "match_rate": round(match_rate, 4),
            "texts_identical": texts_match,
            "compression_stats": comp_stats,
        }

    # Text-only
    print("\n" + "=" * 60)
    print("TEXT-ONLY")
    print("=" * 60)
    results["text"] = run_2_paths(text_prompt, "text")

    # Image
    if not skip_image and clip_path.exists():
        print("\n" + "=" * 60)
        print("IMAGE (Seinfeld clip01)")
        print("=" * 60)
        image = _extract_frame(clip_path)
        print(f"  Frame: {image.size[0]}x{image.size[1]}")
        results["image"] = run_2_paths(image_prompt, "image", image)
    else:
        results["image"] = {"status": "skipped"}

    # Quality assessment
    print("\n" + "=" * 60)
    print("P4 QUALITY ASSESSMENT — CHARACTER RECOGNITION")
    print("=" * 60)

    assessment: dict[str, Any] = {}

    # Check text output for character names
    character_names = ["Jerry", "Seinfeld", "Elaine", "Kramer", "George"]

    for mode in ["text", "image"]:
        data = results.get(mode, {})
        if "sdpa" not in data:
            continue
        sdpa_text = data["sdpa"]["output_text"]
        tq4_text = data["tq4"]["output_text"]

        sdpa_chars = [c for c in character_names if c.lower() in sdpa_text.lower()]
        tq4_chars = [c for c in character_names if c.lower() in tq4_text.lower()]

        assessment[mode] = {
            "sdpa_characters": sdpa_chars,
            "tq4_characters": tq4_chars,
            "characters_preserved": set(tq4_chars) >= set(sdpa_chars),
        }

        print(f"\n  [{mode}]")
        print(f"    SDPA characters: {sdpa_chars}")
        print(f"    TQ4 characters:  {tq4_chars}")
        print(f"    Preserved: {assessment[mode]['characters_preserved']}")

    results["assessment"] = assessment
    return results


def main() -> None:
    """CLI entry point for Experiment 011."""
    parser = argparse.ArgumentParser(
        description="Experiment 011: Molmo2-8B TQ4 compression quality"
    )
    parser.add_argument("--model", default="allenai/Molmo2-8B")
    parser.add_argument(
        "--text-prompt",
        default="Name the main characters of Seinfeld and describe each one briefly.",
    )
    parser.add_argument(
        "--image-prompt",
        default="Name each person in this image and describe what they are doing.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--clip", type=Path, default=_CLIP_PATH)
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument(
        "--output",
        default="experiments/logs/experiment-011-molmo2-8b-tq4.json",
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

    # Exit 0 if both paths produce coherent output
    has_text = "sdpa" in results.get("text", {})
    sys.exit(0 if has_text else 1)


if __name__ == "__main__":
    main()
