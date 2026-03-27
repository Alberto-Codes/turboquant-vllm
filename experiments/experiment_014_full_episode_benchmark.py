r"""Experiment 014 -- Full episode benchmark: vLLM baseline vs TQ4 compression.

P9 Phase 2: real-world video throughput comparison.  Splits a Seinfeld episode
into uniform-duration clips and processes them through two paths:

    A) vLLM baseline (5s clips) -- production inference server with FP8 KV cache
    B) TQ4 CompressedDynamicCache + SDPA (19s clips) -- HF transformers

Hypothesis: TQ4 processes 3.8x fewer clips (19s vs 5s), and even with 1.78x
per-token overhead, total wall-clock time is comparable.  Output quality
should be *better* because TQ4 sees full scene context per clip.

Usage:
    ```bash
    # Quick 2-minute test (default)
    uv run python experiments/experiment_014_full_episode_benchmark.py

    # Full 5-minute segment
    uv run python experiments/experiment_014_full_episode_benchmark.py --minutes 5

    # TQ4 path only (no vLLM required)
    uv run python experiments/experiment_014_full_episode_benchmark.py --path tq4

    # Baseline only
    uv run python experiments/experiment_014_full_episode_benchmark.py --path baseline
    ```

Examples:
    ```bash
    uv run python experiments/experiment_014_full_episode_benchmark.py --path tq4 --minutes 1
    ```

See Also:
    :mod:`turboquant_consumer.kv_cache`: CompressedDynamicCache implementation.
    docs/ROADMAP.md: P9 Phase 2 requirements and success criteria.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
import torch

_EPISODE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "molmo-video-analyzer"
    / "data"
    / "tv"
    / "Seinfeld - S07E06 - The Soup Nazi - [WEBDL-720P][AAC 2.0][H264]-NTB.mkv"
)

_PROMPT = (
    "Describe what is happening in this video clip in detail. "
    "Include the names of any characters you recognize, the setting, "
    "and any notable actions or dialogue."
)

_CHARACTERS = [
    "jerry",
    "george",
    "kramer",
    "elaine",
    "newman",
    "soup nazi",
    "yev kassem",
]


# -- Helpers ----------------------------------------------------------------


def _get_vram_mb() -> float:
    """Return peak GPU memory in MiB.

    Returns:
        Peak VRAM in MiB, or 0.0 if CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_vram() -> None:
    """Reset CUDA peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe.

    Args:
        video_path: Path to video file.

    Returns:
        Duration in seconds.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _split_video(
    video_path: Path,
    clip_duration: int,
    output_dir: Path,
    max_seconds: float,
) -> list[Path]:
    """Split video into uniform-duration clips using ffmpeg segment muxer.

    Uses stream copy (no re-encoding) for speed.  Audio is stripped since
    the vision model doesn't use it.

    Args:
        video_path: Source video file.
        clip_duration: Target duration per clip in seconds.
        output_dir: Directory for output clips.
        max_seconds: Maximum seconds of video to process.

    Returns:
        Sorted list of clip paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "clip_%04d.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-t",
            str(max_seconds),
            "-f",
            "segment",
            "-segment_time",
            str(clip_duration),
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            "-an",
            pattern,
        ],
        check=True,
        capture_output=True,
    )
    return sorted(output_dir.glob("clip_*.mp4"))


def _count_characters(text: str) -> dict[str, int]:
    """Count Seinfeld character name mentions in text.

    Args:
        text: Model output text to scan.

    Returns:
        Dict of character name to mention count (only non-zero).
    """
    lower = text.lower()
    return {name: lower.count(name) for name in _CHARACTERS if lower.count(name) > 0}


# -- Path A: vLLM Baseline -------------------------------------------------


def _run_vllm(
    clips: list[Path],
    url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Process clips through vLLM OpenAI-compatible API.

    Sends each clip as a base64-encoded video_url, matching the format
    used by molmo-video-analyzer's MolmoVllmAdapter.

    Args:
        clips: List of video clip paths.
        url: vLLM API base URL (e.g. ``http://127.0.0.1:8100/v1``).
        model_id: Model identifier for the API request.
        prompt: Prompt text for scene description.
        max_tokens: Maximum tokens to generate per clip.

    Returns:
        Dict with per-clip results and aggregate metrics.
    """
    endpoint = f"{url}/chat/completions"

    # Health check before starting
    try:
        resp = requests.get(f"{url}/models", timeout=10)
        resp.raise_for_status()
        print("  vLLM server is reachable")
    except requests.RequestException as exc:
        print(f"  vLLM server not reachable at {url}: {exc}")
        return {"error": str(exc), "clips": [], "total_elapsed_s": 0, "num_clips": 0}

    results: dict[str, Any] = {"clips": [], "total_elapsed_s": 0.0}

    for i, clip in enumerate(clips):
        b64 = base64.b64encode(clip.read_bytes()).decode("ascii")
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": max_tokens,
        }

        try:
            start = time.perf_counter()
            resp = requests.post(endpoint, json=payload, timeout=300)
            elapsed = time.perf_counter() - start
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [baseline] clip {i + 1}/{len(clips)} FAILED: {exc}")
            results["clips"].append({"clip": clip.name, "error": str(exc)})
            continue

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        clip_result = {
            "clip": clip.name,
            "elapsed_s": round(elapsed, 2),
            "output_text": text,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "characters": _count_characters(text),
        }
        results["clips"].append(clip_result)
        results["total_elapsed_s"] += elapsed

        print(
            f"  [baseline] clip {i + 1}/{len(clips)}: "
            f"{elapsed:.1f}s, {clip_result['input_tokens']} in, "
            f"{clip_result['output_tokens']} out"
        )

    results["total_elapsed_s"] = round(results["total_elapsed_s"], 2)
    results["num_clips"] = len(clips)
    results["total_output_tokens"] = sum(
        c.get("output_tokens", 0) for c in results["clips"] if "error" not in c
    )
    results["total_input_tokens"] = sum(
        c.get("input_tokens", 0) for c in results["clips"] if "error" not in c
    )
    return results


# -- Path B: TQ4 CompressedDynamicCache ------------------------------------


def _run_tq4(
    clips: list[Path],
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Process clips through HF transformers with TQ4 KV cache compression.

    Loads the model once, patches DynamicCache to use CompressedDynamicCache,
    then processes each clip sequentially.

    Args:
        clips: List of video clip paths.
        model_id: HuggingFace model identifier.
        prompt: Prompt text for scene description.
        max_tokens: Maximum tokens to generate per clip.

    Returns:
        Dict with per-clip results, compression stats, and aggregate metrics.
    """
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        DynamicCache,
    )

    from turboquant_consumer.kv_cache import CompressedDynamicCache

    print("  Loading model and processor...")
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    load_elapsed = time.perf_counter() - load_start
    print(f"  Model loaded in {load_elapsed:.1f}s")

    text_config = getattr(model.config, "text_config", model.config)
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )

    # Patch DynamicCache.__init__ to wrap each new cache with TQ4
    original_init = DynamicCache.__init__
    last_wrapper: list[CompressedDynamicCache | None] = [None]

    def patched_init(self_cache: Any, *a: Any, **kw: Any) -> None:
        """Wrap new DynamicCache with CompressedDynamicCache (TQ4).

        Other Parameters:
            **kw: Forwarded to original DynamicCache init.
        """
        original_init(self_cache, *a, **kw)
        w = CompressedDynamicCache(self_cache, head_dim=head_dim, bits=4)
        last_wrapper[0] = w

    DynamicCache.__init__ = patched_init  # type: ignore[method-assign]

    results: dict[str, Any] = {
        "clips": [],
        "total_elapsed_s": 0.0,
        "model_load_s": round(load_elapsed, 2),
    }

    try:
        for i, clip in enumerate(clips):
            # Free previous clip's cache to prevent VRAM accumulation
            if last_wrapper[0] is not None:
                last_wrapper[0].restore()
                last_wrapper[0] = None
            gc.collect()
            torch.cuda.empty_cache()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(clip)},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt"
            )
            inputs = {
                k: v.to(model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }
            input_len = inputs["input_ids"].shape[-1]

            _reset_vram()
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False
                )
            elapsed = time.perf_counter() - start
            vram_peak = _get_vram_mb()

            generated = output_ids[0, input_len:]
            text = processor.decode(generated, skip_special_tokens=True)
            comp_stats = last_wrapper[0].compression_stats() if last_wrapper[0] else {}

            clip_result = {
                "clip": clip.name,
                "elapsed_s": round(elapsed, 2),
                "output_text": text,
                "input_tokens": input_len,
                "output_tokens": len(generated),
                "tok_per_s": round(len(generated) / elapsed, 2) if elapsed > 0 else 0,
                "vram_peak_mib": round(vram_peak, 1),
                "compression": comp_stats,
                "characters": _count_characters(text),
            }
            results["clips"].append(clip_result)
            results["total_elapsed_s"] += elapsed

            print(
                f"  [tq4] clip {i + 1}/{len(clips)}: "
                f"{elapsed:.1f}s, {input_len} in, {len(generated)} out, "
                f"{vram_peak:.0f} MiB"
            )
    finally:
        DynamicCache.__init__ = original_init  # type: ignore[method-assign]
        del model
        torch.cuda.empty_cache()

    results["total_elapsed_s"] = round(results["total_elapsed_s"], 2)
    results["num_clips"] = len(clips)
    results["total_output_tokens"] = sum(
        c.get("output_tokens", 0) for c in results["clips"] if "error" not in c
    )
    results["total_input_tokens"] = sum(
        c.get("input_tokens", 0) for c in results["clips"] if "error" not in c
    )
    return results


# -- Comparison -------------------------------------------------------------


def _compare(baseline: dict[str, Any], tq4: dict[str, Any]) -> dict[str, Any]:
    """Compare baseline and TQ4 results.

    Args:
        baseline: Results dict from ``_run_vllm``.
        tq4: Results dict from ``_run_tq4``.

    Returns:
        Dict with time ratios, clip counts, and character recognition comparison.
    """
    b_time = baseline["total_elapsed_s"]
    t_time = tq4["total_elapsed_s"]

    b_chars: dict[str, int] = {}
    for clip in baseline["clips"]:
        for name, count in clip.get("characters", {}).items():
            b_chars[name] = b_chars.get(name, 0) + count

    t_chars: dict[str, int] = {}
    for clip in tq4["clips"]:
        for name, count in clip.get("characters", {}).items():
            t_chars[name] = t_chars.get(name, 0) + count

    return {
        "baseline_total_s": round(b_time, 2),
        "tq4_total_s": round(t_time, 2),
        "time_ratio": round(t_time / b_time, 2) if b_time > 0 else 0,
        "baseline_clips": baseline["num_clips"],
        "tq4_clips": tq4["num_clips"],
        "clip_ratio": round(baseline["num_clips"] / tq4["num_clips"], 2)
        if tq4["num_clips"] > 0
        else 0,
        "baseline_total_tokens": baseline.get("total_output_tokens", 0),
        "tq4_total_tokens": tq4.get("total_output_tokens", 0),
        "baseline_characters": b_chars,
        "tq4_characters": t_chars,
    }


# -- CLI --------------------------------------------------------------------


def main() -> None:
    """CLI entry point for Experiment 014."""
    parser = argparse.ArgumentParser(
        description="Experiment 014: Full episode benchmark -- vLLM baseline vs TQ4"
    )
    parser.add_argument("--episode", type=Path, default=_EPISODE_PATH)
    parser.add_argument(
        "--baseline-duration",
        type=int,
        default=5,
        help="Clip duration in seconds for vLLM baseline (default: 5)",
    )
    parser.add_argument(
        "--tq4-duration",
        type=int,
        default=19,
        help="Clip duration in seconds for TQ4 path (default: 19)",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="allenai/Molmo2-4B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--minutes",
        type=float,
        default=2.0,
        help="Minutes of episode to process (default: 2)",
    )
    parser.add_argument(
        "--path",
        choices=["both", "baseline", "tq4"],
        default="both",
        help="Which path(s) to run (default: both)",
    )
    parser.add_argument(
        "--output",
        default="experiments/logs/experiment-014-full-episode-benchmark.json",
    )
    args = parser.parse_args()

    if not args.episode.exists():
        print(f"Episode not found: {args.episode}")
        sys.exit(1)

    max_seconds = args.minutes * 60
    episode_duration = _get_duration(args.episode)

    if max_seconds > episode_duration:
        max_seconds = episode_duration
        print(f"  Note: episode is {episode_duration:.0f}s, capping at full length")

    print(f"\n  Episode: {args.episode.name}")
    print(f"  Segment: {max_seconds:.0f}s ({args.minutes} min)")
    print(f"  Model:   {args.model}")
    print(f"  Tokens:  {args.max_new_tokens} max per clip")

    results: dict[str, Any] = {
        "experiment": "014-full-episode-benchmark",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model,
        "max_new_tokens": args.max_new_tokens,
        "episode": args.episode.name,
        "episode_duration_s": round(episode_duration, 1),
        "segment_duration_s": round(max_seconds, 1),
        "baseline_clip_duration_s": args.baseline_duration,
        "tq4_clip_duration_s": args.tq4_duration,
    }

    with tempfile.TemporaryDirectory(prefix="exp014_") as tmpdir:
        tmp = Path(tmpdir)

        if args.path in ("both", "baseline"):
            print("\n" + "=" * 60)
            print(f"SPLITTING: {args.baseline_duration}s clips (baseline)")
            print("=" * 60)
            baseline_clips = _split_video(
                args.episode, args.baseline_duration, tmp / "baseline", max_seconds
            )
            print(f"  {len(baseline_clips)} clips created")

            print("\n" + "=" * 60)
            print("PATH A: vLLM BASELINE")
            print("=" * 60)
            results["baseline"] = _run_vllm(
                baseline_clips,
                args.vllm_url,
                args.model,
                _PROMPT,
                args.max_new_tokens,
            )

        if args.path in ("both", "tq4"):
            print("\n" + "=" * 60)
            print(f"SPLITTING: {args.tq4_duration}s clips (TQ4)")
            print("=" * 60)
            tq4_clips = _split_video(
                args.episode, args.tq4_duration, tmp / "tq4", max_seconds
            )
            print(f"  {len(tq4_clips)} clips created")

            print("\n" + "=" * 60)
            print("PATH B: TQ4 (CompressedDynamicCache + SDPA)")
            print("=" * 60)
            results["tq4"] = _run_tq4(
                tq4_clips, args.model, _PROMPT, args.max_new_tokens
            )

        if args.path == "both" and "baseline" in results and "tq4" in results:
            print("\n" + "=" * 60)
            print("COMPARISON")
            print("=" * 60)
            comparison = _compare(results["baseline"], results["tq4"])
            results["comparison"] = comparison
            print(
                f"  Baseline: {comparison['baseline_total_s']}s "
                f"({comparison['baseline_clips']} clips)"
            )
            print(
                f"  TQ4:      {comparison['tq4_total_s']}s "
                f"({comparison['tq4_clips']} clips)"
            )
            print(f"  Time ratio: {comparison['time_ratio']}x (TQ4 / baseline)")
            print(f"  Clip ratio: {comparison['clip_ratio']}x fewer calls with TQ4")
            print(f"  Baseline chars: {comparison['baseline_characters']}")
            print(f"  TQ4 chars:      {comparison['tq4_characters']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
