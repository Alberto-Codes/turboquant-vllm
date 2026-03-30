r"""Experiment 022 -- Clip duration comparison: baseline vLLM vs TQ4.

Sends identical video segments at varying clip durations (5s, 15s, 30s, 60s)
to a vLLM OpenAI-compatible API.  Run once against the baseline quadlet,
then again against the TQ4 container to compare wall-clock time,
token usage, and output clarity at each duration.

Usage:
    # Against baseline quadlet (FP8 KV):
    uv run python experiments/experiment_022_clip_duration_comparison.py \
        --tag baseline

    # Against TQ4 container (--attention-backend CUSTOM):
    uv run python experiments/experiment_022_clip_duration_comparison.py \
        --tag tq4

    # Custom durations:
    uv run python experiments/experiment_022_clip_duration_comparison.py \
        --durations 5 10 20 40 --tag baseline
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

_EPISODE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "molmo-video-analyzer"
    / "data"
    / "tv"
    / "Seinfeld - S05E12 - The Stall - [WEBDL-720P][AAC 2.0][H264]-NTB.mkv"
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
    "puddy",
    "jane",
]

# Start at 2 minutes in to skip the cold open / credits
_START_OFFSET_S = 120


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
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


def _extract_clip(
    video_path: Path,
    start_s: float,
    duration_s: int,
    output_path: Path,
) -> Path:
    """Extract a single clip from the video at a specific offset."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
            "-i",
            str(video_path),
            "-t",
            str(duration_s),
            "-c",
            "copy",
            "-an",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _count_characters(text: str) -> dict[str, int]:
    """Count Seinfeld character name mentions in text."""
    lower = text.lower()
    return {name: lower.count(name) for name in _CHARACTERS if lower.count(name) > 0}


def _send_clip(
    clip_path: Path,
    url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Send a clip to the vLLM API and return results."""
    b64 = base64.b64encode(clip_path.read_bytes()).decode("ascii")
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

    start = time.perf_counter()
    resp = requests.post(f"{url}/chat/completions", json=payload, timeout=600)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "elapsed_s": round(elapsed, 2),
        "output_text": text,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "characters": _count_characters(text),
    }


def main() -> None:
    """CLI entry point for Experiment 022."""
    parser = argparse.ArgumentParser(
        description="Experiment 022: Clip duration comparison"
    )
    parser.add_argument("--episode", type=Path, default=_EPISODE_PATH)
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[5, 15, 30, 60],
        help="Clip durations to test in seconds (default: 5 15 30 60)",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="allenai/Molmo2-8B")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--tag",
        required=True,
        help="Run tag: 'baseline' or 'tq4' (used in output filename)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=_PROMPT,
        help="Prompt to send with each clip (default: detailed scene description)",
    )
    parser.add_argument(
        "--start-offset",
        type=float,
        default=_START_OFFSET_S,
        help="Seconds into episode to start extracting clips (default: 120)",
    )
    args = parser.parse_args()

    if not args.episode.exists():
        print(f"Episode not found: {args.episode}")
        sys.exit(1)

    episode_duration = _get_duration(args.episode)

    # Health check
    try:
        resp = requests.get(f"{args.vllm_url}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()
        print(f"vLLM ready — serving: {[m['id'] for m in models['data']]}")
    except requests.RequestException as exc:
        print(f"vLLM not reachable at {args.vllm_url}: {exc}")
        sys.exit(1)

    print(f"\nEpisode: {args.episode.name} ({episode_duration:.0f}s)")
    print(f"Start offset: {args.start_offset}s")
    print(f"Durations to test: {args.durations}s")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Tag: {args.tag}")

    results: dict[str, Any] = {
        "experiment": "022-clip-duration-comparison",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "model_id": args.model,
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "episode": args.episode.name,
        "episode_duration_s": round(episode_duration, 1),
        "start_offset_s": args.start_offset,
        "clips": [],
    }

    with tempfile.TemporaryDirectory(prefix="exp022_") as tmpdir:
        tmp = Path(tmpdir)

        for dur in args.durations:
            if args.start_offset + dur > episode_duration:
                print(f"\n  Skipping {dur}s — exceeds episode length")
                continue

            clip_path = tmp / f"clip_{dur}s.mp4"
            print(f"\n{'=' * 60}")
            print(f"CLIP: {dur}s (from {args.start_offset}s offset)")
            print(f"{'=' * 60}")

            _extract_clip(args.episode, args.start_offset, dur, clip_path)
            clip_size_mb = clip_path.stat().st_size / (1024 * 1024)
            print(f"  Clip size: {clip_size_mb:.1f} MB")

            try:
                result = _send_clip(
                    clip_path,
                    args.vllm_url,
                    args.model,
                    args.prompt,
                    args.max_new_tokens,
                )
                result["duration_s"] = dur
                result["clip_size_mb"] = round(clip_size_mb, 1)
                results["clips"].append(result)

                print(f"  Elapsed: {result['elapsed_s']}s")
                print(
                    f"  Tokens:  {result['input_tokens']} in, {result['output_tokens']} out"
                )
                print(f"  Characters detected: {result['characters']}")
                print(f"  Output preview: {result['output_text'][:200]}...")
            except requests.RequestException as exc:
                print(f"  FAILED: {exc}")
                results["clips"].append(
                    {
                        "duration_s": dur,
                        "clip_size_mb": round(clip_size_mb, 1),
                        "error": str(exc),
                    }
                )

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Dur':>5} {'Elapsed':>8} {'In Tok':>8} {'Out Tok':>8} {'Characters'}")
    print(f"{'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 20}")
    for clip in results["clips"]:
        if "error" in clip:
            print(f"{clip['duration_s']:>4}s {'FAIL':>8}")
        else:
            chars = ", ".join(f"{k}:{v}" for k, v in clip["characters"].items())
            print(
                f"{clip['duration_s']:>4}s "
                f"{clip['elapsed_s']:>7.1f}s "
                f"{clip['input_tokens']:>8} "
                f"{clip['output_tokens']:>8} "
                f"{chars}"
            )

    output_path = Path(f"experiments/logs/experiment-022-{args.tag}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
