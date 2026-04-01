r"""Experiment 028 -- Decode OOM resolution validation.

Validates that hotfix 1 (PR #43) and hotfix 2 (PR #45) resolved the
decode buffer OOM on RTX 4090 (24 GiB).  Captures system-level VRAM
measurements via nvidia-smi and records per-phase memory usage.

Expects a running vLLM server with TQ4 backend (--attention-backend CUSTOM).

Usage:
    # Start TQ4 vLLM server:
    uv run vllm serve allenai/Molmo2-4B \
        --attention-backend CUSTOM --enforce-eager \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        --trust-remote-code

    # Run experiment:
    uv run python experiments/experiment_028_decode_oom_validation.py \
        --tag molmo2-4b-tq4-085 --model allenai/Molmo2-4B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


def _get_vram_mib() -> dict[str, int]:
    """Get GPU VRAM usage via nvidia-smi (system-level, same as mem_get_info)."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = result.stdout.strip().split(", ")
    return {
        "used_mib": int(parts[0]),
        "free_mib": int(parts[1]),
        "total_mib": int(parts[2]),
    }


def _send_chat(
    url: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    """Send a chat completion request and return results."""
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    resp = requests.post(f"{url}/chat/completions", json=payload, timeout=180)
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
        "finish_reason": data["choices"][0].get("finish_reason", "unknown"),
    }


def _run_multi_turn(
    url: str,
    model_id: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Run a 5-turn conversation to exercise prefill + multiple decode steps."""
    turns = [
        "What are the three laws of thermodynamics? Explain each briefly.",
        "Now compare the second law to the concept of entropy in information theory.",
        "Give me a real-world engineering example where both apply.",
        "What common mistakes do students make when learning these concepts?",
        "Summarize our entire conversation in 3 bullet points.",
    ]

    messages: list[dict[str, str]] = []
    results = []

    for i, content in enumerate(turns):
        messages.append({"role": "user", "content": content})

        vram_before = _get_vram_mib()
        result = _send_chat(url, model_id, messages, max_tokens)
        vram_after = _get_vram_mib()

        messages.append({"role": "assistant", "content": result["output_text"]})

        result["turn"] = i + 1
        result["vram_before_mib"] = vram_before["used_mib"]
        result["vram_after_mib"] = vram_after["used_mib"]
        results.append(result)

    return results


def main() -> None:
    """CLI entry point for Experiment 028."""
    parser = argparse.ArgumentParser(
        description="Experiment 028: Decode OOM validation",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Model ID being served")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--tag", required=True, help="Run tag (e.g., 'molmo2-4b-tq4')")
    args = parser.parse_args()

    # Health check
    try:
        resp = requests.get(f"{args.vllm_url}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()
        serving = [m["id"] for m in models["data"]]
        print(f"vLLM ready — serving: {serving}")
    except requests.RequestException as exc:
        print(f"vLLM not reachable at {args.vllm_url}: {exc}")
        sys.exit(1)

    print("\nExperiment 028: Decode OOM Validation")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Tag: {args.tag}")

    vram_idle = _get_vram_mib()
    print(f"VRAM at idle: {vram_idle['used_mib']} MiB / {vram_idle['total_mib']} MiB")

    results: dict[str, Any] = {
        "experiment": "028-decode-oom-validation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "model_id": args.model,
        "max_new_tokens": args.max_new_tokens,
        "vram_idle_mib": vram_idle["used_mib"],
        "vram_total_mib": vram_idle["total_mib"],
    }

    # Phase 1: Single prompt (prefill + decode)
    print(f"\n{'=' * 60}")
    print("PHASE 1: Single prompt (prefill + decode)")
    print(f"{'=' * 60}")

    vram_pre = _get_vram_mib()
    try:
        single = _send_chat(
            args.vllm_url,
            args.model,
            [{"role": "user", "content": "Hello, what is 2+2?"}],
            args.max_new_tokens,
        )
        vram_post = _get_vram_mib()
        single["vram_before_mib"] = vram_pre["used_mib"]
        single["vram_after_mib"] = vram_post["used_mib"]
        results["single_prompt"] = single
        print(f"  OK: {single['output_tokens']} tokens in {single['elapsed_s']}s")
        print(f"  VRAM: {vram_pre['used_mib']} -> {vram_post['used_mib']} MiB")
        print(f"  Response: {single['output_text'][:200]}")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        results["single_prompt"] = {"error": str(exc)}

    # Phase 2: Multi-turn conversation (prefill + 5 decode rounds)
    print(f"\n{'=' * 60}")
    print("PHASE 2: Multi-turn conversation (5 turns)")
    print(f"{'=' * 60}")

    try:
        mt_results = _run_multi_turn(args.vllm_url, args.model, args.max_new_tokens)
        results["multi_turn"] = mt_results
        for r in mt_results:
            print(
                f"  Turn {r['turn']}: {r['input_tokens']}in/{r['output_tokens']}out "
                f"({r['elapsed_s']}s) VRAM={r['vram_after_mib']}MiB"
            )
    except Exception as exc:
        print(f"  FAILED: {exc}")
        results["multi_turn"] = {"error": str(exc)}

    # Peak VRAM
    vram_peak = _get_vram_mib()
    results["vram_peak_mib"] = vram_peak["used_mib"]
    results["vram_under_22gib"] = vram_peak["used_mib"] < 22528  # 22 GiB in MiB

    # Verdict
    single_ok = "error" not in results.get("single_prompt", {"error": True})
    mt_ok = isinstance(results.get("multi_turn"), list)
    under_budget = results["vram_under_22gib"]

    verdict = "PASS" if (single_ok and mt_ok and under_budget) else "FAIL"
    results["summary"] = {
        "single_prompt": "PASS" if single_ok else "FAIL",
        "multi_turn": "PASS" if mt_ok else "FAIL",
        "vram_under_22gib": "PASS" if under_budget else "FAIL",
        "peak_vram_mib": vram_peak["used_mib"],
        "verdict": verdict,
    }

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict}")
    print(f"  Single prompt: {'PASS' if single_ok else 'FAIL'}")
    print(f"  Multi-turn: {'PASS' if mt_ok else 'FAIL'}")
    print(
        f"  VRAM < 22 GiB: {'PASS' if under_budget else 'FAIL'} ({vram_peak['used_mib']} MiB)"
    )
    print(f"{'=' * 60}")

    output_path = Path(f"experiments/logs/experiment-028-{args.tag}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
