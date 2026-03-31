r"""Experiment 024 -- Zero-change model probe: Llama 3.1 8B + Mistral 7B through TQ4 vLLM.

Quick probe to check if text-only models work through the TQ4 vLLM backend
with zero code changes.  Both models use head_dim=128 and standard GQA
(same as Molmo2), so they should work out of the box.

Tests:
    1. Does the model load and serve with --attention-backend CUSTOM?
    2. Does a simple text prompt produce coherent output?
    3. What are the token counts and latency?

Workflow:
    # Llama 3.1 8B baseline (FP8 KV):
    podman run -d --name vllm-exp024 \
        --security-opt=label=disable --device nvidia.com/gpu=all --shm-size=8g \
        -v vllm-models:/root/.cache/huggingface -p 8100:8000 \
        docker.io/vllm/vllm-openai:v0.18.0 \
        --model meta-llama/Llama-3.1-8B-Instruct --dtype auto \
        --max-model-len 4096 --enforce-eager --gpu-memory-utilization 0.90 \
        --kv-cache-dtype fp8 --trust-remote-code
    uv run python experiments/experiment_024_zero_change_model_probe.py \
        --tag llama-baseline --model meta-llama/Llama-3.1-8B-Instruct

    # Llama 3.1 8B TQ4:
    podman run ... localhost/vllm-turboquant:1.2.2 \
        --model meta-llama/Llama-3.1-8B-Instruct --attention-backend CUSTOM ...
    uv run python experiments/experiment_024_zero_change_model_probe.py \
        --tag llama-tq4 --model meta-llama/Llama-3.1-8B-Instruct

    # Repeat for Mistral 7B with mistralai/Mistral-7B-Instruct-v0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

_PROMPTS = [
    {
        "id": "factual",
        "prompt": "What are the three laws of thermodynamics? Explain each briefly.",
    },
    {
        "id": "reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
    },
    {
        "id": "creative",
        "prompt": "Write a haiku about debugging code.",
    },
    {
        "id": "long_context",
        "prompt": (
            "Summarize the key differences between TCP and UDP protocols. "
            "Include at least 5 specific technical differences and explain "
            "when you would choose one over the other."
        ),
    },
]


# Multi-turn conversation that builds up KV cache across turns.
# Each turn references prior context — tests incremental compression.
_MULTI_TURN = [
    {
        "role": "user",
        "content": "I'm planning a trip to Japan in April. What should I know?",
    },
    {
        "role": "user",
        "content": (
            "Great tips. Now I want to visit Kyoto specifically. "
            "What are the top 5 temples I should see, and what makes each unique?"
        ),
    },
    {
        "role": "user",
        "content": (
            "For the third temple you mentioned, what's the best time of day "
            "to visit, and is there a nearby restaurant you'd recommend for lunch?"
        ),
    },
    {
        "role": "user",
        "content": (
            "Actually, let me reconsider. Compare the first and fourth temples "
            "you listed. Which one would be better for someone interested in "
            "Zen meditation? Explain your reasoning."
        ),
    },
    {
        "role": "user",
        "content": (
            "Summarize our entire conversation so far in 3 bullet points. "
            "Include the specific temple names you mentioned."
        ),
    },
]

# Long passage with embedded facts, followed by comprehension questions.
# ~2,500 tokens of input — matches video prefill token counts.
_LONG_PASSAGE = (
    "The following is a detailed technical report on the Kepler-442 planetary system. "
    "Read it carefully, then answer the questions that follow.\n\n"
    "Kepler-442b is a confirmed near-Earth-sized exoplanet orbiting within the habitable "
    "zone of the K-type main-sequence star Kepler-442, located approximately 1,206 "
    "light-years from Earth in the constellation Lyra. The planet was discovered by "
    "NASA's Kepler space telescope using the transit method and announced on January 6, "
    "2015, as part of a batch of eight new habitable zone planets.\n\n"
    "Physical characteristics: Kepler-442b has a radius of approximately 1.34 Earth "
    "radii and an estimated mass of 2.34 Earth masses, giving it a surface gravity of "
    "approximately 1.31 times that of Earth. Its orbital period is 112.3 days, and it "
    "receives about 73% of the solar flux that Earth receives from the Sun. The planet's "
    "equilibrium temperature is estimated at 233 K (-40 degrees Celsius), though a "
    "greenhouse effect could raise surface temperatures significantly.\n\n"
    "The host star, Kepler-442, has a mass of 0.61 solar masses, a radius of 0.60 solar "
    "radii, and an effective temperature of 4,402 K. It is significantly older than the "
    "Sun, with an estimated age of 2.9 billion years. The star has a metallicity of "
    "[Fe/H] = -0.37, indicating it is metal-poor compared to the Sun. The apparent "
    "magnitude of the star is 14.97, making it too faint to observe with the naked eye.\n\n"
    "Habitability assessment: Kepler-442b has one of the highest Earth Similarity "
    "Index (ESI) values among confirmed exoplanets, at 0.836. The planet orbits well "
    "within the conservative habitable zone of its star. However, several factors "
    "complicate habitability assessments. The planet's K-type host star is less luminous "
    "than the Sun but more stable, with fewer stellar flares than M-dwarf stars. This "
    "stellar stability is considered favorable for the development of complex life. The "
    "planet's slightly higher surface gravity could support a thicker atmosphere, "
    "potentially enhancing the greenhouse effect and raising surface temperatures above "
    "the equilibrium estimate.\n\n"
    "Atmospheric modeling suggests that with a CO2-rich atmosphere similar to early "
    "Earth, surface temperatures could reach 260-280 K, well within the range for "
    "liquid water. However, without direct atmospheric characterization, these remain "
    "theoretical estimates. The James Webb Space Telescope (JWST) has been proposed as "
    "a tool for atmospheric characterization, but Kepler-442b's distance and the "
    "faintness of its host star (magnitude 14.97) make transit spectroscopy extremely "
    "challenging with current technology.\n\n"
    "Comparison with other habitable zone planets: Among the Kepler discoveries, "
    "Kepler-442b stands out for several reasons. Unlike Kepler-438b, which orbits an "
    "active M-dwarf prone to superflares that could strip its atmosphere, Kepler-442b's "
    "K-type host provides a more stable radiation environment. Compared to Kepler-452b "
    "(often called 'Earth's cousin'), Kepler-442b receives less stellar flux but orbits "
    "a smaller, cooler star that will remain on the main sequence for much longer than "
    "the Sun, providing a stable environment for billions of additional years.\n\n"
    "The discovery team, led by Guillermo Torres of the Harvard-Smithsonian Center for "
    "Astrophysics, used a combination of transit photometry, radial velocity measurements, "
    "and statistical validation techniques (specifically the BLENDER software package) "
    "to confirm the planetary nature of the transit signal. The false positive probability "
    "was calculated to be less than 0.01%, giving high confidence in the detection.\n\n"
    "Future observations: The PLATO mission (scheduled for launch in 2026) may provide "
    "improved characterization of the Kepler-442 system, including better constraints on "
    "the planet's mass through improved radial velocity precision. Additionally, the "
    "Extremely Large Telescope (ELT), expected to achieve first light in 2028, could "
    "potentially detect biosignature gases in the atmospheres of nearby habitable zone "
    "planets, though Kepler-442b's distance makes it a challenging target even for "
    "next-generation facilities.\n\n"
    "Questions:\n"
    "1. What is Kepler-442b's orbital period in days?\n"
    "2. What percentage of Earth's solar flux does Kepler-442b receive?\n"
    "3. What is the host star's metallicity [Fe/H] value?\n"
    "4. Who led the discovery team, and what institution are they from?\n"
    "5. Why is Kepler-442b considered more habitable than Kepler-438b?"
)


def _send_prompt(
    url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Send a chat completion request and return results."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    resp = requests.post(f"{url}/chat/completions", json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    finish = data["choices"][0].get("finish_reason", "unknown")

    return {
        "elapsed_s": round(elapsed, 2),
        "output_text": text,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "finish_reason": finish,
    }


def _send_multi_turn(
    url: str,
    model_id: str,
    turns: list[dict[str, str]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Send a multi-turn conversation, accumulating context each turn."""
    messages: list[dict[str, str]] = []
    results = []

    for i, turn in enumerate(turns):
        messages.append(turn)
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        start = time.perf_counter()
        resp = requests.post(f"{url}/chat/completions", json=payload, timeout=120)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": text})

        results.append(
            {
                "turn": i + 1,
                "elapsed_s": round(elapsed, 2),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "output_text": text,
                "finish_reason": data["choices"][0].get("finish_reason", "unknown"),
            }
        )

    return results


def main() -> None:
    """CLI entry point for Experiment 024."""
    parser = argparse.ArgumentParser(
        description="Experiment 024: Zero-change model probe",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", required=True, help="Model ID being served")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tag", required=True, help="Run tag (e.g., 'llama-tq4')")
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

    print("\nExperiment 024: Zero-Change Model Probe")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Tag: {args.tag}")
    print(f"Prompts: {len(_PROMPTS)}")

    results: dict[str, Any] = {
        "experiment": "024-zero-change-model-probe",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "model_id": args.model,
        "max_new_tokens": args.max_new_tokens,
        "prompts": [],
    }

    for p in _PROMPTS:
        print(f"\n{'=' * 60}")
        print(f"PROMPT: {p['id']}")
        print(f"{'=' * 60}")
        print(f"  Q: {p['prompt'][:80]}...")

        try:
            result = _send_prompt(
                args.vllm_url,
                args.model,
                p["prompt"],
                args.max_new_tokens,
            )
            result["prompt_id"] = p["id"]
            result["prompt"] = p["prompt"]
            results["prompts"].append(result)

            print(f"  Elapsed: {result['elapsed_s']}s")
            print(
                f"  Tokens:  {result['input_tokens']} in, "
                f"{result['output_tokens']} out ({result['finish_reason']})",
            )
            print(f"  A: {result['output_text'][:200]}...")
        except requests.RequestException as exc:
            print(f"  FAILED: {exc}")
            results["prompts"].append(
                {"prompt_id": p["id"], "prompt": p["prompt"], "error": str(exc)},
            )

    # ── Phase 2: Long passage comprehension (~2,500 input tokens) ──
    print(f"\n{'=' * 60}")
    print("LONG PASSAGE COMPREHENSION (~2,500 input tokens)")
    print(f"{'=' * 60}")

    try:
        lp_result = _send_prompt(
            args.vllm_url,
            args.model,
            _LONG_PASSAGE,
            args.max_new_tokens,
        )
        lp_result["prompt_id"] = "long_passage"
        results["long_passage"] = lp_result

        print(f"  Elapsed: {lp_result['elapsed_s']}s")
        print(
            f"  Tokens:  {lp_result['input_tokens']} in, "
            f"{lp_result['output_tokens']} out ({lp_result['finish_reason']})",
        )
        print(f"  A: {lp_result['output_text'][:300]}...")

        # Check factual answers
        text_lower = lp_result["output_text"].lower()
        facts = {
            "orbital_period": "112" in lp_result["output_text"],
            "solar_flux_73pct": "73" in lp_result["output_text"],
            "metallicity": "-0.37" in lp_result["output_text"]
            or "0.37" in lp_result["output_text"],
            "discovery_lead": "torres" in text_lower or "guillermo" in text_lower,
            "kepler438_comparison": "flare" in text_lower
            or "m-dwarf" in text_lower
            or "m dwarf" in text_lower,
        }
        correct = sum(facts.values())
        print(f"  Facts correct: {correct}/5 — {facts}")
        lp_result["facts_correct"] = correct
        lp_result["facts_detail"] = facts
    except requests.RequestException as exc:
        print(f"  FAILED: {exc}")
        results["long_passage"] = {"error": str(exc)}

    # ── Phase 3: Multi-turn conversation (5 turns, growing KV cache) ──
    print(f"\n{'=' * 60}")
    print("MULTI-TURN CONVERSATION (5 turns, growing KV cache)")
    print(f"{'=' * 60}")

    try:
        mt_results = _send_multi_turn(
            args.vllm_url,
            args.model,
            _MULTI_TURN,
            args.max_new_tokens,
        )
        results["multi_turn"] = mt_results

        for r in mt_results:
            print(
                f"\n  Turn {r['turn']}: {r['input_tokens']} in, "
                f"{r['output_tokens']} out, {r['elapsed_s']}s"
            )
            print(f"  A: {r['output_text'][:150]}...")

        # Check if turn 5 (summary) references earlier content
        if mt_results:
            last = mt_results[-1]
            last_lower = last["output_text"].lower()
            coherence = {
                "mentions_kyoto": "kyoto" in last_lower,
                "mentions_temple": "temple" in last_lower,
                "has_bullets": "•" in last["output_text"]
                or "- " in last["output_text"]
                or "1." in last["output_text"],
            }
            print(f"\n  Turn 5 coherence: {coherence}")
            results["multi_turn_coherence"] = coherence
    except requests.RequestException as exc:
        print(f"  FAILED: {exc}")
        results["multi_turn"] = {"error": str(exc)}

    # ── Summary ──
    successful = [r for r in results["prompts"] if "error" not in r]
    failed = [r for r in results["prompts"] if "error" in r]

    lp_ok = "error" not in results.get("long_passage", {"error": True})
    mt_ok = isinstance(results.get("multi_turn"), list)

    total_tests = len(_PROMPTS) + (1 if lp_ok else 0) + (1 if mt_ok else 0)
    total_pass = len(successful) + (1 if lp_ok else 0) + (1 if mt_ok else 0)
    total_fail = len(failed) + (0 if lp_ok else 1) + (0 if mt_ok else 1)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Tag: {args.tag}")
    print(f"Short prompts: {len(successful)}/{len(_PROMPTS)}")
    lp_detail = ""
    if lp_ok:
        lp_detail = f" ({results['long_passage']['facts_correct']}/5 facts)"
    print(f"Long passage: {'PASS' if lp_ok else 'FAIL'}{lp_detail}")
    mt_detail = ""
    if mt_ok:
        mt_detail = f" ({len(results['multi_turn'])} turns)"
    print(f"Multi-turn: {'PASS' if mt_ok else 'FAIL'}{mt_detail}")
    print(f"Total: {total_pass}/{total_tests}")

    if successful:
        avg_elapsed = sum(r["elapsed_s"] for r in successful) / len(successful)
        print(f"Avg short prompt latency: {avg_elapsed:.1f}s")
    if mt_ok:
        last_turn = results["multi_turn"][-1]
        print(f"Turn 5 KV cache: {last_turn['input_tokens']} tokens")

    verdict = "PASS" if total_fail == 0 else "FAIL"
    results["summary"] = {
        "short_prompts_passed": len(successful),
        "long_passage": "PASS" if lp_ok else "FAIL",
        "multi_turn": "PASS" if mt_ok else "FAIL",
        "total_passed": total_pass,
        "total_tests": total_tests,
        "verdict": verdict,
    }

    output_path = Path(f"experiments/logs/experiment-024-{args.tag}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nVerdict: {verdict}")
    print(f"Results saved to {output_path}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
