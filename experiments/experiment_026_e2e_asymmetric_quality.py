r"""Experiment 026 -- End-to-end text quality: K4/V4 vs K4/V3.

Validates that K4/V3 asymmetric compression produces equivalent text output
to K4/V4 baseline.  Unlike Experiment 025 (per-layer cosine), this measures
what users actually care about: does the model say the same thing?

Protocol (mirrors Experiment 024):
    1. Short prompts  — 4 diverse prompts (factual, reasoning, creative, long-form)
    2. Long passage    — ~960 tokens of context + 5 factual questions
    3. Multi-turn      — 5-turn conversation with growing context

For each test, the model generates text through both K4/V4 and K4/V3
CompressedDynamicCache.  Outputs are compared for equivalence.

Models:
    - allenai/Molmo2-4B        (primary use case — video inference)
    - meta-llama/Llama-3.1-8B  (mainstream reference)
    - Qwen/Qwen2.5-3B          (highest K/V norm ratio)
    - google/gemma-3-4b-it      (tightest cosine in Exp 025)

Usage::

    eval "$(direnv export bash)"
    uv run python experiments/experiment_026_e2e_asymmetric_quality.py

    # Single model
    uv run python experiments/experiment_026_e2e_asymmetric_quality.py \\
        --model meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

# ── Test prompts ──────────────────────────────────────────────────────────────

SHORT_PROMPTS = [
    {
        "id": "factual",
        "prompt": "What is the capital of France and what is it known for?",
        "max_new_tokens": 100,
    },
    {
        "id": "reasoning",
        "prompt": (
            "A farmer has 15 sheep. All but 9 run away. "
            "How many sheep does the farmer have left?"
        ),
        "max_new_tokens": 50,
    },
    {
        "id": "creative",
        "prompt": "Write a haiku about debugging code.",
        "max_new_tokens": 50,
    },
    {
        "id": "long-form",
        "prompt": (
            "Explain the difference between a stack and a queue "
            "in computer science. Give an example of when you would use each."
        ),
        "max_new_tokens": 200,
    },
]

LONG_PASSAGE = (
    "The Great Wall of China is a series of fortifications that were built "
    "across the historical northern borders of ancient Chinese states and "
    "Imperial China as protection against various nomadic groups from the "
    "Eurasian Steppe. Several walls were built from as early as the 7th "
    "century BC, with selective stretches later joined together by Qin Shi "
    "Huang, the first emperor of China. Little of the Qin wall remains. "
    "Later on, many successive dynasties built and maintained multiple "
    "stretches of border walls. The best-known sections of the wall were "
    "built by the Ming dynasty. Apart from defense, other purposes of the "
    "Great Wall have included border controls, allowing the imposition of "
    "duties on goods transported along the Silk Road, regulation or "
    "encouragement of trade and the control of immigration and emigration. "
    "The frontier walls built by different dynasties have multiple courses. "
    "Collectively, they stretch from Liaodong in the east to Lop Lake in "
    "the west, from the present-day Sino-Russian border in the north to "
    "Tao River in the south, spanning across a total length of over "
    "20,000 kilometers."
)

LONG_PASSAGE_QUESTIONS = [
    "What was the primary purpose of the Great Wall?",
    "Who first joined the wall stretches together?",
    "Which dynasty built the best-known sections?",
    "Name one non-defense purpose of the wall.",
    "What is the total length of the wall?",
]

MULTI_TURN_PROMPTS = [
    "Tell me about Kyoto, Japan in two sentences.",
    "What temples are famous there?",
    "Which one should I visit first if I only have one day?",
    "What food should I try nearby?",
    "Summarize our conversation in one sentence.",
]

# ── Models ────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "allenai/Molmo2-4B",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-3B",
    "google/gemma-3-4b-it",
]

CONFIGS = [
    {"name": "K4V4", "k_bits": 4, "v_bits": 4},
    {"name": "K4V3", "k_bits": 4, "v_bits": 3},
]


# ── Generation helper ─────────────────────────────────────────────────────────


def _generate_with_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    k_bits: int,
    v_bits: int,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """Generate text using CompressedDynamicCache with given bit config.

    Uses temperature=0 (greedy) for deterministic output comparison.
    """
    from transformers import DynamicCache

    from turboquant_vllm.kv_cache import CompressedDynamicCache
    from turboquant_vllm.verify import _detect_model_config

    model_cfg = _detect_model_config(model)
    head_dim = model_cfg["head_dim"]
    config = model.config

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    cache = DynamicCache(config=config)
    # Shim for Phi-3 compatibility
    if not hasattr(cache, "get_usable_length"):
        cache.get_usable_length = lambda new_seq_len, layer_idx=0: cache.get_seq_length(
            layer_idx
        )

    text_config = getattr(config, "text_config", config)
    cdc = CompressedDynamicCache(
        cache,
        head_dim=head_dim,
        k_bits=k_bits,
        v_bits=v_bits,
        model_config=text_config,
    )

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
            )
    finally:
        cdc.restore()

    # Decode only the generated tokens (exclude prompt)
    generated_ids = outputs[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ── Test runners ──────────────────────────────────────────────────────────────


def _run_short_prompts(
    model: Any, tokenizer: Any, configs: list[dict]
) -> list[dict[str, Any]]:
    """Run 4 short prompts through each config, compare outputs."""
    results = []
    for sp in SHORT_PROMPTS:
        outputs = {}
        for cfg in configs:
            text = _generate_with_cache(
                model,
                tokenizer,
                sp["prompt"],
                k_bits=cfg["k_bits"],
                v_bits=cfg["v_bits"],
                max_new_tokens=sp["max_new_tokens"],
            )
            outputs[cfg["name"]] = text

        # Compare: exact match or semantic equivalence
        baseline = outputs[configs[0]["name"]]
        experimental = outputs[configs[1]["name"]]
        exact_match = baseline == experimental

        results.append(
            {
                "test": f"short-{sp['id']}",
                "prompt": sp["prompt"],
                "outputs": outputs,
                "exact_match": exact_match,
            }
        )
        status = "EXACT" if exact_match else "DIFF"
        print(f"    [{status}] {sp['id']}")

    return results


def _run_long_passage(
    model: Any, tokenizer: Any, configs: list[dict]
) -> list[dict[str, Any]]:
    """Run long passage + 5 questions through each config."""
    results = []
    for i, question in enumerate(LONG_PASSAGE_QUESTIONS):
        prompt = (
            f"Read this passage and answer the question.\n\n"
            f"Passage: {LONG_PASSAGE}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        outputs = {}
        for cfg in configs:
            text = _generate_with_cache(
                model,
                tokenizer,
                prompt,
                k_bits=cfg["k_bits"],
                v_bits=cfg["v_bits"],
                max_new_tokens=80,
            )
            outputs[cfg["name"]] = text

        baseline = outputs[configs[0]["name"]]
        experimental = outputs[configs[1]["name"]]
        exact_match = baseline == experimental

        results.append(
            {
                "test": f"passage-q{i + 1}",
                "question": question,
                "outputs": outputs,
                "exact_match": exact_match,
            }
        )
        status = "EXACT" if exact_match else "DIFF"
        print(f"    [{status}] Q{i + 1}: {question[:50]}...")

    return results


def _run_multi_turn(
    model: Any, tokenizer: Any, configs: list[dict]
) -> list[dict[str, Any]]:
    """Run 5-turn conversation through each config."""
    results = []

    for cfg in configs:
        conversation_history = ""
        turns = []
        for i, user_msg in enumerate(MULTI_TURN_PROMPTS):
            if conversation_history:
                prompt = f"{conversation_history}\nUser: {user_msg}\nAssistant:"
            else:
                prompt = f"User: {user_msg}\nAssistant:"

            response = _generate_with_cache(
                model,
                tokenizer,
                prompt,
                k_bits=cfg["k_bits"],
                v_bits=cfg["v_bits"],
                max_new_tokens=100,
            )
            turns.append({"turn": i + 1, "user": user_msg, "response": response})
            conversation_history = f"{prompt} {response}"

        results.append({"config": cfg["name"], "turns": turns})

    # Compare turn-by-turn
    turn_comparisons = []
    baseline_turns = results[0]["turns"]
    experimental_turns = results[1]["turns"]
    for i in range(len(MULTI_TURN_PROMPTS)):
        b = baseline_turns[i]["response"]
        e = experimental_turns[i]["response"]
        exact_match = b == e
        turn_comparisons.append(
            {
                "turn": i + 1,
                "baseline": b,
                "experimental": e,
                "exact_match": exact_match,
            }
        )
        status = "EXACT" if exact_match else "DIFF"
        print(f"    [{status}] Turn {i + 1}: {MULTI_TURN_PROMPTS[i][:40]}...")

    return [{"conversations": results, "turn_comparisons": turn_comparisons}]


# ── Per-model runner ──────────────────────────────────────────────────────────


def _run_model(model_id: str, output_dir: Path) -> dict[str, Any]:
    """Run the full end-to-end test suite for one model."""
    from transformers import AutoConfig, AutoTokenizer

    print(f"\n{'=' * 72}")
    print(f"Model: {model_id}")
    print(f"{'=' * 72}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    is_vlm = hasattr(config, "text_config")

    print("  Loading model...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if is_vlm:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    load_s = time.perf_counter() - t0
    print(f"  Loaded in {load_s:.1f}s")

    result: dict[str, Any] = {
        "experiment": "026-e2e-asymmetric-quality",
        "model": model_id,
        "model_type": config.model_type,
        "configs": [c["name"] for c in CONFIGS],
        "load_s": round(load_s, 2),
    }

    # Short prompts
    print("  Short prompts:")
    t1 = time.perf_counter()
    result["short_prompts"] = _run_short_prompts(model, tokenizer, CONFIGS)
    result["short_prompts_s"] = round(time.perf_counter() - t1, 2)

    # Long passage
    print("  Long passage:")
    t2 = time.perf_counter()
    result["long_passage"] = _run_long_passage(model, tokenizer, CONFIGS)
    result["long_passage_s"] = round(time.perf_counter() - t2, 2)

    # Multi-turn
    print("  Multi-turn:")
    t3 = time.perf_counter()
    result["multi_turn"] = _run_multi_turn(model, tokenizer, CONFIGS)
    result["multi_turn_s"] = round(time.perf_counter() - t3, 2)

    # Tally
    all_tests = result["short_prompts"] + result["long_passage"]
    exact_count = sum(1 for t in all_tests if t["exact_match"])
    total_tests = len(all_tests)

    turn_comparisons = result["multi_turn"][0]["turn_comparisons"]
    turn_exact = sum(1 for t in turn_comparisons if t["exact_match"])
    total_turns = len(turn_comparisons)

    result["summary"] = {
        "prompt_tests_exact": exact_count,
        "prompt_tests_total": total_tests,
        "turn_tests_exact": turn_exact,
        "turn_tests_total": total_turns,
        "all_exact": exact_count == total_tests and turn_exact == total_turns,
    }

    print(
        f"  Result: {exact_count}/{total_tests} prompts exact, "
        f"{turn_exact}/{total_turns} turns exact"
    )

    # Cleanup
    try:
        model.to("cpu")
    except RuntimeError:
        pass
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Save
    safe_name = model_id.replace("/", "--")
    out_path = output_dir / f"experiment-026-{safe_name}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {out_path}")

    return result


# ── Summary ───────────────────────────────────────────────────────────────────


def _print_summary(all_results: list[dict[str, Any]]) -> None:
    """Print summary table."""
    print(f"\n{'=' * 72}")
    print("SUMMARY — Experiment 026: K4/V4 vs K4/V3 End-to-End Text Quality")
    print(f"{'=' * 72}")
    print(f"{'Model':<25} {'Prompts':<15} {'Turns':<15} {'Verdict'}")
    print("-" * 72)

    all_pass = True
    for res in all_results:
        if "error" in res:
            print(f"{res['model']:<25} ERROR: {res['error']}")
            all_pass = False
            continue

        s = res["summary"]
        short_name = res["model"].split("/")[-1]
        prompts = f"{s['prompt_tests_exact']}/{s['prompt_tests_total']}"
        turns = f"{s['turn_tests_exact']}/{s['turn_tests_total']}"
        # Verdict: EQUIVALENT if all exact, or if outputs are semantically
        # equivalent (human judgment needed for diffs)
        total_exact = s["prompt_tests_exact"] + s["turn_tests_exact"]
        total_all = s["prompt_tests_total"] + s["turn_tests_total"]
        if total_exact == total_all:
            verdict = "IDENTICAL"
        elif total_exact >= total_all * 0.8:
            verdict = "EQUIVALENT"
        else:
            verdict = "DIVERGENT"
            all_pass = False

        print(f"{short_name:<25} {prompts:<15} {turns:<15} {verdict}")

    print("-" * 72)
    overall = "PASS" if all_pass else "REVIEW NEEDED"
    print(f"Overall: {overall}")
    print(
        "Note: DIFF outputs need human review — temperature=0 generates "
        "deterministically but compressed KV can shift logit boundaries."
    )
    print(f"{'=' * 72}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment 026: E2E text quality K4/V4 vs K4/V3"
    )
    parser.add_argument("--model", default=None, help="Single model ID")
    parser.add_argument(
        "--output-dir", default="experiments/logs", help="Output directory"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else list(DEFAULT_MODELS)

    print("Experiment 026 — E2E Asymmetric Quality: K4/V4 vs K4/V3")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Models: {len(models)}")
    print("Tests: 4 short + 5 passage + 5 multi-turn = 14 per model")
    print("Temperature: 0.0 (greedy, deterministic)")

    all_results: list[dict[str, Any]] = []
    for model_id in models:
        try:
            result = _run_model(model_id, output_dir)
            all_results.append(result)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            all_results.append({"model": model_id, "error": str(exc)})
            gc.collect()
            torch.cuda.empty_cache()

    # Combined output
    combined = output_dir / "experiment-026-combined.json"
    combined.write_text(json.dumps(all_results, indent=2, default=str))

    _print_summary(all_results)


if __name__ == "__main__":
    main()
