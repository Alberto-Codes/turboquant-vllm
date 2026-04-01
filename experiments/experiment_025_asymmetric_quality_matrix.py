r"""Experiment 025 -- AC3 asymmetric quality matrix with real model activations.

Validates AC3 from Story 10.2 (Asymmetric K/V Compression) using actual model
forward passes rather than random Gaussian data.  The core hypothesis is that
real activations have lower entropy than Gaussian noise, so cosine similarity
under compression should meet or exceed the random-data baseline.

Gate (AC3 release criterion):
    K4/V4  >= 0.99 cosine on all validated models  (existing gate)
    K4/V3  >= 0.99 cosine on all validated models  (AC3 gate)
    K4/V2  data collection only, no gate

For each model the script:
    1. Loads the model in bfloat16 on CUDA.
    2. Runs a real 128-token text prefill through the model, capturing the
       KV tensors written to a DynamicCache hook.
    3. Re-compresses those captured tensors with each bit config.
    4. Reports per-layer cosine similarity (K and V separately + min).
    5. Writes per-model JSON to experiments/logs/.
    6. Prints a summary table at the end.

Models are loaded and unloaded one at a time to fit within 24 GiB.

Usage::

    # All models (default)
    uv run python experiments/experiment_025_asymmetric_quality_matrix.py

    # Single model (debug)
    uv run python experiments/experiment_025_asymmetric_quality_matrix.py \\
        --model mistralai/Mistral-7B-v0.1

    # Custom output dir
    uv run python experiments/experiment_025_asymmetric_quality_matrix.py \\
        --output-dir /tmp/exp025

Notes:
    - eval "$(direnv export bash)" is required before running for HF_TOKEN.
    - phi-4 (14B) may require --skip-large-models if VRAM is tight.  The
      script will attempt it and log OOM gracefully rather than crashing.
    - SWA models (Gemma-2, Gemma-3) use DynamicCache(config=config) to
      enable correct layer-type metadata.  SWA layers are excluded from
      the quality matrix (same as CompressedDynamicCache behaviour).
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

# ── Prefill corpus ─────────────────────────────────────────────────────────────
# ~128 tokens of real English text.  A technical paragraph is chosen because
# it produces diverse KV activations across layers (vs repetitive text).
_PREFILL_TEXT = (
    "The transformer architecture relies on scaled dot-product attention to "
    "model relationships between tokens. Each attention layer maintains key "
    "and value projections that are stored in a KV cache during autoregressive "
    "decoding. Quantizing this cache reduces memory bandwidth and enables "
    "longer context windows on fixed hardware. The TurboQuant approach applies "
    "Lloyd-Max scalar quantization after rotating activations with a learned "
    "orthogonal matrix, achieving compression ratios above 3x while preserving "
    "cosine similarity above 0.99 on validated model families including Llama, "
    "Mistral, Qwen, Phi, Gemma, and Molmo."
)

# Models to evaluate.  Ordered roughly by parameter count (smallest first) to
# fail fast on VRAM issues before reaching the largest models.
_DEFAULT_MODELS: list[str] = [
    "Qwen/Qwen2.5-3B",
    "google/gemma-2-2b",
    "allenai/Molmo2-4B",
    "google/gemma-3-4b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-3.1-8B",
    "microsoft/phi-4",
]

# Bit configs: (k_bits, v_bits, ac3_gated)
# K4/V4: existing baseline gate (>= 0.99)
# K4/V3: AC3 gate (>= 0.99 on real activations)
# K4/V2: data collection only
_BITS_CONFIGS: list[tuple[int, int, bool]] = [
    (4, 4, True),
    (4, 3, True),
    (4, 2, False),
]

AC3_THRESHOLD = 0.99


# ── KV capture hook ───────────────────────────────────────────────────────────


class _KVCaptureCache:
    """Wraps a DynamicCache to record the raw KV tensors written per layer.

    The hook replaces cache.update(), records the tensors passed in, then
    delegates to the original update so the model forward pass completes
    normally.  Restore() removes the hook.
    """

    def __init__(self, cache: Any) -> None:
        self.cache = cache
        self.captured_keys: list[torch.Tensor] = []
        self.captured_values: list[torch.Tensor] = []
        self._original_update = cache.update
        cache.update = self._capture_update

    def _capture_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Extend lists on first encounter of each layer index
        while len(self.captured_keys) <= layer_idx:
            self.captured_keys.append(None)  # type: ignore[arg-type]
            self.captured_values.append(None)  # type: ignore[arg-type]
        # First call per layer: just store.  Subsequent calls (decode steps)
        # concatenate along the sequence dimension.
        if self.captured_keys[layer_idx] is None:
            self.captured_keys[layer_idx] = key_states.detach().clone()
            self.captured_values[layer_idx] = value_states.detach().clone()
        else:
            self.captured_keys[layer_idx] = torch.cat(
                [self.captured_keys[layer_idx], key_states.detach().clone()], dim=-2
            )
            self.captured_values[layer_idx] = torch.cat(
                [self.captured_values[layer_idx], value_states.detach().clone()], dim=-2
            )
        return self._original_update(key_states, value_states, layer_idx, cache_kwargs)

    def restore(self) -> None:
        self.cache.update = self._original_update


# ── Model helpers ─────────────────────────────────────────────────────────────


def _detect_model_config(model: Any) -> dict[str, int]:
    """Extract head_dim, num_kv_heads, num_layers from model config.

    Handles VLM text_config wrappers and the head_dim fallback.
    """
    config = model.config
    text_config = getattr(config, "text_config", config)
    hidden_size = text_config.hidden_size
    num_heads = text_config.num_attention_heads
    raw_head_dim = getattr(text_config, "head_dim", None)

    if raw_head_dim is not None and raw_head_dim > 0:
        head_dim = raw_head_dim
    elif num_heads > 0:
        head_dim = hidden_size // num_heads
    else:
        msg = f"Cannot determine head_dim for {model.config.model_type}"
        raise ValueError(msg)

    num_kv_heads = getattr(text_config, "num_key_value_heads", num_heads)
    num_layers = text_config.num_hidden_layers
    return {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
    }


def _load_model(model_id: str) -> tuple[Any, Any]:
    """Load tokenizer and model.  Returns (tokenizer, model)."""
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    is_vlm = hasattr(config, "text_config")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if is_vlm:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


def _run_prefill(
    model: Any,
    tokenizer: Any,
    text: str,
    config: Any,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run a real text prefill and return captured (keys, values) per layer.

    Returns a list of (key_tensor, value_tensor) indexed by layer.  Each
    tensor has shape (batch=1, num_kv_heads, seq_len, head_dim).

    Uses DynamicCache(config=config) for SWA-aware layer instantiation.
    Only captures; no decode step is performed.
    """
    from transformers import DynamicCache

    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    cache = DynamicCache(config=config)
    hook = _KVCaptureCache(cache)

    try:
        with torch.no_grad():
            # Some models (e.g. Phi-3) call get_usable_length() which was
            # removed in newer transformers DynamicCache.  Patch if missing.
            if not hasattr(cache, "get_usable_length"):
                cache.get_usable_length = lambda new_seq_len, layer_idx=0: (
                    cache.get_seq_length(layer_idx)
                )
            model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )
    finally:
        hook.restore()

    # Collect per-layer tensors; None entries are SWA-bypassed layers in
    # models that don't write to the global cache for those indices.
    captured = []
    for k, v in zip(hook.captured_keys, hook.captured_values):
        captured.append((k, v))

    return captured


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scalar cosine similarity between two flattened tensors."""
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def _compress_and_measure(
    captured: list[tuple[torch.Tensor, torch.Tensor]],
    head_dim: int,
    k_bits: int,
    v_bits: int,
    config: Any,
) -> dict[str, Any]:
    """Compress captured KV tensors and compute per-layer cosine similarity.

    Skips layers where the captured tensor is None (SWA-bypassed layers).
    Returns per_layer list with k_cos, v_cos, min_cos per layer, plus
    overall min_k_cos, min_v_cos, min_cos.
    """
    from transformers import DynamicCache

    from turboquant_vllm.kv_cache import TurboQuantKVCache

    # Build per-component compressors via TurboQuantKVCache (same path as
    # CompressedDynamicCache but simpler: we re-compress each layer's
    # captured tensors directly without going through the model again).
    dummy_cache = DynamicCache()
    tq = TurboQuantKVCache(dummy_cache, head_dim=head_dim, bits=k_bits)
    tq.restore()
    k_compressor = tq.key_compressor

    dummy_cache2 = DynamicCache()
    tq2 = TurboQuantKVCache(dummy_cache2, head_dim=head_dim, bits=v_bits)
    tq2.restore()
    v_compressor = tq2.value_compressor

    per_layer: list[dict[str, Any]] = []
    for layer_idx, (ref_k, ref_v) in enumerate(captured):
        if ref_k is None or ref_v is None:
            # SWA-bypassed layer: skip and record as None
            per_layer.append(
                {"layer": layer_idx, "skipped": True, "reason": "SWA-bypassed"}
            )
            continue

        comp_k = k_compressor.compress(ref_k)
        decomp_k = k_compressor.decompress(comp_k)

        comp_v = v_compressor.compress(ref_v)
        decomp_v = v_compressor.decompress(comp_v)

        k_cos = _cosine_sim(ref_k, decomp_k)
        v_cos = _cosine_sim(ref_v, decomp_v)

        per_layer.append(
            {
                "layer": layer_idx,
                "k_cos": round(k_cos, 6),
                "v_cos": round(v_cos, 6),
                "min_cos": round(min(k_cos, v_cos), 6),
                "skipped": False,
            }
        )

    active = [e for e in per_layer if not e["skipped"]]

    if not active:
        return {
            "per_layer": per_layer,
            "min_k_cos": None,
            "min_v_cos": None,
            "min_cos": None,
        }

    min_k_cos = min(e["k_cos"] for e in active)
    min_v_cos = min(e["v_cos"] for e in active)
    min_cos = min(e["min_cos"] for e in active)

    return {
        "per_layer": per_layer,
        "min_k_cos": round(min_k_cos, 6),
        "min_v_cos": round(min_v_cos, 6),
        "min_cos": round(min_cos, 6),
    }


# ── Per-model runner ──────────────────────────────────────────────────────────


def _run_model(model_id: str, output_dir: Path) -> dict[str, Any]:
    """Run the full quality matrix for one model.

    Returns a result dict with all configs and per-layer data.
    """
    from transformers import AutoConfig

    print(f"\n{'=' * 72}")
    print(f"Model: {model_id}")
    print(f"{'=' * 72}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    print("  Loading model...", flush=True)
    t0 = time.perf_counter()
    tokenizer, model = _load_model(model_id)
    load_s = time.perf_counter() - t0
    print(f"  Loaded in {load_s:.1f}s")

    model_cfg = _detect_model_config(model)
    head_dim = model_cfg["head_dim"]
    num_layers = model_cfg["num_layers"]
    print(
        f"  head_dim={head_dim}, num_kv_heads={model_cfg['num_kv_heads']}, "
        f"num_layers={num_layers}"
    )

    # Count tokens in the prefill text
    n_tokens = tokenizer(_PREFILL_TEXT, return_tensors="pt")["input_ids"].shape[-1]
    print(f"  Prefill tokens: {n_tokens}")

    # Run real prefill once and capture KV tensors
    print("  Running prefill...", flush=True)
    t1 = time.perf_counter()
    captured = _run_prefill(model, tokenizer, _PREFILL_TEXT, config)
    prefill_s = time.perf_counter() - t1
    active_layers = sum(1 for k, v in captured if k is not None)
    print(
        f"  Prefill done in {prefill_s:.2f}s — {active_layers}/{len(captured)} "
        f"active layers (non-SWA)"
    )

    result: dict[str, Any] = {
        "experiment": "025-asymmetric-quality-matrix",
        "model": model_id,
        "model_type": config.model_type,
        "head_dim": head_dim,
        "num_kv_heads": model_cfg["num_kv_heads"],
        "num_layers": num_layers,
        "active_layers": active_layers,
        "prefill_tokens": n_tokens,
        "prefill_text": _PREFILL_TEXT,
        "load_s": round(load_s, 2),
        "prefill_s": round(prefill_s, 2),
        "configs": {},
    }

    # Compress with each bit config using captured tensors (no model reload)
    for k_bits, v_bits, ac3_gated in _BITS_CONFIGS:
        cfg_key = f"K{k_bits}V{v_bits}"
        print(f"  Computing {cfg_key}...", flush=True)

        t2 = time.perf_counter()
        metrics = _compress_and_measure(captured, head_dim, k_bits, v_bits, config)
        compress_s = time.perf_counter() - t2

        min_cos = metrics["min_cos"]
        passed: bool | None
        if ac3_gated and min_cos is not None:
            passed = min_cos >= AC3_THRESHOLD
        else:
            passed = None  # data collection, no gate

        gate_str = "PASS" if passed is True else ("FAIL" if passed is False else "N/A")
        print(
            f"    min_k={metrics['min_k_cos']:.4f}  "
            f"min_v={metrics['min_v_cos']:.4f}  "
            f"min={min_cos:.4f}  gate={gate_str}  ({compress_s:.2f}s)"
        )

        result["configs"][cfg_key] = {
            "k_bits": k_bits,
            "v_bits": v_bits,
            "ac3_gated": ac3_gated,
            "threshold": AC3_THRESHOLD if ac3_gated else None,
            "passed": passed,
            "compress_s": round(compress_s, 2),
            **metrics,
        }

    # Release model memory before next model
    try:
        model.to("cpu")
    except RuntimeError:
        pass
    del model
    del tokenizer
    del captured
    gc.collect()
    torch.cuda.empty_cache()

    # Write per-model JSON
    safe_name = model_id.replace("/", "--")
    out_path = output_dir / f"experiment-025-{safe_name}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {out_path}")

    return result


# ── Summary table ─────────────────────────────────────────────────────────────


def _print_summary(all_results: list[dict[str, Any]]) -> None:
    """Print a compact summary table with pass/fail per model per config."""
    col_w = 22
    cfg_keys = [f"K{k}V{v}" for k, v, _ in _BITS_CONFIGS]

    # Header
    header = f"{'Model':<{col_w}}"
    for ck in cfg_keys:
        header += f"  {ck:<14}"
    print(f"\n{'=' * 72}")
    print("SUMMARY — AC3 Asymmetric Quality Matrix (real activations)")
    print(f"{'=' * 72}")
    print(header)
    print("-" * 72)

    all_gated_pass = True
    for res in all_results:
        if "error" in res:
            row = f"{res['model']:<{col_w}}  ERROR: {res['error']}"
            print(row)
            all_gated_pass = False
            continue

        short_name = res["model"].split("/")[-1]
        row = f"{short_name:<{col_w}}"
        for ck, (k_bits, v_bits, gated) in zip(cfg_keys, _BITS_CONFIGS):
            cfg = res.get("configs", {}).get(ck, {})
            min_cos = cfg.get("min_cos")
            passed = cfg.get("passed")
            if min_cos is None:
                cell = f"{'ERR':<14}"
            elif not gated:
                cell = f"{min_cos:.4f}(data) "
            elif passed:
                cell = f"{min_cos:.4f} PASS  "
            else:
                cell = f"{min_cos:.4f} FAIL  "
                all_gated_pass = False
            row += f"  {cell}"
        print(row)

    print("-" * 72)
    print(f"Threshold: {AC3_THRESHOLD} (gated configs only)")
    print(
        f"AC3 gate (K4/V3 >= {AC3_THRESHOLD}): {'PASS' if all_gated_pass else 'FAIL'}"
    )
    print(f"{'=' * 72}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for Experiment 025."""
    parser = argparse.ArgumentParser(
        description="Experiment 025: AC3 asymmetric quality matrix (real activations)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Single model ID to test (default: all models)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/logs",
        help="Directory for JSON output files (default: experiments/logs)",
    )
    parser.add_argument(
        "--skip-large-models",
        action="store_true",
        default=False,
        help="Skip models >= 10B parameters (e.g. phi-4 at 14B)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.  Run on the GPU machine.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = [args.model] if args.model else list(_DEFAULT_MODELS)

    _LARGE_MODELS = {"microsoft/phi-4"}
    if args.skip_large_models:
        models_to_run = [m for m in models_to_run if m not in _LARGE_MODELS]
        print(f"Skipping large models: {_LARGE_MODELS}")

    print("Experiment 025 — AC3 Asymmetric Quality Matrix (Real Activations)")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Models: {len(models_to_run)}")
    print("Bit configs: K4/V4, K4/V3, K4/V2")
    print(f"AC3 threshold: {AC3_THRESHOLD}")
    print(f"Output: {output_dir}")

    all_results: list[dict[str, Any]] = []

    for model_id in models_to_run:
        try:
            result = _run_model(model_id, output_dir)
            all_results.append(result)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  OOM: {exc}", file=sys.stderr)
            all_results.append({"model": model_id, "error": f"OOM: {exc}"})
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            all_results.append(
                {"model": model_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            gc.collect()
            torch.cuda.empty_cache()

    # Write combined results
    combined_path = output_dir / "experiment-025-combined.json"
    combined_path.write_text(
        json.dumps(
            {
                "experiment": "025-asymmetric-quality-matrix",
                "models_run": len(models_to_run),
                "models_completed": len([r for r in all_results if "error" not in r]),
                "results": all_results,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Combined results saved to {combined_path}")

    _print_summary(all_results)

    # Exit 0 only if all gated configs passed for all models
    any_fail = any(
        any(cfg.get("passed") is False for cfg in res.get("configs", {}).values())
        for res in all_results
        if "error" not in res
    )
    any_error = any("error" in res for res in all_results)
    sys.exit(1 if (any_fail or any_error) else 0)


if __name__ == "__main__":
    main()
