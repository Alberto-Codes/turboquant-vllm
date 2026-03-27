r"""Experiment 012 -- TurboQuantProd quality validation for P9.

P9 Phase 1 gate: validates that TurboQuantProd (3-bit MSE + 1-bit QJL)
with ``estimate_inner_product`` produces coherent output on Molmo2-4B.

Experiment 001 showed TurboQuantProd FAILS in drop-in mode (dequant then
standard attention). P9 uses ``estimate_inner_product`` instead — the QJL
correction fixes inner product bias. This experiment tests that path.

Three comparison paths:
    1. SDPA baseline (fp16 KV).
    2. TurboQuantMSE 4-bit (current working approach, drop-in).
    3. TurboQuantProd 4-bit (3MSE+1QJL) with estimate_inner_product for Q@K^T.

Usage:
    ```bash
    uv run python experiments/experiment_012_turboquantprod_quality.py
    ```

Examples:
    ```bash
    uv run python experiments/experiment_012_turboquantprod_quality.py
    ```

See Also:
    :mod:`turboquant_consumer.quantizer`: TurboQuantProd.estimate_inner_product.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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


def _run_inference(
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int,
    label: str,
) -> dict[str, Any]:
    """Run inference and collect metrics.

    Returns:
        Dict with output text, token IDs, timing.
    """
    content = [{"type": "text", "text": prompt}]
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

    print(f"  [{label}] {output_len} tokens, {tok_s:.1f} tok/s, {elapsed:.1f}s")
    print(f"    {output_text[:120]}...")

    return {
        "label": label,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "output_text": output_text,
        "output_token_ids": generated_ids.tolist(),
        "vram_peak_mib": round(vram_peak, 1),
        "tok_per_s": round(tok_s, 2),
    }


def run_experiment(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    bits: int,
) -> dict[str, Any]:
    """Run the 3-path TurboQuantProd quality comparison.

    Returns:
        Dict with results for each path.
    """
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        DynamicCache,
    )

    from turboquant_consumer.kv_cache import CompressedDynamicCache

    results: dict[str, Any] = {
        "experiment": "012-turboquantprod-quality",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "bits": bits,
    }

    # Load model
    print(f"\nLoading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
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
    print(f"  head_dim={head_dim}, bits={bits}")

    # Path 1: SDPA baseline
    print("\n--- Path 1: SDPA baseline ---")
    sdpa = _run_inference(model, processor, prompt, max_new_tokens, "SDPA")
    results["sdpa"] = sdpa

    # Path 2: TurboQuantMSE 4-bit (known working)
    print("\n--- Path 2: TurboQuantMSE 4-bit (drop-in) ---")
    original_init = DynamicCache.__init__
    mse_wrappers: list[CompressedDynamicCache] = []

    def mse_patched_init(self_cache: Any, *a: Any, **kw: Any) -> None:
        """Wrap with MSE compressor.

        Other Parameters:
            **kw: Forwarded to original init.
        """
        original_init(self_cache, *a, **kw)
        mse_wrappers.append(
            CompressedDynamicCache(self_cache, head_dim=head_dim, bits=bits)
        )

    DynamicCache.__init__ = mse_patched_init  # type: ignore[method-assign]
    try:
        mse_result = _run_inference(
            model, processor, prompt, max_new_tokens, "TQ-MSE-4bit"
        )
    finally:
        DynamicCache.__init__ = original_init  # type: ignore[method-assign]
    results["tq_mse"] = mse_result

    # Path 3: TurboQuantProd 4-bit with estimate_inner_product
    # This patches DynamicCache to store TurboQuantProd compressed keys
    # and replaces Q@K^T with estimate_inner_product in attention.
    print("\n--- Path 3: TurboQuantProd 4-bit (estimate_inner_product) ---")

    # We need a custom attention function that uses estimate_inner_product
    # for Q@K^T instead of standard matmul. We'll monkey-patch via
    # AttentionInterface.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Create shared quantizer
    from turboquant_consumer.quantizer import TurboQuantProd

    prod_quantizer = TurboQuantProd(head_dim, bits=bits, seed=42)

    # Storage for compressed keys per layer
    prod_keys: dict[int, dict[str, torch.Tensor]] = {}

    def prod_cache_update(
        self_cache: Any,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache update that stores TurboQuantProd compressed keys.

        Returns:
            ``(keys, values)`` — keys are decompressed (for V matmul shape compat),
            values are uncompressed.
        """
        B, H, S, D = key_states.shape

        # Compress keys with TurboQuantProd
        flat_k = key_states.float().reshape(-1, D)
        q = prod_quantizer
        q.mse_quantizer.rotation = q.mse_quantizer.rotation.to(flat_k.device)
        q.qjl_matrix = q.qjl_matrix.to(flat_k.device)

        indices, norms, qjl_signs, residual_norms = q.quantize(flat_k)

        # Store compressed data for this layer
        entry = {
            "indices": indices.reshape(B, H, S, D),
            "norms": norms.reshape(B, H, S, 1),
            "qjl_signs": qjl_signs.reshape(B, H, S, -1),
            "residual_norms": residual_norms.reshape(B, H, S, 1),
        }

        if layer_idx in prod_keys:
            for k_name in entry:
                prod_keys[layer_idx][k_name] = torch.cat(
                    [prod_keys[layer_idx][k_name], entry[k_name]], dim=2
                )
        else:
            prod_keys[layer_idx] = entry

        # Still call original update for value caching + seq length tracking
        return original_init_update(
            self_cache, key_states, value_states, layer_idx, cache_kwargs
        )

    # Attention function using estimate_inner_product for Q@K^T
    def prod_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
        dropout: float = 0.0,
        scaling: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Attention using estimate_inner_product for Q@K^T.

        Other Parameters:
            **kwargs: Additional model-specific arguments (ignored).

        Returns:
            ``(output, None)`` with output transposed to HF convention.
        """
        layer_idx = getattr(module, "layer_idx", 0)
        B, H_Q, S_Q, D = query.shape
        H_KV = key.shape[1]
        n_groups = H_Q // H_KV

        if layer_idx not in prod_keys:
            # Fallback to standard attention
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )
            return out.transpose(1, 2).contiguous(), None

        comp = prod_keys[layer_idx]
        S_KV = comp["indices"].shape[2]
        q = prod_quantizer

        # Compute attention scores via estimate_inner_product
        # For each query head group sharing a KV head
        all_scores = []
        for kv_h in range(H_KV):
            idx = comp["indices"][:, kv_h]
            nrm = comp["norms"][:, kv_h]
            sgn = comp["qjl_signs"][:, kv_h]
            rnm = comp["residual_norms"][:, kv_h]

            for g in range(n_groups):
                q_h = kv_h * n_groups + g
                qry = query[:, q_h]

                # Expand for broadcasting: [B, S_Q, 1, D] vs [B, 1, S_KV, D]
                q_exp = qry.float().unsqueeze(2).expand(B, S_Q, S_KV, D)
                idx_exp = idx.unsqueeze(1).expand(B, S_Q, S_KV, D)
                nrm_exp = nrm.unsqueeze(1).expand(B, S_Q, S_KV, 1)
                sgn_exp = sgn.unsqueeze(1).expand(B, S_Q, S_KV, sgn.shape[-1])
                rnm_exp = rnm.unsqueeze(1).expand(B, S_Q, S_KV, 1)

                scores = q.estimate_inner_product(
                    q_exp, idx_exp, nrm_exp, sgn_exp, rnm_exp
                )
                all_scores.append(scores.squeeze(-1))

        # Stack to [B, H_Q, S_Q, S_KV]
        attn_scores = torch.stack(all_scores, dim=1)

        if scaling is not None:
            attn_scores = attn_scores * scaling
        else:
            attn_scores = attn_scores / (D**0.5)

        # Diagnostic: compare estimated vs reference scores at layer 0
        if layer_idx == 0 and not hasattr(prod_attention_forward, "_diag_done"):
            prod_attention_forward._diag_done = True  # type: ignore[attr-defined]
            # Reference: standard Q @ K^T * scaling
            key_expanded = key.repeat_interleave(n_groups, dim=1)
            ref_scores = torch.matmul(query, key_expanded.transpose(2, 3)).float()
            if scaling is not None:
                ref_scores = ref_scores * scaling
            else:
                ref_scores = ref_scores / (D**0.5)

            cos = F.cosine_similarity(
                attn_scores.flatten(), ref_scores.flatten(), dim=0
            ).item()
            e_mean, e_std = attn_scores.mean().item(), attn_scores.std().item()
            r_mean, r_std = ref_scores.mean().item(), ref_scores.std().item()
            print(f"\n    [DIAG L0] estimated: mean={e_mean:.3f} std={e_std:.3f}")
            print(f"    [DIAG L0] reference: mean={r_mean:.3f} std={r_std:.3f}")
            print(f"    [DIAG L0] cosine similarity: {cos:.6f}")
            print(
                f"    [DIAG L0] max abs diff: {(attn_scores - ref_scores).abs().max():.3f}"
            )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :S_Q, :S_KV]
            attn_scores = attn_scores + causal_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query.dtype)

        # V matmul with GQA expansion
        value_expanded = value.repeat_interleave(n_groups, dim=1)
        out = torch.matmul(attn_weights, value_expanded)
        out = out.transpose(1, 2).contiguous()
        return out, None

    # Install the custom attention + cache
    original_impl = model.config._attn_implementation
    ALL_ATTENTION_FUNCTIONS.register("prod_estimate", prod_attention_forward)
    model.config._attn_implementation = "prod_estimate"

    # Patch DynamicCache to use prod_cache_update
    original_init_update = DynamicCache.update
    DynamicCache.update = prod_cache_update  # type: ignore[assignment]

    try:
        prod_result = _run_inference(
            model, processor, prompt, max_new_tokens, "TQ-Prod-4bit"
        )
    finally:
        DynamicCache.update = original_init_update  # type: ignore[assignment]
        model.config._attn_implementation = original_impl
        prod_keys.clear()

    results["tq_prod"] = prod_result

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    for name, res in [("TQ-MSE", mse_result), ("TQ-Prod", prod_result)]:
        b_ids = sdpa["output_token_ids"]
        e_ids = res["output_token_ids"]
        n = min(len(b_ids), len(e_ids))
        match = sum(1 for a, b in zip(b_ids[:n], e_ids[:n]) if a == b)
        rate = match / n if n > 0 else 0
        same = sdpa["output_text"].strip() == res["output_text"].strip()
        print(
            f"  SDPA → {name}: {match}/{n} ({rate:.1%}), text={'same' if same else 'DIFF'}"
        )

    # Check coherence (not garbled)
    prod_text = prod_result["output_text"]
    is_coherent = len(prod_text) > 20 and not any(
        x in prod_text.lower() for x in ["�", "\x00", "unk"]
    )
    results["prod_coherent"] = is_coherent
    print(f"\n  TQ-Prod coherent: {is_coherent}")
    print(f"  TQ-Prod output:   {prod_text[:200]}")

    return results


def main() -> None:
    """CLI entry point for Experiment 012."""
    parser = argparse.ArgumentParser(
        description="Experiment 012: TurboQuantProd quality validation"
    )
    parser.add_argument("--model", default="allenai/Molmo2-4B")
    parser.add_argument(
        "--prompt",
        default="Name the main characters of Seinfeld and describe each one briefly.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--output",
        default="experiments/logs/experiment-012-turboquantprod-quality.json",
    )
    args = parser.parse_args()

    results = run_experiment(
        model_id=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        bits=args.bits,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    sys.exit(0 if results.get("prod_coherent", False) else 1)


if __name__ == "__main__":
    main()
