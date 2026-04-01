"""Verify TurboQuant compression quality on a specific model and environment.

Runs a 128-token random Gaussian prefill through CompressedDynamicCache and
reports per-layer cosine similarity vs uncompressed DynamicCache. Outputs
PASS/FAIL against a configurable threshold (default 0.99, compression quality tier).

Validated model families (Molmo2, Mistral, Llama, Qwen2.5, Phi, Gemma 2, Gemma 3)
report ``"validation": "VALIDATED"`` in the output; unvalidated models report
``"UNVALIDATED"`` as a warning.

Usage:
    ```bash
    # Human-readable summary to stdout
    python -m turboquant_vllm.verify --model allenai/Molmo2-4B --bits 4

    # JSON to stdout, human summary to stderr (pipe-friendly)
    python -m turboquant_vllm.verify --model mistralai/Mistral-7B-v0.1 --bits 4 --json
    ```

Examples:
    ```python
    from turboquant_vllm.verify import main

    main(["--model", "allenai/Molmo2-4B", "--bits", "4", "--json"])
    ```

See Also:
    :mod:`turboquant_vllm.benchmark`: Full inference benchmark harness.
    :class:`turboquant_vllm.CompressedDynamicCache`: Compressed cache wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# model_type (from config.model_type) -> display name
# Only models validated by the project are listed here.
# Llama and Mistral are added by stories 2.3 and 2.4 respectively.
VALIDATED_MODELS: dict[str, str] = {
    "molmo2": "Molmo2",
    "mistral": "Mistral",
    "llama": "Llama",
    "qwen2": "Qwen2.5",
    "phi3": "Phi",
    "gemma2": "Gemma 2",
    "gemma3": "Gemma 3",
}

COMPRESSION_QUALITY_THRESHOLD = (
    0.99  # compression quality tier (compressed vs uncompressed)
)


def _detect_model_config(model: Any) -> dict[str, int]:
    """Extract KV cache parameters from a model's config.

    Handles both VLMs (``text_config`` wrapper) and text-only models.
    Falls back to ``hidden_size // num_heads`` when ``head_dim`` is
    absent or ``None`` and ``num_attention_heads`` is positive.

    Args:
        model: A loaded HuggingFace model.

    Returns:
        Dict with head_dim, num_heads, num_kv_heads, num_layers.

    Raises:
        ValueError: If ``head_dim`` is explicit but non-positive, or if
            ``num_attention_heads`` is 0 with no explicit ``head_dim``.
    """
    config = model.config
    # Molmo2 wraps a text config inside the main config
    text_config = getattr(config, "text_config", config)
    hidden_size = text_config.hidden_size
    num_heads = text_config.num_attention_heads
    raw_head_dim = getattr(text_config, "head_dim", None)
    if raw_head_dim is not None:
        if raw_head_dim <= 0:
            msg = f"Model config has head_dim={raw_head_dim}; head_dim must be positive"
            raise ValueError(msg)
        head_dim = raw_head_dim
    elif num_heads == 0:
        msg = (
            "Model config has num_attention_heads=0 and no explicit head_dim; "
            "cannot compute head_dim"
        )
        raise ValueError(msg)
    else:
        head_dim = hidden_size // num_heads
    num_kv_heads = getattr(text_config, "num_key_value_heads", num_heads)
    num_layers = text_config.num_hidden_layers
    return {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
    }


def _run_verification(
    model_id: str,
    bits: int,
    threshold: float,
    *,
    k_bits: int | None = None,
    v_bits: int | None = None,
) -> dict[str, Any]:
    """Run the verification protocol and return a result dict.

    Loads the model, runs a 128-token random Gaussian prefill through both
    uncompressed and compressed caches, and computes per-layer cosine
    similarity. Caches are created with ``DynamicCache(config=config)`` for
    SWA-aware layer instantiation (Gemma models).

    Args:
        model_id: HuggingFace model identifier.
        bits: Quantization bits per coordinate (3 or 4).
        threshold: Minimum cosine similarity for PASS.
        k_bits: Key quantization bits (overrides ``bits`` for keys).
        v_bits: Value quantization bits (overrides ``bits`` for values).

    Returns:
        Dict with model, bits, k_bits, v_bits, status, validation,
        threshold, per_layer_cosine, min_cosine, and versions fields.

    Raises:
        RuntimeError: If cache layers are missing keys or values after
            population (indicates a broken compression pipeline).
    """
    from importlib.metadata import version

    import torch
    from transformers import AutoConfig, DynamicCache

    from turboquant_vllm.kv_cache import CompressedDynamicCache

    # Load config first (cheap, no GPU memory)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = config.model_type

    # Determine validation status
    if model_type in VALIDATED_MODELS:
        validation = "VALIDATED"
        family_name = VALIDATED_MODELS[model_type]
    else:
        validation = "UNVALIDATED"
        family_name = None

    # Detect loader class from config
    is_vlm = hasattr(config, "text_config")
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

    model_cfg = _detect_model_config(model)
    head_dim = model_cfg["head_dim"]
    num_kv_heads = model_cfg["num_kv_heads"]
    num_layers = model_cfg["num_layers"]
    device = next(model.parameters()).device
    text_config = getattr(config, "text_config", config)

    # 128-token random Gaussian prefill
    seq_len = 128
    input_shape = (1, num_kv_heads, seq_len, head_dim)
    fake_keys = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    fake_values = torch.randn(input_shape, dtype=torch.bfloat16, device=device)

    # Uncompressed reference cache (config enables SWA layer types for Gemma)
    ref_cache = DynamicCache(config=config)
    for layer_idx in range(num_layers):
        ref_cache.update(fake_keys, fake_values, layer_idx)

    # Resolve per-component bits
    resolved_k = k_bits if k_bits is not None else bits
    resolved_v = v_bits if v_bits is not None else bits

    # Compressed cache via context manager
    compressed_cache = DynamicCache(config=config)
    with CompressedDynamicCache(
        compressed_cache,
        head_dim=head_dim,
        bits=bits,
        k_bits=k_bits,
        v_bits=v_bits,
        model_config=text_config,
    ):
        for layer_idx in range(num_layers):
            compressed_cache.update(fake_keys, fake_values, layer_idx)

        # Compute per-layer cosine similarity
        per_layer_cosine: list[float] = []
        for layer_idx in range(num_layers):
            ref_layer = ref_cache.layers[layer_idx]
            comp_layer = compressed_cache.layers[layer_idx]

            ref_k = ref_layer.keys
            ref_v = ref_layer.values
            if ref_k is None or ref_v is None:
                raise RuntimeError(
                    f"Reference cache missing keys/values for layer {layer_idx}"
                )

            comp_k = comp_layer.keys
            comp_v = comp_layer.values
            if comp_k is None or comp_v is None:
                raise RuntimeError(
                    f"Compressed cache missing keys/values for layer {layer_idx}"
                )

            k_cos = torch.nn.functional.cosine_similarity(
                ref_k.flatten().float(), comp_k.flatten().float(), dim=0
            ).item()
            v_cos = torch.nn.functional.cosine_similarity(
                ref_v.flatten().float(), comp_v.flatten().float(), dim=0
            ).item()
            per_layer_cosine.append(min(k_cos, v_cos))

    min_cosine = min(per_layer_cosine)
    status = "PASS" if min_cosine >= threshold else "FAIL"

    versions = {
        "turboquant_vllm": version("turboquant-vllm"),
        "transformers": version("transformers"),
        "torch": version("torch"),
    }

    result: dict[str, Any] = {
        "model": model_id,
        "bits": bits,
        "k_bits": resolved_k,
        "v_bits": resolved_v,
        "status": status,
        "validation": validation,
        "threshold": threshold,
        "per_layer_cosine": per_layer_cosine,
        "min_cosine": min_cosine,
        "versions": versions,
    }
    if family_name is not None:
        result["family_name"] = family_name

    return result


def _format_human_summary(result: dict[str, Any]) -> str:
    """Format a human-readable verification summary.

    Displays asymmetric K/V bits when ``k_bits != v_bits``.

    Args:
        result: Result dict from _run_verification.

    Returns:
        Multi-line human-readable summary string.
    """
    lines = []
    lines.append(f"Model: {result['model']}")
    if result["k_bits"] != result["v_bits"]:
        lines.append(f"K bits: {result['k_bits']}, V bits: {result['v_bits']}")
    else:
        lines.append(f"Bits: {result['bits']}")
    lines.append(f"Validation: {result['validation']}")
    if "family_name" in result:
        lines.append(f"Family: {result['family_name']}")

    per_layer = result["per_layer_cosine"]
    num_layers = len(per_layer)
    if num_layers <= 8:
        for i, cos in enumerate(per_layer):
            lines.append(f"  Layer {i}: {cos:.6f}")
    else:
        for i in range(3):
            lines.append(f"  Layer {i}: {per_layer[i]:.6f}")
        lines.append(f"  ... ({num_layers - 6} more layers)")
        for i in range(num_layers - 3, num_layers):
            lines.append(f"  Layer {i}: {per_layer[i]:.6f}")

    lines.append(f"Min cosine: {result['min_cosine']:.6f}")
    lines.append(f"Threshold: {result['threshold']}")
    lines.append(f"Result: {result['status']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the verify command.

    Parses ``--model``, ``--bits`` (or ``--k-bits`` and ``--v-bits`` together),
    ``--threshold`` (default 0.99), and ``--json`` flags, runs verification,
    and exits 0 (PASS) or 1 (FAIL).  ``--k-bits`` and ``--v-bits`` must be
    used together; ``--bits`` cannot be combined with per-component flags.

    Args:
        argv: Command-line arguments. Uses sys.argv[1:] if None.
    """
    parser = argparse.ArgumentParser(
        description="Verify TurboQuant compression quality on a model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., allenai/Molmo2-4B)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[2, 3, 4, 5],
        default=None,
        help="Quantization bits per coordinate (shorthand for --k-bits and --v-bits)",
    )
    parser.add_argument(
        "--k-bits",
        type=int,
        choices=[2, 3, 4, 5],
        default=None,
        help="Key quantization bits (overrides --bits for keys)",
    )
    parser.add_argument(
        "--v-bits",
        type=int,
        choices=[2, 3, 4, 5],
        default=None,
        help="Value quantization bits (overrides --bits for values)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=COMPRESSION_QUALITY_THRESHOLD,
        help=f"Minimum cosine similarity for PASS (default: {COMPRESSION_QUALITY_THRESHOLD})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output JSON to stdout (human summary to stderr)",
    )

    args = parser.parse_args(argv)

    # Validate argument combinations
    if args.bits is not None and (args.k_bits is not None or args.v_bits is not None):
        parser.error("--bits cannot be used with --k-bits or --v-bits")
    if args.bits is None and args.k_bits is None and args.v_bits is None:
        parser.error("Specify --bits or --k-bits/--v-bits")
    if (args.k_bits is None) != (args.v_bits is None):
        parser.error("--k-bits and --v-bits must be used together")

    result = _run_verification(
        args.model,
        args.bits if args.bits is not None else args.k_bits,
        args.threshold,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
    )
    human_summary = _format_human_summary(result)

    if args.json_output:
        # JSON to stdout, human to stderr
        print(json.dumps(result, indent=2), file=sys.stdout)
        print(human_summary, file=sys.stderr)
    else:
        # Human-readable to stdout only
        print(human_summary, file=sys.stdout)

    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
