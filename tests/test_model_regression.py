"""Model regression tests for pre-release validation.

Validates compression quality across all supported model families by running
a 128-token random Gaussian prefill through CompressedDynamicCache and checking
per-layer cosine similarity against an uncompressed DynamicCache reference.

Usage:
    uv run pytest tests/test_model_regression.py -v
"""

from __future__ import annotations

import gc

import pytest
import torch
from transformers import AutoConfig, DynamicCache

from turboquant_vllm.kv_cache import CompressedDynamicCache

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

COMPRESSION_QUALITY_THRESHOLD = 0.99  # compression quality tier -- see architecture doc

REGRESSION_MODELS = [
    pytest.param("allenai/Molmo2-4B", id="molmo2-4b"),
    pytest.param("mistralai/Mistral-7B-v0.1", id="mistral-7b"),
    pytest.param("meta-llama/Llama-3.1-8B", id="llama-3.1-8b"),
    pytest.param("Qwen/Qwen2.5-3B", id="qwen2.5-3b"),
    pytest.param("microsoft/phi-4", id="phi-4"),
    pytest.param("microsoft/Phi-3-mini-4k-instruct", id="phi-3-mini"),
    pytest.param("google/gemma-2-2b", id="gemma-2-2b"),
    pytest.param("google/gemma-3-4b-it", id="gemma-3-4b"),
]

# Asymmetric bit configs: (k_bits, v_bits, threshold, is_gated)
# K4/V4 covered by test_model_regression() — not duplicated here.
# K4/V3: random data gate at 0.97 (3-bit V on Gaussian data has inherently
#         lower quality; AC3 release gate of 0.99 applies to real activations)
# K4/V2: data collection only, no gate (publish whatever the numbers are)
BITS_CONFIGS = [
    pytest.param(4, 3, 0.97, True, id="k4v3"),
    pytest.param(4, 2, 0.0, False, id="k4v2-data"),
]


def _run_model_regression(
    model_id: str, k_bits: int, v_bits: int, threshold: float, *, gated: bool
) -> None:
    """Run regression test for a single model + bit config."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

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

    try:
        text_config = getattr(config, "text_config", config)
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        num_kv_heads = getattr(
            text_config, "num_key_value_heads", text_config.num_attention_heads
        )
        num_layers = text_config.num_hidden_layers
        device = next(model.parameters()).device

        seq_len = 128
        input_shape = (1, num_kv_heads, seq_len, head_dim)
        fake_keys = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
        fake_values = torch.randn(input_shape, dtype=torch.bfloat16, device=device)

        ref_cache = DynamicCache(config=config)
        for layer_idx in range(num_layers):
            ref_cache.update(fake_keys, fake_values, layer_idx)

        compressed_cache = DynamicCache(config=config)
        with CompressedDynamicCache(
            compressed_cache,
            head_dim=head_dim,
            k_bits=k_bits,
            v_bits=v_bits,
            model_config=text_config,
        ):
            for layer_idx in range(num_layers):
                compressed_cache.update(fake_keys, fake_values, layer_idx)

            for layer_idx in range(num_layers):
                ref_k = ref_cache.layers[layer_idx].keys
                comp_k = compressed_cache.layers[layer_idx].keys
                ref_v = ref_cache.layers[layer_idx].values
                comp_v = compressed_cache.layers[layer_idx].values
                assert ref_k is not None and comp_k is not None
                assert ref_v is not None and comp_v is not None

                k_cos = torch.nn.functional.cosine_similarity(
                    ref_k.flatten().float(), comp_k.flatten().float(), dim=0
                ).item()
                v_cos = torch.nn.functional.cosine_similarity(
                    ref_v.flatten().float(), comp_v.flatten().float(), dim=0
                ).item()
                layer_cos = min(k_cos, v_cos)
                if gated:
                    assert layer_cos >= threshold, (
                        f"Layer {layer_idx}: K{k_bits}/V{v_bits} cosine "
                        f"{layer_cos:.6f} < {threshold}"
                    )
    finally:
        try:
            model.to("cpu")  # ty: ignore[invalid-argument-type]
        except RuntimeError:
            pass  # accelerate-offloaded models can't be moved
        del model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("model_id", REGRESSION_MODELS)
def test_model_regression(model_id: str) -> None:
    """Validate K4/V4 compress-decompress cosine parity (existing baseline)."""
    _run_model_regression(model_id, 4, 4, COMPRESSION_QUALITY_THRESHOLD, gated=True)


@pytest.mark.parametrize("model_id", REGRESSION_MODELS)
@pytest.mark.parametrize(("k_bits", "v_bits", "threshold", "gated"), BITS_CONFIGS)
def test_model_regression_asymmetric(
    model_id: str, k_bits: int, v_bits: int, threshold: float, gated: bool
) -> None:
    """Asymmetric quality matrix: 8 models x 2 configs (K4/V3, K4/V2)."""
    _run_model_regression(model_id, k_bits, v_bits, threshold, gated=gated)
