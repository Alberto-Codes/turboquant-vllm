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
]


@pytest.mark.parametrize("model_id", REGRESSION_MODELS)
def test_model_regression(model_id: str) -> None:
    """Validate compress-decompress cosine parity for a single model."""
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
        head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
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

        ref_cache = DynamicCache()
        for layer_idx in range(num_layers):
            ref_cache.update(fake_keys, fake_values, layer_idx)

        compressed_cache = DynamicCache()
        with CompressedDynamicCache(compressed_cache, head_dim=head_dim, bits=4):
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
                assert layer_cos > COMPRESSION_QUALITY_THRESHOLD, (
                    f"Layer {layer_idx}: cosine {layer_cos:.6f} < {COMPRESSION_QUALITY_THRESHOLD}"
                )
    finally:
        try:
            model.to("cpu")  # ty: ignore[invalid-argument-type]
        except RuntimeError:
            pass  # accelerate-offloaded models can't be moved
        del model
        gc.collect()
        torch.cuda.empty_cache()
