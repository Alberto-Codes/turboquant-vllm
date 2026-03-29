"""Regression tests for precision at realistic sequence lengths.

These tests simulate multi-layer transformer KV cache usage at
1000+ tokens across 36 layers to catch precision bugs (like the
fp16 norms issue) that don't manifest in short-sequence unit tests.
"""

from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F

from turboquant_vllm.kv_cache import CompressedDynamicCache, TurboQuantKVCache

from .conftest import BITS, DIM

_NUM_LAYERS = 36
_NUM_HEADS = 8
_PREFILL_LEN = 1024
_GEN_STEPS = 32

_WRAPPER_CLS_MAP: dict[str, type] = {
    "TurboQuantKVCache": TurboQuantKVCache,
    "CompressedDynamicCache": CompressedDynamicCache,
}


@functools.lru_cache(maxsize=None)
def _cached_regression_run(
    cls_name: str,
    bits: int,
) -> tuple[torch.Tensor, ...]:
    """Run prefill + generation simulation, cached by (cls_name, bits).

    Args:
        cls_name: Name of wrapper class (key into ``_WRAPPER_CLS_MAP``).
        bits: Quantization bit width.

    Returns:
        Tuple of decompressed key tensors (one per layer) for the final
        generation step.
    """
    from transformers import DynamicCache

    torch.manual_seed(42)

    wrapper_cls = _WRAPPER_CLS_MAP[cls_name]
    cache = DynamicCache()
    _ = wrapper_cls(cache, head_dim=DIM, bits=bits)

    for layer_idx in range(_NUM_LAYERS):
        k = torch.randn(1, _NUM_HEADS, _PREFILL_LEN, DIM)
        v = torch.randn(1, _NUM_HEADS, _PREFILL_LEN, DIM)
        cache.update(k, v, layer_idx=layer_idx)

    final_keys: list[torch.Tensor] = []
    for step in range(_GEN_STEPS):
        step_keys = []
        for layer_idx in range(_NUM_LAYERS):
            k = torch.randn(1, _NUM_HEADS, 1, DIM)
            v = torch.randn(1, _NUM_HEADS, 1, DIM)
            out_k, _ = cache.update(k, v, layer_idx=layer_idx)
            step_keys.append(out_k)
        final_keys = step_keys

    return tuple(final_keys)


@pytest.mark.unit
@pytest.mark.slow
class TestLongSequenceRegression:
    """Regression tests for precision at realistic sequence lengths."""

    NUM_LAYERS = _NUM_LAYERS
    NUM_HEADS = _NUM_HEADS
    PREFILL_LEN = _PREFILL_LEN
    GEN_STEPS = _GEN_STEPS

    def _run_prefill_and_generate(
        self,
        wrapper_cls: type,
        bits: int = BITS,
    ) -> list[torch.Tensor]:
        """Simulate prefill + generation through all layers.

        Args:
            wrapper_cls: TurboQuantKVCache or CompressedDynamicCache.
            bits: Quantization bit width.

        Returns:
            List of decompressed key tensors (one per layer) after
            prefill + generation steps, for the final generation step.
        """
        return list(_cached_regression_run(wrapper_cls.__name__, bits))

    def test_compressed_matches_accuracy_only(self) -> None:
        """CompressedDynamicCache should closely match TurboQuantKVCache.

        Both use TurboQuantCompressorMSE at the same bit-width. The
        only difference is storage format (uint8+fp32 vs fp32 round-trip).
        After 36 layers and 1000+ tokens, the outputs should still be
        nearly identical. A cosine similarity below 0.999 per layer
        would indicate a precision regression.
        """
        keys_accuracy = self._run_prefill_and_generate(TurboQuantKVCache)
        keys_compressed = self._run_prefill_and_generate(CompressedDynamicCache)

        assert len(keys_accuracy) == self.NUM_LAYERS
        assert len(keys_compressed) == self.NUM_LAYERS

        for layer_idx in range(self.NUM_LAYERS):
            ka = keys_accuracy[layer_idx]
            kc = keys_compressed[layer_idx]
            assert ka.shape == kc.shape, (
                f"Layer {layer_idx}: shape mismatch {ka.shape} vs {kc.shape}"
            )

            cos = F.cosine_similarity(ka.flatten(), kc.flatten(), dim=0)
            assert cos > 0.999, (
                f"Layer {layer_idx}: cosine similarity {cos:.6f} < 0.999 — "
                f"precision regression detected"
            )

    def test_no_nan_or_inf_at_scale(self) -> None:
        """No NaN or Inf should appear after many layers and tokens."""
        keys = self._run_prefill_and_generate(CompressedDynamicCache)

        for layer_idx, k in enumerate(keys):
            assert not k.isnan().any(), f"Layer {layer_idx}: NaN in keys"
            assert not k.isinf().any(), f"Layer {layer_idx}: Inf in keys"

    def test_seq_length_tracks_correctly(self) -> None:
        """get_seq_length() should return prefill + generation tokens."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # Prefill
        for layer_idx in range(self.NUM_LAYERS):
            cache.update(
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                layer_idx=layer_idx,
            )

        assert cache.get_seq_length(0) == self.PREFILL_LEN

        # Generate 10 tokens
        for _ in range(10):
            for layer_idx in range(self.NUM_LAYERS):
                cache.update(
                    torch.randn(1, self.NUM_HEADS, 1, DIM),
                    torch.randn(1, self.NUM_HEADS, 1, DIM),
                    layer_idx=layer_idx,
                )

        assert cache.get_seq_length(0) == self.PREFILL_LEN + 10

    def test_compression_stats_at_scale(self) -> None:
        """Compression stats should reflect realistic cache sizes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        for layer_idx in range(self.NUM_LAYERS):
            cache.update(
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                layer_idx=layer_idx,
            )

        stats = cc.compression_stats()
        assert stats["num_layers"] == self.NUM_LAYERS
        assert stats["seq_len"] == self.PREFILL_LEN
        assert stats["num_heads"] == self.NUM_HEADS
        assert stats["compression_ratio"] > 1.9
        # At 36 layers × 8 heads × 1024 tokens, savings should be substantial
        assert stats["savings_mib"] > 50

    def test_cache_hits_for_regression_run(self) -> None:
        """Verify _cached_regression_run produces cache hits across tests."""
        _cached_regression_run.cache_clear()
        self._run_prefill_and_generate(TurboQuantKVCache)
        self._run_prefill_and_generate(CompressedDynamicCache)
        # Second call with same args should hit cache
        self._run_prefill_and_generate(TurboQuantKVCache)
        info = _cached_regression_run.cache_info()
        assert info.hits > 0, (
            f"Expected cache hits but got {info.hits} "
            f"(misses={info.misses}, size={info.currsize})"
        )

    def test_4bit_nibble_at_scale(self) -> None:
        """TQ4 nibble-packed should work at 36 layers with 1024 tokens."""
        keys_accuracy = self._run_prefill_and_generate(TurboQuantKVCache, bits=4)
        keys_compressed = self._run_prefill_and_generate(CompressedDynamicCache, bits=4)

        for layer_idx in range(self.NUM_LAYERS):
            cos = F.cosine_similarity(
                keys_accuracy[layer_idx].flatten(),
                keys_compressed[layer_idx].flatten(),
                dim=0,
            )
            assert cos > 0.999, (
                f"Layer {layer_idx}: TQ4 cosine similarity {cos:.6f} < 0.999"
            )
