"""Tests for sliding window attention bypass in CompressedDynamicCache."""

from __future__ import annotations

import warnings

import pytest
import torch

from turboquant_vllm.kv_cache import CompressedDynamicCache

from .conftest import BITS, DIM

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Minimal config stubs (plain classes, no Mock)
# ---------------------------------------------------------------------------


class _GemmaLikeConfig:
    """Gemma pattern: mixed global + sliding window layers."""

    sliding_window = 4096
    layer_types = ["sliding_attention", "full_attention"] * 3


class _MistralLikeConfig:
    """Mistral pattern: uniform SWA, no layer_types distinction."""

    sliding_window = 4096
    layer_types = None


class _LlamaLikeConfig:
    """Llama pattern: no SWA at all."""

    sliding_window = None
    layer_types = None


# ---------------------------------------------------------------------------
# SWA bypass tests
# ---------------------------------------------------------------------------


class TestSWABypass:
    """Validate sliding window layer compression behavior.

    The is_sliding SWA bypass was removed — it was dead code due to
    DynamicCache lazy initialization (cache starts with 0 layers).
    SWA layers now compress normally. Full attention layers in mixed
    models are bypassed via _full_attn_bypass (config-based).
    """

    def test_swa_layer_compresses_without_model_config(
        self, device: torch.device
    ) -> None:
        """SWA layers compress when no model_config is provided."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.layers.clear()
        cache.layers.extend(
            [DynamicLayer(), DynamicSlidingWindowLayer(sliding_window=4096)]
        )

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        # Layer 1 (SWA) — compresses normally (no model_config = no bypass)
        cache.update(key, val, layer_idx=1)
        assert len(cdc._compressed_keys) >= 2
        assert cdc._compressed_keys[1] is not None

    def test_global_layer_still_compresses_with_swa_present(
        self, device: torch.device
    ) -> None:
        """Global layers are compressed even when SWA layers exist."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.layers.clear()
        cache.layers.extend(
            [DynamicLayer(), DynamicSlidingWindowLayer(sliding_window=4096)]
        )

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        # Layer 0 (global) — should compress
        cache.update(key, val, layer_idx=0)

        assert len(cdc._compressed_keys) == 1

    def test_mixed_global_swa_per_layer_behavior(self, device: torch.device) -> None:
        """Alternating global/SWA layers get correct per-layer treatment."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # Gemma-2-like: alternating global/SWA
        cache.layers.clear()
        cache.layers.extend(
            [
                DynamicLayer(),
                DynamicSlidingWindowLayer(sliding_window=4096),
                DynamicLayer(),
                DynamicSlidingWindowLayer(sliding_window=4096),
            ]
        )

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        # Two decode steps to verify index stability across steps
        for _step in range(2):
            for idx in range(4):
                cache.update(key.clone(), val.clone(), layer_idx=idx)

        # All 4 layers compressed (SWA bypass removed, no model_config).
        assert len(cdc._compressed_keys) == 4
        assert all(ck is not None for ck in cdc._compressed_keys)
        # Each layer has 2 tokens (from 2 decode steps)
        ck0 = cdc._compressed_keys[0]
        ck2 = cdc._compressed_keys[2]
        assert ck0 is not None
        assert ck2 is not None
        assert ck0.indices.shape[-2] == 2
        assert ck2.indices.shape[-2] == 2

    def test_lazy_layer_never_triggers_bypass(self, device: torch.device) -> None:
        """Layers beyond current list (lazy creation) should compress, not bypass."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # No layers in list — layer_idx 0 is beyond len(cache.layers)
        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        cache.update(key, val, layer_idx=0)

        # Should have compressed (not bypassed)
        assert len(cdc._compressed_keys) == 1

    def test_get_seq_length_swa_layer_compressed(self, device: torch.device) -> None:
        """get_seq_length on SWA layer returns compressed count (SWA bypass removed)."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.layers.clear()
        cache.layers.extend(
            [DynamicLayer(), DynamicSlidingWindowLayer(sliding_window=4096)]
        )

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        # Update both layers
        cache.update(key.clone(), val.clone(), layer_idx=0)
        cache.update(key.clone(), val.clone(), layer_idx=1)

        # SWA layer should report actual seq length (1), not 0
        assert cache.get_seq_length(1) == 1
        # Global layer also reports 1
        assert cache.get_seq_length(0) == 1

    def test_get_compressed_swa_layer_works(self, device: torch.device) -> None:
        """get_compressed on SWA layer returns data (SWA bypass removed)."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.layers.clear()
        cache.layers.extend(
            [DynamicLayer(), DynamicSlidingWindowLayer(sliding_window=4096)]
        )

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        cache.update(key.clone(), val.clone(), layer_idx=0)
        cache.update(key.clone(), val.clone(), layer_idx=1)

        # Both layers compressed (SWA bypass removed)
        k_p, k_n, v_p, v_n = cdc.get_compressed(0)
        assert k_p is not None
        k_p1, k_n1, v_p1, v_n1 = cdc.get_compressed(1)
        assert k_p1 is not None

    def test_vram_bytes_excludes_swa_layers(self, device: torch.device) -> None:
        """VRAM calculation skips None entries from SWA-bypassed layers."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.layers.clear()
        cache.layers.extend(
            [DynamicLayer(), DynamicSlidingWindowLayer(sliding_window=4096)]
        )

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        cache.update(key.clone(), val.clone(), layer_idx=0)
        cache.update(key.clone(), val.clone(), layer_idx=1)

        # Should not raise (None entries are skipped)
        vram = cdc.vram_bytes()
        assert vram > 0

    def test_compression_stats_with_swa_gaps(self, device: torch.device) -> None:
        """compression_stats counts only compressed layers, not None entries."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.layers.clear()
        cache.layers.extend(
            [
                DynamicLayer(),
                DynamicSlidingWindowLayer(sliding_window=4096),
                DynamicLayer(),
                DynamicSlidingWindowLayer(sliding_window=4096),
            ]
        )

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        for idx in range(4):
            cache.update(key.clone(), val.clone(), layer_idx=idx)

        stats = cdc.compression_stats()
        # All 4 layers compressed (SWA bypass removed)
        assert stats["num_layers"] == 4


# ---------------------------------------------------------------------------
# Defensive warning tests
# ---------------------------------------------------------------------------


class TestSWAWarning:
    """Validate SWA detection warnings in __init__()."""

    def test_gemma_config_without_swa_layers_warns(self) -> None:
        """Warning fires when config has sliding layer_types but cache lacks SWA layers."""
        from transformers import DynamicCache

        cache = DynamicCache()  # lazy mode — no SWA layers

        with pytest.warns(UserWarning, match="sliding window layer metadata"):
            CompressedDynamicCache(
                cache, head_dim=DIM, bits=BITS, model_config=_GemmaLikeConfig()
            )

    def test_no_warning_when_model_config_is_none(self) -> None:
        """No warning when model_config is not provided."""
        from transformers import DynamicCache

        cache = DynamicCache()

        # Should NOT warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
            swa_warnings = [x for x in w if "sliding window" in str(x.message)]
            assert len(swa_warnings) == 0
        assert cdc.enabled

    def test_no_warning_when_swa_layers_present(self) -> None:
        """No warning when cache correctly has SWA layers."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        cache.layers.clear()
        cache.layers.extend(
            [
                DynamicLayer(),
                DynamicSlidingWindowLayer(sliding_window=4096),
            ]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CompressedDynamicCache(
                cache, head_dim=DIM, bits=BITS, model_config=_GemmaLikeConfig()
            )
            swa_warnings = [x for x in w if "sliding window" in str(x.message)]
            assert len(swa_warnings) == 0

    def test_mistral_config_no_warning(self) -> None:
        """No warning for Mistral — uniform SWA, layer_types is None."""
        from transformers import DynamicCache

        cache = DynamicCache()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cdc = CompressedDynamicCache(
                cache, head_dim=DIM, bits=BITS, model_config=_MistralLikeConfig()
            )
            swa_warnings = [x for x in w if "sliding window" in str(x.message)]
            assert len(swa_warnings) == 0
        assert cdc.enabled

    def test_llama_config_no_warning(self) -> None:
        """No warning for Llama — no SWA at all."""
        from transformers import DynamicCache

        cache = DynamicCache()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cdc = CompressedDynamicCache(
                cache, head_dim=DIM, bits=BITS, model_config=_LlamaLikeConfig()
            )
            swa_warnings = [x for x in w if "sliding window" in str(x.message)]
            assert len(swa_warnings) == 0
        assert cdc.enabled


class TestFullAttentionBypass:
    """Full attention layer bypass for sliding window model quality."""

    def test_swa_compresses_full_attn_bypassed(self, device: torch.device) -> None:
        """SWA layers compress, full attn layers bypass via config."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

        cache = DynamicCache()
        cdc = CompressedDynamicCache(
            cache, head_dim=DIM, bits=BITS, model_config=_GemmaLikeConfig()
        )

        # Gemma pattern: [sliding, full, sliding, full]
        cache.layers.clear()
        cache.layers.extend(
            [
                DynamicSlidingWindowLayer(sliding_window=4096),
                DynamicLayer(),
                DynamicSlidingWindowLayer(sliding_window=4096),
                DynamicLayer(),
            ]
        )

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        for idx in range(4):
            cache.update(key.clone(), val.clone(), layer_idx=idx)

        # _GemmaLikeConfig: layer_types = [sliding, full] * 3
        # full_attn_bypass = {1, 3, 5}
        # SWA layers (0, 2) compress. Full attn layers (1, 3) bypass.
        assert len(cdc._compressed_keys) >= 3
        assert cdc._compressed_keys[0] is not None  # SWA: compressed
        assert cdc._compressed_keys[1] is None  # full attn: bypassed
        assert cdc._compressed_keys[2] is not None  # SWA: compressed

    def test_full_attn_bypass_indices_from_config(self) -> None:
        """_full_attn_bypass set is populated from layer_types."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cdc = CompressedDynamicCache(
            cache, head_dim=DIM, bits=BITS, model_config=_GemmaLikeConfig()
        )
        # _GemmaLikeConfig: ["sliding", "full"] * 3 → full at 1, 3, 5
        assert cdc._full_attn_bypass == {1, 3, 5}

    def test_no_bypass_without_model_config(self) -> None:
        """Without model_config, no full attention bypass occurs."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
        assert cdc._full_attn_bypass == set()

    def test_full_attn_values_are_lossless(self, device: torch.device) -> None:
        """Bypassed full attention layers preserve exact values."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer

        cache = DynamicCache()
        CompressedDynamicCache(
            cache, head_dim=DIM, bits=BITS, model_config=_GemmaLikeConfig()
        )
        cache.layers.clear()
        cache.layers.extend([DynamicLayer(), DynamicLayer()])

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        # Layer 1 is full_attention → bypassed → lossless
        k_out, v_out = cache.update(key.clone(), val.clone(), layer_idx=1)
        assert torch.equal(k_out, key)
        assert torch.equal(v_out, val)

    def test_get_seq_length_bypassed_full_attn(self, device: torch.device) -> None:
        """Bypassed full attn layers report correct seq length, not 0."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer

        cache = DynamicCache()
        CompressedDynamicCache(
            cache, head_dim=DIM, bits=BITS, model_config=_GemmaLikeConfig()
        )
        cache.layers.clear()
        cache.layers.extend([DynamicLayer(), DynamicLayer()])

        key = torch.randn(1, 4, 1, DIM, device=device)
        val = torch.randn(1, 4, 1, DIM, device=device)

        # Layer 1 is full_attention → bypassed
        cache.update(key.clone(), val.clone(), layer_idx=1)
        assert cache.get_seq_length(1) == 1
