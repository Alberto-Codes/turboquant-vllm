"""Tests for per-layer cosine similarity and multi-layer composition.

Formalizes experimentally validated results from Experiments 002-005:
- Per-layer cosine similarity thresholds at different bit widths
- Multi-layer composition: quality should not degrade catastrophically
  across 36 layers (the precision compounding issue)
- Per-layer quality ordering: higher bits = better reconstruction
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from turboquant_vllm.kv_cache import CompressedDynamicCache

from .conftest import DIM

NUM_LAYERS = 36
NUM_HEADS = 8
SEQ_LEN = 128


def _per_layer_cosine(
    bits: int,
    num_layers: int = NUM_LAYERS,
    seq_len: int = SEQ_LEN,
) -> list[float]:
    """Compress random KV through multiple layers, return per-layer cosine.

    Args:
        bits: Quantization bit width.
        num_layers: Number of transformer layers to simulate.
        seq_len: Sequence length per layer.

    Returns:
        List of cosine similarities (one per layer) between original
        and decompressed keys.
    """
    from transformers import DynamicCache

    torch.manual_seed(42)
    cache = DynamicCache()
    _ = CompressedDynamicCache(cache, head_dim=DIM, bits=bits)

    originals: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
        keys = torch.randn(1, NUM_HEADS, seq_len, DIM)
        values = torch.randn(1, NUM_HEADS, seq_len, DIM)
        originals.append(keys.clone())
        cache.update(keys, values, layer_idx=layer_idx)

    cosines = []
    for layer_idx in range(num_layers):
        decompressed = cache.layers[layer_idx].keys
        assert decompressed is not None  # populated by cache.update above
        original = originals[layer_idx]
        cos = F.cosine_similarity(original.flatten(), decompressed.flatten(), dim=0)
        cosines.append(cos.item())

    return cosines


@pytest.mark.unit
class TestPerLayerCosine:
    """Validate per-layer cosine similarity at different bit widths."""

    @pytest.mark.parametrize(
        ("bits", "min_cosine"),
        [
            (2, 0.70),
            (3, 0.90),
            (4, 0.95),
            (5, 0.90),
        ],
        ids=["2bit", "3bit", "4bit", "5bit"],
    )
    def test_per_layer_cosine_threshold(self, bits: int, min_cosine: float) -> None:
        """Every layer should meet its bit-width cosine threshold."""
        cosines = _per_layer_cosine(bits)

        for layer_idx, cos in enumerate(cosines):
            assert cos > min_cosine, (
                f"Layer {layer_idx}: {bits}-bit cosine {cos:.4f} "
                f"below threshold {min_cosine}"
            )

    def test_quality_ordering_mean(self) -> None:
        """Higher bit widths should produce higher mean cosine across layers."""
        cos_3 = _per_layer_cosine(3)
        cos_4 = _per_layer_cosine(4)

        mean_3 = sum(cos_3) / len(cos_3)
        mean_4 = sum(cos_4) / len(cos_4)
        assert mean_4 > mean_3, (
            f"TQ4 mean ({mean_4:.4f}) should beat TQ3 mean ({mean_3:.4f})"
        )

    def test_cosine_stable_across_layers(self) -> None:
        """Cosine similarity should not degrade significantly across layers.

        Each layer compresses independently, so per-layer cosine should
        be roughly constant (no compounding). A spread > 0.05 between
        min and max would indicate a systematic issue.
        """
        cosines = _per_layer_cosine(4)
        spread = max(cosines) - min(cosines)
        assert spread < 0.05, (
            f"Per-layer cosine spread {spread:.4f} too large "
            f"(min={min(cosines):.4f}, max={max(cosines):.4f})"
        )


@pytest.mark.unit
class TestMultiLayerComposition:
    """Validate multi-layer composition does not degrade quality.

    Simulates the autoregressive decode pattern where each token passes
    through all layers sequentially. Tests that the cumulative effect of
    compression across layers does not cause catastrophic quality loss.
    """

    def _simulate_autoregressive_decode(
        self,
        bits: int,
        num_layers: int = NUM_LAYERS,
        prefill_len: int = 256,
        gen_steps: int = 64,
    ) -> tuple[list[float], list[float]]:
        """Simulate prefill + decode, return per-layer cosine for both phases.

        Args:
            bits: Quantization bit width.
            num_layers: Number of transformer layers.
            prefill_len: Tokens in prefill phase.
            gen_steps: Number of decode steps.

        Returns:
            Tuple of (prefill_cosines, decode_cosines) per layer.
        """
        from transformers import DynamicCache

        torch.manual_seed(42)
        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=bits)

        # Prefill
        prefill_originals: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            keys = torch.randn(1, NUM_HEADS, prefill_len, DIM)
            values = torch.randn(1, NUM_HEADS, prefill_len, DIM)
            prefill_originals.append(keys.clone())
            cache.update(keys, values, layer_idx=layer_idx)

        prefill_cosines = []
        for layer_idx in range(num_layers):
            layer_keys = cache.layers[layer_idx].keys
            assert layer_keys is not None  # populated by cache.update above
            decompressed = layer_keys[:, :, :prefill_len, :]
            cos = F.cosine_similarity(
                prefill_originals[layer_idx].flatten(),
                decompressed.flatten(),
                dim=0,
            )
            prefill_cosines.append(cos.item())

        # Decode — accumulate tokens one at a time
        decode_originals: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        for _ in range(gen_steps):
            for layer_idx in range(num_layers):
                keys = torch.randn(1, NUM_HEADS, 1, DIM)
                values = torch.randn(1, NUM_HEADS, 1, DIM)
                decode_originals[layer_idx].append(keys.clone())
                cache.update(keys, values, layer_idx=layer_idx)

        decode_cosines = []
        for layer_idx in range(num_layers):
            all_orig = torch.cat(decode_originals[layer_idx], dim=2)
            layer_keys = cache.layers[layer_idx].keys
            assert layer_keys is not None  # populated by decode loop above
            decompressed = layer_keys[:, :, prefill_len:, :]
            cos = F.cosine_similarity(all_orig.flatten(), decompressed.flatten(), dim=0)
            decode_cosines.append(cos.item())

        return prefill_cosines, decode_cosines

    @pytest.mark.parametrize(
        ("bits", "min_cosine"),
        [(3, 0.90), (4, 0.95)],
        ids=["3bit", "4bit"],
    )
    def test_prefill_quality(self, bits: int, min_cosine: float) -> None:
        """Prefill phase should maintain per-layer cosine threshold."""
        prefill_cosines, _ = self._simulate_autoregressive_decode(bits)

        for layer_idx, cos in enumerate(prefill_cosines):
            assert cos > min_cosine, (
                f"Prefill layer {layer_idx}: {bits}-bit cosine {cos:.4f} "
                f"below threshold {min_cosine}"
            )

    @pytest.mark.parametrize(
        ("bits", "min_cosine"),
        [(3, 0.90), (4, 0.95)],
        ids=["3bit", "4bit"],
    )
    def test_decode_quality(self, bits: int, min_cosine: float) -> None:
        """Decode phase should maintain per-layer cosine threshold.

        Incremental dequantization means only new tokens are decompressed
        each step. Quality should not degrade over generation steps.
        """
        _, decode_cosines = self._simulate_autoregressive_decode(bits)

        for layer_idx, cos in enumerate(decode_cosines):
            assert cos > min_cosine, (
                f"Decode layer {layer_idx}: {bits}-bit cosine {cos:.4f} "
                f"below threshold {min_cosine}"
            )

    def test_decode_quality_matches_prefill(self) -> None:
        """Decode-phase cosine should be comparable to prefill-phase.

        If incremental dequant introduces additional error, decode cosine
        would be systematically lower than prefill. Tolerance: 0.02.
        """
        prefill, decode = self._simulate_autoregressive_decode(4)

        for layer_idx in range(NUM_LAYERS):
            delta = prefill[layer_idx] - decode[layer_idx]
            assert abs(delta) < 0.02, (
                f"Layer {layer_idx}: prefill/decode gap {delta:.4f} "
                f"exceeds 0.02 tolerance"
            )

    def test_no_catastrophic_outlier(self) -> None:
        """No single layer should be a cosine outlier at TQ4.

        A catastrophic outlier would indicate a bug in rotation matrix
        generation or codebook sharing, not normal quantization noise.
        This checks that no layer is much worse than the others.
        """
        cosines = _per_layer_cosine(4, num_layers=NUM_LAYERS)

        worst = min(cosines)
        best = max(cosines)
        spread = best - worst
        assert spread < 0.02, (
            f"Catastrophic quality outlier: cosine spread {spread:.4f} "
            f"(best={best:.4f}, worst={worst:.4f}) exceeds 0.02"
        )
