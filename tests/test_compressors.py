"""Tests for production compressor wrappers and attention score estimation."""

import pytest
import torch

from turboquant_vllm.compressors import (
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
)

from .conftest import DIM


@pytest.mark.unit
class TestKeyCompressor:
    """Validate TurboQuantCompressorV2 (key cache compressor)."""

    def test_round_trip_shape(
        self, key_compressor: TurboQuantCompressorV2, device: torch.device
    ) -> None:
        """Compress/decompress should preserve tensor shape."""
        keys = torch.randn(1, 8, 32, DIM).to(device)
        compressed = key_compressor.compress(keys)
        decompressed = key_compressor.decompress(compressed)
        assert decompressed.shape == keys.shape

    def test_cosine_similarity(
        self, key_compressor: TurboQuantCompressorV2, device: torch.device
    ) -> None:
        """Decompressed keys should have high cosine similarity to originals."""
        keys = torch.randn(1, 4, 64, DIM).to(device)
        compressed = key_compressor.compress(keys)
        decompressed = key_compressor.decompress(compressed)

        flat_orig = keys.reshape(-1, DIM)
        flat_recon = decompressed.reshape(-1, DIM)
        cos_sim = torch.nn.functional.cosine_similarity(flat_orig, flat_recon, dim=-1)

        mean_sim = cos_sim.mean().item()
        # TurboQuantProd uses (bits-1)=2 bits for MSE + 1 bit QJL,
        # so reconstruction cosine sim is ~0.87 at 3-bit for non-unit vectors.
        # QJL compensates via inner product estimation, not reconstruction.
        assert mean_sim > 0.80, f"Mean cosine similarity {mean_sim:.4f} too low"

    def test_single_element_sequence(
        self, key_compressor: TurboQuantCompressorV2, device: torch.device
    ) -> None:
        """seq_len=1 should work without errors."""
        keys = torch.randn(1, 4, 1, DIM).to(device)
        compressed = key_compressor.compress(keys)
        decompressed = key_compressor.decompress(compressed)
        assert decompressed.shape == (1, 4, 1, DIM)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype_preserved(
        self,
        key_compressor: TurboQuantCompressorV2,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Output dtype should match input dtype."""
        keys = torch.randn(1, 2, 8, DIM).to(dtype).to(device)
        compressed = key_compressor.compress(keys)
        decompressed = key_compressor.decompress(compressed)
        assert decompressed.dtype == dtype


@pytest.mark.unit
class TestValueCompressor:
    """Validate TurboQuantCompressorMSE (value cache compressor)."""

    def test_round_trip_shape(
        self, value_compressor: TurboQuantCompressorMSE, device: torch.device
    ) -> None:
        """Compress/decompress should preserve tensor shape."""
        values = torch.randn(1, 8, 32, DIM).to(device)
        compressed = value_compressor.compress(values)
        decompressed = value_compressor.decompress(compressed)
        assert decompressed.shape == values.shape

    def test_cosine_similarity_higher_than_keys(
        self, value_compressor: TurboQuantCompressorMSE, device: torch.device
    ) -> None:
        """Value compressor (full 3-bit MSE) should have higher quality than keys."""
        values = torch.randn(1, 4, 64, DIM).to(device)
        compressed = value_compressor.compress(values)
        decompressed = value_compressor.decompress(compressed)

        flat_orig = values.reshape(-1, DIM)
        flat_recon = decompressed.reshape(-1, DIM)
        cos_sim = torch.nn.functional.cosine_similarity(flat_orig, flat_recon, dim=-1)

        mean_sim = cos_sim.mean().item()
        # MSE-only with full 3 bits should beat key compressor (2-bit MSE).
        # Empirical: ~0.92 for non-unit random vectors at 3-bit MSE.
        assert mean_sim > 0.90, f"Mean cosine similarity {mean_sim:.4f} too low"


@pytest.mark.unit
class TestAsymmetricAttention:
    """Validate attention score estimation from compressed keys."""

    def test_scores_shape(
        self, key_compressor: TurboQuantCompressorV2, device: torch.device
    ) -> None:
        """Asymmetric attention scores should have correct output shape."""
        batch, heads, q_len, kv_len = 1, 4, 1, 32
        queries = torch.randn(batch, heads, q_len, DIM).to(device)
        keys = torch.randn(batch, heads, kv_len, DIM).to(device)

        compressed = key_compressor.compress(keys)
        scores = key_compressor.asymmetric_attention_scores(queries, compressed)

        assert scores.shape == (batch, heads, q_len, kv_len)

    def test_scores_correlate_with_true(
        self, key_compressor: TurboQuantCompressorV2, device: torch.device
    ) -> None:
        """Estimated attention scores should correlate highly with true scores.

        This is the critical production API for computing attention directly
        from compressed keys.
        """
        batch, heads, q_len, kv_len = 1, 4, 1, 32
        queries = torch.randn(batch, heads, q_len, DIM).to(device)
        keys = torch.randn(batch, heads, kv_len, DIM).to(device)

        true_scores = torch.matmul(queries.float(), keys.float().transpose(-2, -1))

        compressed = key_compressor.compress(keys)
        estimated_scores = key_compressor.asymmetric_attention_scores(
            queries, compressed
        )

        for h in range(heads):
            true_flat = true_scores[0, h].flatten()
            est_flat = estimated_scores[0, h].flatten().float()
            correlation = torch.corrcoef(torch.stack([true_flat, est_flat]))[
                0, 1
            ].item()
            assert correlation > 0.85, (
                f"Head {h} attention score correlation {correlation:.4f} too low"
            )

    def test_multi_query_tokens(
        self, key_compressor: TurboQuantCompressorV2, device: torch.device
    ) -> None:
        """Asymmetric scores should work with multiple query tokens."""
        batch, heads, q_len, kv_len = 1, 2, 4, 16
        queries = torch.randn(batch, heads, q_len, DIM).to(device)
        keys = torch.randn(batch, heads, kv_len, DIM).to(device)

        compressed = key_compressor.compress(keys)
        scores = key_compressor.asymmetric_attention_scores(queries, compressed)

        assert scores.shape == (batch, heads, q_len, kv_len)


@pytest.mark.unit
class TestCompressionRatio:
    """Validate compression ratio claims."""

    def test_theoretical_compression_ratio(
        self, key_compressor: TurboQuantCompressorV2
    ) -> None:
        """Compressed info content should achieve at least 3x compression vs FP16.

        At 3-bit with QJL: ~4 bits/value effective vs 16 bits FP16 = 4x theoretical.
        """
        seq_len = 256
        keys = torch.randn(1, 8, seq_len, DIM, dtype=torch.float16)
        fp16_bytes = keys.nelement() * 2

        bits_per_value = key_compressor.bits + 1  # MSE bits + 1 QJL bit
        norm_overhead = 32  # one float32 norm per vector
        info_bits_per_vector = DIM * bits_per_value + norm_overhead
        total_info_bits = info_bits_per_vector * seq_len * 8  # 8 heads
        theoretical_compressed_bytes = total_info_bits / 8

        ratio = fp16_bytes / theoretical_compressed_bytes
        assert ratio > 3.0, f"Theoretical compression ratio {ratio:.1f}x too low"
