"""Tests for bit-width support and TQ4 nibble-packed storage."""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.kv_cache import CompressedDynamicCache
from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

from .conftest import BITS, BITS_4, DIM, cosine_similarity_flat


@pytest.mark.unit
class TestBitWidthSupport:
    """Validate CompressedDynamicCache at non-default bit widths."""

    @pytest.mark.parametrize(
        ("bits", "min_cosine"),
        [(2, 0.75), (5, 0.90)],
        ids=["2bit", "5bit"],
    )
    def test_compress_decompress_quality(self, bits: int, min_cosine: float) -> None:
        """Compression quality scales with bit width."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=bits)

        keys = torch.randn(1, 4, 32, DIM)
        values = torch.randn(1, 4, 32, DIM)
        out_k, out_v = cache.update(keys, values, layer_idx=0)

        assert out_k.shape == keys.shape
        cos = cosine_similarity_flat(keys, out_k)
        assert cos > min_cosine, f"{bits}-bit cosine {cos:.4f} below {min_cosine}"

    def test_5bit_better_quality_than_3bit(self) -> None:
        """5-bit (32 levels) should beat 3-bit (8 levels) in reconstruction."""
        from transformers import DynamicCache

        original = torch.randn(1, 4, 50, DIM)

        cache3 = DynamicCache()
        _ = CompressedDynamicCache(cache3, head_dim=DIM, bits=3)
        out3, _ = cache3.update(
            original.clone(), torch.randn(1, 4, 50, DIM), layer_idx=0
        )

        cache5 = DynamicCache()
        _ = CompressedDynamicCache(cache5, head_dim=DIM, bits=5)
        out5, _ = cache5.update(
            original.clone(), torch.randn(1, 4, 50, DIM), layer_idx=0
        )

        cos3 = cosine_similarity_flat(original, out3)
        cos5 = cosine_similarity_flat(original, out5)
        assert cos5 > cos3, f"5-bit ({cos5:.4f}) should beat 3-bit ({cos3:.4f})"


@pytest.mark.unit
class TestNibblePacking:
    """Validate TQ4 nibble-packed storage in CompressedDynamicCache."""

    def test_basic_update_4bit(self, device: torch.device) -> None:
        """4-bit compressed cache should accept and return correct shapes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        keys = torch.randn(1, 8, 1, DIM).to(device)
        values = torch.randn(1, 8, 1, DIM).to(device)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape == (1, 8, 1, DIM)
        assert out_v.shape == (1, 8, 1, DIM)

    def test_indices_are_nibble_packed(self) -> None:
        """At bits=4, indices should be half the head_dim (packed pairs)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=0
        )

        # Packed: head_dim // 2 = 64 instead of 128
        ck = cc._compressed_keys[0]
        assert ck is not None
        assert ck.indices.shape == (1, 4, 10, DIM // 2)
        assert ck.indices.dtype == torch.uint8
        assert ck.packed is True

    def test_nibble_pack_unpack_roundtrip(self, device: torch.device) -> None:
        """Pack then unpack should recover exact original indices."""
        indices = torch.randint(0, 16, (2, 4, 8, DIM), dtype=torch.uint8).to(device)

        packed = CompressedDynamicCache._nibble_pack(indices)
        assert packed.shape == (2, 4, 8, DIM // 2)

        unpacked = CompressedDynamicCache._nibble_unpack(packed)
        assert unpacked.shape == (2, 4, 8, DIM)
        torch.testing.assert_close(unpacked, indices.long())

    def test_compression_ratio_4bit(self, device: torch.device) -> None:
        """4-bit nibble-packed should achieve ~3.7x compression."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        for layer in range(4):
            cache.update(
                torch.randn(1, 8, 100, DIM).to(device),
                torch.randn(1, 8, 100, DIM).to(device),
                layer_idx=layer,
            )

        ratio = cc.baseline_vram_bytes() / cc.vram_bytes()
        # 68 bytes per block vs 256 → ~3.76x
        assert ratio > 3.5, f"Expected >3.5x compression, got {ratio:.2f}x"

    def test_4bit_better_quality_than_3bit(self, device: torch.device) -> None:
        """TQ4 should have higher cosine similarity than TQ3."""
        from transformers import DynamicCache

        original = torch.randn(1, 4, 50, DIM).to(device)

        # TQ3
        cache3 = DynamicCache()
        _ = CompressedDynamicCache(cache3, head_dim=DIM, bits=BITS)
        out3, _ = cache3.update(
            original.clone(), torch.randn(1, 4, 50, DIM).to(device), layer_idx=0
        )
        cos3 = cosine_similarity_flat(original, out3)

        # TQ4
        cache4 = DynamicCache()
        _ = CompressedDynamicCache(cache4, head_dim=DIM, bits=BITS_4)
        out4, _ = cache4.update(
            original.clone(), torch.randn(1, 4, 50, DIM).to(device), layer_idx=0
        )
        cos4 = cosine_similarity_flat(original, out4)

        assert cos4 > cos3, f"TQ4 ({cos4:.4f}) should beat TQ3 ({cos3:.4f})"

    def test_multi_layer_generation_4bit(self) -> None:
        """Nibble-packed cache should handle multi-layer prefill + gen."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        # Prefill 4 layers
        for layer in range(4):
            cache.update(
                torch.randn(1, 4, 64, DIM),
                torch.randn(1, 4, 64, DIM),
                layer_idx=layer,
            )

        # Generate 5 tokens
        for _ in range(5):
            for layer in range(4):
                out_k, _ = cache.update(
                    torch.randn(1, 4, 1, DIM),
                    torch.randn(1, 4, 1, DIM),
                    layer_idx=layer,
                )

        assert out_k.shape == (1, 4, 69, DIM)  # 64 prefill + 5 gen
        assert cache.get_seq_length(0) == 69

    def test_compression_stats_4bit(self) -> None:
        """Stats should report nibble_packed=True and correct head_dim."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS_4)

        cache.update(
            torch.randn(1, 4, 50, DIM), torch.randn(1, 4, 50, DIM), layer_idx=0
        )

        stats = cc.compression_stats()
        assert stats["bits"] == 4
        assert stats["nibble_packed"] is True
        assert stats["head_dim"] == DIM
        assert stats["compression_ratio"] > 3.5


# ---------------------------------------------------------------------------
# Multi head-dim support (Story 5.2)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[64, 96, 128], ids=["dim64", "dim96", "dim128"], scope="module")
def tq4_quantizer_multi(request: pytest.FixtureRequest) -> TurboQuantMSE:
    """Quantizer at various head_dims for kernel validation."""
    return TurboQuantMSE(request.param, 4, seed=42)


def _compress(x: torch.Tensor, q: TurboQuantMSE) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress using a specific quantizer (not hardwired to conftest dim=128)."""
    rot_t = q.rotation.T.contiguous()
    return tq4_compress(
        x,
        rot_t[:, 0::2].contiguous(),
        rot_t[:, 1::2].contiguous(),
        q.codebook.boundaries.clone(),
    )


def _decompress(
    packed: torch.Tensor,
    norms: torch.Tensor,
    q: TurboQuantMSE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decompress using a specific quantizer, applying inverse rotation."""
    rotated = tq4_decompress(packed, norms, q.codebook.centroids.clone(), dtype=dtype)
    return (rotated.float() @ q.rotation).to(dtype)


@pytest.mark.unit
class TestMultiDimSupport:
    """Validate TQ4 compress/decompress and nibble packing at head_dim 64, 96, 128."""

    def test_compress_decompress_roundtrip(
        self, tq4_quantizer_multi: TurboQuantMSE
    ) -> None:
        """Compress-decompress roundtrip quality at various head_dims."""
        dim = tq4_quantizer_multi.rotation.shape[0]
        x = torch.randn(2, 4, dim)
        packed, norms = _compress(x, tq4_quantizer_multi)
        assert packed.shape == (2, 4, dim // 2)
        assert packed.dtype == torch.uint8
        x_hat = _decompress(packed, norms, tq4_quantizer_multi)
        assert x_hat.shape == x.shape
        cos = cosine_similarity_flat(x, x_hat)
        assert cos > 0.80, f"Roundtrip cosine {cos:.4f} < 0.80 at head_dim={dim}"

    def test_nibble_pack_unpack_bit_exact(
        self, tq4_quantizer_multi: TurboQuantMSE
    ) -> None:
        """Pack then unpack must be bit-exact at HALF_D=32, 48, and 64."""
        dim = tq4_quantizer_multi.rotation.shape[0]
        indices = torch.randint(0, 16, (2, 4, 8, dim), dtype=torch.uint8)
        packed = CompressedDynamicCache._nibble_pack(indices)
        assert packed.shape == (2, 4, 8, dim // 2)
        unpacked = CompressedDynamicCache._nibble_unpack(packed)
        assert unpacked.shape == (2, 4, 8, dim)
        torch.testing.assert_close(unpacked, indices.long())

    def test_codebook_cache_entries(self, tq4_quantizer_multi: TurboQuantMSE) -> None:
        """@lru_cache generates and caches codebook for each new dim."""
        from turboquant_vllm.lloyd_max import _solve_lloyd_max_cached, solve_lloyd_max

        dim = tq4_quantizer_multi.rotation.shape[0]
        # Should hit cache — quantizer __init__ already called solve_lloyd_max
        centroids, boundaries = solve_lloyd_max(dim, 4)
        assert centroids.shape == (16,)
        assert boundaries.shape == (15,)
        info = _solve_lloyd_max_cached.cache_info()
        assert info.hits > 0, "Codebook should be served from cache"
        assert info.currsize <= 32, f"Cache has {info.currsize} entries, max is 32"
