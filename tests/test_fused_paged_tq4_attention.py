"""Tests for fused paged TQ4 decode attention kernel.

Validates that the fused paged kernel (reading compressed blocks directly
from the page table) matches the decompress-all reference path at two
precision tiers:

- Cache parity (>0.999): fused paged vs decompress-all + flash_attn
- Kernel correctness (>0.998): fused paged vs contiguous reference

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import cosine_similarity_flat
from tests.helpers.paged_cache import (
    BLOCK_SIZE,
    HEAD_DIM,
    SEED,
    build_paged_cache,
    reference_decompress_and_attend,
)
from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.flash_attention_tq4_kv import (
    triton_flash_attention_tq4_kv,
)
from turboquant_vllm.triton.fused_paged_tq4_attention import fused_paged_tq4_decode
from turboquant_vllm.triton.tq4_compress import tq4_compress

pytestmark = [pytest.mark.gpu]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HALF_D = HEAD_DIM // 2
TQ4_BITS = 4
CACHE_PARITY_THRESHOLD = 0.999
KERNEL_CORRECTNESS_THRESHOLD = 0.998


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> str:
    """CUDA-only device fixture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton kernels")
    return "cuda"


@pytest.fixture(scope="module")
def tq4_quantizer() -> TurboQuantMSE:
    """Module-scoped TQ4 quantizer (dim=128, bits=4)."""
    return TurboQuantMSE(HEAD_DIM, TQ4_BITS, seed=SEED)


# Aliases for brevity (shared helpers)
_build_paged_cache = build_paged_cache
_reference_decompress_and_attend = reference_decompress_and_attend
_cosine_sim = cosine_similarity_flat


# ---------------------------------------------------------------------------
# GQA parametrization
# ---------------------------------------------------------------------------

GQA_CONFIGS = [
    pytest.param(28, 4, id="molmo2_28q_4kv"),
    pytest.param(32, 8, id="llama_32q_8kv"),
    pytest.param(4, 4, id="mha_4q_4kv"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFusedPagedTQ4Decode:
    """Fused paged TQ4 decode kernel correctness tests."""

    @pytest.mark.parametrize(("H_Q", "H_KV"), GQA_CONFIGS)
    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16],
        ids=["fp16", "bf16"],
    )
    def test_cache_parity(
        self,
        device: str,
        tq4_quantizer: TurboQuantMSE,
        H_Q: int,
        H_KV: int,
        dtype: torch.dtype,
    ) -> None:
        """Fused paged output matches decompress-all reference (>0.999)."""
        seq_lens = [64]
        total_tokens = sum(seq_lens)

        k = torch.randn(
            total_tokens, H_KV, HEAD_DIM, device=device, dtype=torch.float16
        )
        v = torch.randn(
            total_tokens, H_KV, HEAD_DIM, device=device, dtype=torch.float16
        )
        q = torch.randn(len(seq_lens), H_Q, HEAD_DIM, device=device, dtype=dtype)

        centroids = tq4_quantizer.codebook.centroids.to(device)
        rotation = tq4_quantizer.rotation.to(device)

        kv_cache, block_table, seq_lens_t = _build_paged_cache(
            k, v, seq_lens, tq4_quantizer, device
        )

        actual = fused_paged_tq4_decode(
            q,
            kv_cache,
            block_table,
            seq_lens_t,
            centroids,
            rotation,
            num_kv_heads=H_KV,
            head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE,
        )

        expected = _reference_decompress_and_attend(
            q,
            kv_cache,
            block_table,
            seq_lens,
            H_KV,
            centroids,
            rotation,
        )

        cos = _cosine_sim(actual, expected)
        assert cos > CACHE_PARITY_THRESHOLD, (
            f"Cache parity cosine {cos:.6f} < {CACHE_PARITY_THRESHOLD}"
        )

    @pytest.mark.parametrize(("H_Q", "H_KV"), GQA_CONFIGS)
    def test_kernel_correctness(
        self,
        device: str,
        tq4_quantizer: TurboQuantMSE,
        H_Q: int,
        H_KV: int,
    ) -> None:
        """Fused paged output matches contiguous reference kernel (>0.998)."""
        seq_len = 64
        seq_lens = [seq_len]

        k = torch.randn(seq_len, H_KV, HEAD_DIM, device=device, dtype=torch.float16)
        v = torch.randn(seq_len, H_KV, HEAD_DIM, device=device, dtype=torch.float16)
        q = torch.randn(1, H_Q, HEAD_DIM, device=device, dtype=torch.float16)

        centroids = tq4_quantizer.codebook.centroids.to(device)
        rotation = tq4_quantizer.rotation.to(device)

        kv_cache, block_table, seq_lens_t = _build_paged_cache(
            k, v, seq_lens, tq4_quantizer, device
        )

        # Fused paged path
        actual = fused_paged_tq4_decode(
            q,
            kv_cache,
            block_table,
            seq_lens_t,
            centroids,
            rotation,
            num_kv_heads=H_KV,
            head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE,
        )

        # Contiguous reference: compress -> separate tensors -> flash_attention_tq4_kv
        rot_t = rotation.T.contiguous()
        rot_t_even = rot_t[:, 0::2].contiguous()
        rot_t_odd = rot_t[:, 1::2].contiguous()
        boundaries = tq4_quantizer.codebook.boundaries.to(device)

        k_packed, k_norms = tq4_compress(
            k.unsqueeze(0).expand(1, -1, -1, -1).reshape(-1, H_KV, HEAD_DIM),
            rot_t_even,
            rot_t_odd,
            boundaries,
        )
        v_packed, v_norms = tq4_compress(
            v.unsqueeze(0).expand(1, -1, -1, -1).reshape(-1, H_KV, HEAD_DIM),
            rot_t_even,
            rot_t_odd,
            boundaries,
        )

        # Reshape for contiguous kernel: (B=1, H_KV, seq_kv, HALF_D)
        k_packed_4d = k_packed.permute(1, 0, 2).unsqueeze(0)
        k_norms_4d = k_norms.squeeze(-1).permute(1, 0).unsqueeze(0)
        v_packed_4d = v_packed.permute(1, 0, 2).unsqueeze(0)
        v_norms_4d = v_norms.squeeze(-1).permute(1, 0).unsqueeze(0)

        q_4d = q.unsqueeze(2)  # (1, H_Q, 1, D)

        expected = triton_flash_attention_tq4_kv(
            q_4d,
            k_packed_4d,
            k_norms_4d,
            v_packed_4d,
            v_norms_4d,
            centroids,
            rotation,
        )
        expected = expected.squeeze(2)  # (1, H_Q, 1, D) -> (1, H_Q, D)

        cos = _cosine_sim(actual, expected)
        assert cos > KERNEL_CORRECTNESS_THRESHOLD, (
            f"Kernel correctness cosine {cos:.6f} < {KERNEL_CORRECTNESS_THRESHOLD}"
        )

    @pytest.mark.parametrize(
        "seq_len",
        [
            pytest.param(1, id="single_token"),
            pytest.param(BLOCK_SIZE, id="exact_page_boundary"),
            pytest.param(BLOCK_SIZE + 1, id="cross_page"),
            pytest.param(BLOCK_SIZE * 3 - 1, id="partial_last_tile"),
        ],
    )
    def test_sequence_length_edge_cases(
        self,
        device: str,
        tq4_quantizer: TurboQuantMSE,
        seq_len: int,
    ) -> None:
        """Edge case sequence lengths produce correct attention output."""
        H_Q, H_KV = 28, 4
        seq_lens = [seq_len]

        k = torch.randn(seq_len, H_KV, HEAD_DIM, device=device, dtype=torch.float16)
        v = torch.randn(seq_len, H_KV, HEAD_DIM, device=device, dtype=torch.float16)
        q = torch.randn(1, H_Q, HEAD_DIM, device=device, dtype=torch.float16)

        centroids = tq4_quantizer.codebook.centroids.to(device)
        rotation = tq4_quantizer.rotation.to(device)

        kv_cache, block_table, seq_lens_t = _build_paged_cache(
            k, v, seq_lens, tq4_quantizer, device
        )

        actual = fused_paged_tq4_decode(
            q,
            kv_cache,
            block_table,
            seq_lens_t,
            centroids,
            rotation,
            num_kv_heads=H_KV,
            head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE,
        )

        expected = _reference_decompress_and_attend(
            q,
            kv_cache,
            block_table,
            seq_lens,
            H_KV,
            centroids,
            rotation,
        )

        cos = _cosine_sim(actual, expected)
        assert cos > CACHE_PARITY_THRESHOLD, (
            f"Edge case seq_len={seq_len} cosine {cos:.6f} < {CACHE_PARITY_THRESHOLD}"
        )

    def test_masking_invalid_positions(
        self,
        device: str,
        tq4_quantizer: TurboQuantMSE,
    ) -> None:
        """Invalid positions (beyond seq_len) produce zero contribution."""
        H_Q, H_KV = 28, 4
        # Short sequence in a cache that has room for more
        seq_lens = [5]

        k = torch.randn(5, H_KV, HEAD_DIM, device=device, dtype=torch.float16)
        v = torch.randn(5, H_KV, HEAD_DIM, device=device, dtype=torch.float16)
        q = torch.randn(1, H_Q, HEAD_DIM, device=device, dtype=torch.float16)

        centroids = tq4_quantizer.codebook.centroids.to(device)
        rotation = tq4_quantizer.rotation.to(device)

        kv_cache, block_table, seq_lens_t = _build_paged_cache(
            k, v, seq_lens, tq4_quantizer, device
        )

        actual = fused_paged_tq4_decode(
            q,
            kv_cache,
            block_table,
            seq_lens_t,
            centroids,
            rotation,
            num_kv_heads=H_KV,
            head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE,
        )

        expected = _reference_decompress_and_attend(
            q,
            kv_cache,
            block_table,
            seq_lens,
            H_KV,
            centroids,
            rotation,
        )

        cos = _cosine_sim(actual, expected)
        assert cos > CACHE_PARITY_THRESHOLD, (
            f"Masking test cosine {cos:.6f} < {CACHE_PARITY_THRESHOLD}"
        )

    def test_multi_sequence_batch(
        self,
        device: str,
        tq4_quantizer: TurboQuantMSE,
    ) -> None:
        """Multiple sequences with different lengths in a single batch."""
        H_Q, H_KV = 28, 4
        seq_lens = [32, 17, 48]
        total_tokens = sum(seq_lens)

        k = torch.randn(
            total_tokens, H_KV, HEAD_DIM, device=device, dtype=torch.float16
        )
        v = torch.randn(
            total_tokens, H_KV, HEAD_DIM, device=device, dtype=torch.float16
        )
        q = torch.randn(
            len(seq_lens), H_Q, HEAD_DIM, device=device, dtype=torch.float16
        )

        centroids = tq4_quantizer.codebook.centroids.to(device)
        rotation = tq4_quantizer.rotation.to(device)

        kv_cache, block_table, seq_lens_t = _build_paged_cache(
            k, v, seq_lens, tq4_quantizer, device
        )

        actual = fused_paged_tq4_decode(
            q,
            kv_cache,
            block_table,
            seq_lens_t,
            centroids,
            rotation,
            num_kv_heads=H_KV,
            head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE,
        )

        expected = _reference_decompress_and_attend(
            q,
            kv_cache,
            block_table,
            seq_lens,
            H_KV,
            centroids,
            rotation,
        )

        cos = _cosine_sim(actual, expected)
        assert cos > CACHE_PARITY_THRESHOLD, (
            f"Multi-seq cosine {cos:.6f} < {CACHE_PARITY_THRESHOLD}"
        )
