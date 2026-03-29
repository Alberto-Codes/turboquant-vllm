"""36-layer composition tests for fused paged TQ4 kernels.

Extracted from ``test_fused_paged_tq4_attention.py`` (Test Maturity
MEDIUM 5) to keep test files under the 500-line module gate.

Validates per-layer cosine stability across 36 layers with 1024+32
tokens for the fused paged decode kernel.
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
from turboquant_vllm.triton.fused_paged_tq4_attention import fused_paged_tq4_decode

pytestmark = [pytest.mark.gpu]

CACHE_PARITY_THRESHOLD = 0.999


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
    return TurboQuantMSE(HEAD_DIM, 4, seed=SEED)


# ---------------------------------------------------------------------------
# 36-layer composition test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFusedPaged36LayerComposition:
    """36-layer composition test for release gate validation."""

    def test_36_layer_cache_parity(
        self,
        device: str,
        tq4_quantizer: TurboQuantMSE,
    ) -> None:
        """Per-layer cosine >0.999 across 36 layers with 1024+32 tokens."""
        H_Q, H_KV = 28, 4
        num_layers = 36
        prefill_len = 1024
        gen_tokens = 32
        total_kv_len = prefill_len + gen_tokens

        centroids = tq4_quantizer.codebook.centroids.to(device)
        rotation = tq4_quantizer.rotation.to(device)

        for layer_idx in range(num_layers):
            torch.manual_seed(SEED + layer_idx)

            k = torch.randn(
                total_kv_len, H_KV, HEAD_DIM, device=device, dtype=torch.float16
            )
            v = torch.randn(
                total_kv_len, H_KV, HEAD_DIM, device=device, dtype=torch.float16
            )
            q = torch.randn(1, H_Q, HEAD_DIM, device=device, dtype=torch.float16)

            seq_lens = [total_kv_len]
            kv_cache, block_table, seq_lens_t = build_paged_cache(
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

            expected = reference_decompress_and_attend(
                q,
                kv_cache,
                block_table,
                seq_lens,
                H_KV,
                centroids,
                rotation,
            )

            cos = cosine_similarity_flat(actual, expected)
            assert cos > CACHE_PARITY_THRESHOLD, (
                f"Layer {layer_idx}: cosine {cos:.6f} < {CACHE_PARITY_THRESHOLD}"
            )
