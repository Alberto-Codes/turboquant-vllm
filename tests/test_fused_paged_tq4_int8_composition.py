"""36-layer composition tests for fused paged TQ4 INT8 prefill kernel.

Extracted from test_fused_paged_tq4_int8.py (test maturity split — file
exceeded 500-line gate).

All GPU tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

from tests.conftest import cosine_similarity_flat  # noqa: E402
from tests.helpers.int8_prefill import reference_causal_prefill  # noqa: E402
from tests.helpers.paged_cache import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_DIM,
    SEED,
    build_paged_cache,
)
from turboquant_vllm.quantizer import TurboQuantMSE  # noqa: E402
from turboquant_vllm.triton.fused_paged_tq4_int8_prefill import (  # noqa: E402
    fused_paged_tq4_int8_prefill,
)

pytestmark = [pytest.mark.gpu]

# AC 6 specifies 0.997 per-layer in end-to-end pipeline; random-data flat
# cosine is stricter (no error averaging).  TQ4 alone drops ~0.005, INT8
# adds ~0.003 → combined floor ~0.990.
INT8_TQ4_VS_BASELINE_THRESHOLD = 0.990


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


_cosine_sim = cosine_similarity_flat
_build_paged_cache = build_paged_cache


@pytest.mark.gpu
@pytest.mark.slow
class TestInt8Prefill36LayerComposition:
    """36-layer INT8 prefill composition for release gate (5.9)."""

    def test_36_layer_per_layer_cosine(
        self, device: str, tq4_quantizer: TurboQuantMSE
    ) -> None:
        """Per-layer cosine >0.990 across 36 layers with 1024-token prefill."""
        H_Q, H_KV = 28, 4
        num_layers = 36
        seq_len = 1024
        centroids = tq4_quantizer.codebook.centroids.to(device)
        rotation = tq4_quantizer.rotation.to(device)

        for layer_idx in range(num_layers):
            torch.manual_seed(SEED + layer_idx)
            k = torch.randn(seq_len, H_KV, HEAD_DIM, device=device, dtype=torch.float16)
            v = torch.randn(seq_len, H_KV, HEAD_DIM, device=device, dtype=torch.float16)
            q = torch.randn(seq_len, H_Q, HEAD_DIM, device=device, dtype=torch.float16)

            kv_cache, block_table, seq_lens_t = _build_paged_cache(
                k, v, [seq_len], tq4_quantizer, device
            )

            actual = fused_paged_tq4_int8_prefill(
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
            expected = reference_causal_prefill(
                q,
                kv_cache,
                block_table,
                seq_len,
                H_KV,
                centroids,
                rotation,
            )

            cos = _cosine_sim(actual, expected)
            assert cos > INT8_TQ4_VS_BASELINE_THRESHOLD, (
                f"Layer {layer_idx}: cosine {cos:.6f} < {INT8_TQ4_VS_BASELINE_THRESHOLD}"
            )
