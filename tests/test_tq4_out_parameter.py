"""Tests for the ``out`` parameter on tq4_compress and tq4_decompress.

Validates the PyTorch ``out`` convention: when a pre-allocated buffer is
provided, results are written into it and the *same tensor object* is
returned.  Tests run on CPU (pure PyTorch fallback) unconditionally —
no vLLM or CUDA required.
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

pytestmark = [pytest.mark.unit]

# Dimensions matching project defaults (DIM=128, BITS_4=4 from conftest)
N, H, D = 2, 4, 128
HALF_D = D // 2


@pytest.fixture(scope="module")
def compress_args(tq4_quantizer: TurboQuantMSE) -> dict[str, torch.Tensor]:
    """Rotation halves and boundaries for tq4_compress (CPU)."""
    rotation_t = tq4_quantizer.rotation.T.contiguous()
    return {
        "rotation_T_even": rotation_t[:, 0::2].contiguous(),
        "rotation_T_odd": rotation_t[:, 1::2].contiguous(),
        "boundaries": tq4_quantizer.codebook.boundaries.clone(),
    }


@pytest.fixture(scope="module")
def centroids(tq4_quantizer: TurboQuantMSE) -> torch.Tensor:
    """Centroid table for tq4_decompress (CPU)."""
    return tq4_quantizer.codebook.centroids.clone()


def _make_input(n: int = N, h: int = H, d: int = D) -> torch.Tensor:
    """Random fp32 input on CPU."""
    return torch.randn(n, h, d)


# ── tq4_compress ────────────────────────────────────────────────────


class TestTq4CompressOut:
    """Out-parameter contract for tq4_compress."""

    def test_backward_compat_no_out(self, compress_args: dict) -> None:
        """Calling without out behaves identically to before."""
        x = _make_input()
        packed, norms = tq4_compress(x, **compress_args)
        assert packed.shape == (N, H, HALF_D)
        assert norms.shape == (N, H, 1)
        assert packed.dtype == torch.uint8
        assert norms.dtype == torch.float32

    def test_out_identity(self, compress_args: dict) -> None:
        """Returned tensors are the same objects as the provided buffers."""
        x = _make_input()
        packed_buf = torch.empty(N, H, HALF_D, dtype=torch.uint8)
        norms_buf = torch.empty(N, H, 1, dtype=torch.float32)

        result = tq4_compress(x, **compress_args, out=(packed_buf, norms_buf))

        assert result[0] is packed_buf
        assert result[1] is norms_buf

    def test_out_numerical_equivalence(self, compress_args: dict) -> None:
        """Results with out are identical to results without out."""
        x = _make_input()
        ref_packed, ref_norms = tq4_compress(x, **compress_args)

        packed_buf = torch.empty(N, H, HALF_D, dtype=torch.uint8)
        norms_buf = torch.empty(N, H, 1, dtype=torch.float32)
        out_packed, out_norms = tq4_compress(
            x, **compress_args, out=(packed_buf, norms_buf)
        )

        assert torch.equal(out_packed, ref_packed)
        assert torch.equal(out_norms, ref_norms)

    def test_out_buffer_reuse(self, compress_args: dict) -> None:
        """Calling twice with the same buffers overwrites correctly."""
        packed_buf = torch.empty(N, H, HALF_D, dtype=torch.uint8)
        norms_buf = torch.empty(N, H, 1, dtype=torch.float32)
        out = (packed_buf, norms_buf)

        x1 = _make_input()
        tq4_compress(x1, **compress_args, out=out)
        snap1_packed = packed_buf.clone()

        x2 = _make_input()
        tq4_compress(x2, **compress_args, out=out)

        ref_packed, _ = tq4_compress(x2, **compress_args)
        assert torch.equal(packed_buf, ref_packed)
        assert not torch.equal(snap1_packed, packed_buf)

    def test_out_multi_token_batch(self, compress_args: dict) -> None:
        """Out parameter works with N>1 batch (validates reshape logic)."""
        n = 4
        x = _make_input(n=n)
        packed_buf = torch.empty(n, H, HALF_D, dtype=torch.uint8)
        norms_buf = torch.empty(n, H, 1, dtype=torch.float32)

        result = tq4_compress(x, **compress_args, out=(packed_buf, norms_buf))
        ref_packed, ref_norms = tq4_compress(x, **compress_args)

        assert result[0] is packed_buf
        assert torch.equal(result[0], ref_packed)
        assert torch.equal(result[1], ref_norms)


# ── tq4_decompress ──────────────────────────────────────────────────


class TestTq4DecompressOut:
    """Out-parameter contract for tq4_decompress."""

    def test_backward_compat_no_out(
        self,
        compress_args: dict,
        centroids: torch.Tensor,
    ) -> None:
        """Calling without out behaves identically to before."""
        x = _make_input()
        packed, norms = tq4_compress(x, **compress_args)
        result = tq4_decompress(packed, norms, centroids, dtype=torch.float32)
        assert result.shape == (N, H, D)
        assert result.dtype == torch.float32

    def test_out_identity(
        self,
        compress_args: dict,
        centroids: torch.Tensor,
    ) -> None:
        """Returned tensor is the same object as the provided buffer."""
        x = _make_input()
        packed, norms = tq4_compress(x, **compress_args)
        out_buf = torch.empty(N, H, D, dtype=torch.float32)

        result = tq4_decompress(
            packed,
            norms,
            centroids,
            dtype=torch.float32,
            out=out_buf,
        )

        assert result is out_buf

    def test_out_numerical_equivalence(
        self,
        compress_args: dict,
        centroids: torch.Tensor,
    ) -> None:
        """Results with out are identical to results without out."""
        x = _make_input()
        packed, norms = tq4_compress(x, **compress_args)

        ref = tq4_decompress(packed, norms, centroids, dtype=torch.float32)
        out_buf = torch.empty(N, H, D, dtype=torch.float32)
        out_result = tq4_decompress(
            packed,
            norms,
            centroids,
            dtype=torch.float32,
            out=out_buf,
        )

        assert torch.equal(out_result, ref)

    def test_out_buffer_reuse(
        self,
        compress_args: dict,
        centroids: torch.Tensor,
    ) -> None:
        """Calling twice with the same buffer overwrites correctly."""
        out_buf = torch.empty(N, H, D, dtype=torch.float32)

        x1 = _make_input()
        packed1, norms1 = tq4_compress(x1, **compress_args)
        tq4_decompress(packed1, norms1, centroids, dtype=torch.float32, out=out_buf)
        snap1 = out_buf.clone()

        x2 = _make_input()
        packed2, norms2 = tq4_compress(x2, **compress_args)
        tq4_decompress(packed2, norms2, centroids, dtype=torch.float32, out=out_buf)

        ref = tq4_decompress(packed2, norms2, centroids, dtype=torch.float32)
        assert torch.equal(out_buf, ref)
        assert not torch.equal(snap1, out_buf)

    def test_out_multi_token_batch(
        self,
        compress_args: dict,
        centroids: torch.Tensor,
    ) -> None:
        """Out parameter works with N>1 batch."""
        n = 4
        x = _make_input(n=n)
        packed, norms = tq4_compress(x, **compress_args)
        out_buf = torch.empty(n, H, D, dtype=torch.float32)

        result = tq4_decompress(
            packed,
            norms,
            centroids,
            dtype=torch.float32,
            out=out_buf,
        )
        ref = tq4_decompress(packed, norms, centroids, dtype=torch.float32)

        assert result is out_buf
        assert torch.equal(result, ref)
