"""Multi head-dim Triton vs PyTorch equivalence tests (Story 5.2).

Validates that TQ4 compress/decompress kernels produce identical output
across Triton and PyTorch paths at head_dim 64, 96, and 128.  Does NOT
require vLLM — only core turboquant_vllm Triton kernels.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from turboquant_vllm.quantizer import TurboQuantMSE

pytestmark = [pytest.mark.gpu]


@pytest.fixture(params=[64, 96, 128], ids=["dim64", "dim96", "dim128"], scope="module")
def multi_dim_quantizer(request: pytest.FixtureRequest) -> TurboQuantMSE:
    """Quantizer at various head_dims for multi-dim Triton validation."""
    return TurboQuantMSE(request.param, 4, seed=42)


class TestMultiDimTritonEquivalence:
    """Triton vs PyTorch equivalence at head_dim 64, 96, and 128.

    Validates that compress and decompress produce identical output
    across both code paths at non-standard head dimensions.
    """

    NUM_KV_HEADS = 8

    @staticmethod
    def _make_tensors(q: TurboQuantMSE) -> dict[str, Any]:
        """Derive rotation, boundaries, centroids from a quantizer."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rotation = q.rotation.to(device).clone()
        rotation_t = rotation.T.contiguous()
        return {
            "device": device,
            "rotation": rotation,
            "rotation_t": rotation_t,
            "boundaries": q.codebook.boundaries.to(device).clone(),
            "centroids": q.codebook.centroids.to(device).clone(),
            "rot_T_even": rotation_t[:, 0::2].contiguous(),
            "rot_T_odd": rotation_t[:, 1::2].contiguous(),
        }

    @staticmethod
    def _pytorch_compress(
        x: torch.Tensor, rotation_t: torch.Tensor, boundaries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference PyTorch compress path."""
        N, H, D = x.shape
        flat = x.reshape(N * H, D).float()
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ rotation_t
        indices = torch.bucketize(rotated, boundaries).clamp(0, 15).to(torch.uint8)
        packed = (indices[:, 0::2] << 4) | indices[:, 1::2]
        return packed.reshape(N, H, D // 2), norms.reshape(N, H, 1)

    @staticmethod
    def _pytorch_decompress(
        packed: torch.Tensor,
        norms: torch.Tensor,
        centroids: torch.Tensor,
        rotation: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reference PyTorch decompress path."""
        N, H, half_D = packed.shape
        D = half_D * 2
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(N * H, D)
        flat_norms = norms.reshape(N * H, 1)
        reconstructed = centroids[indices]
        unrotated = reconstructed @ rotation
        return (unrotated * flat_norms).reshape(N, H, D).to(dtype)

    def test_compress_output_match(self, multi_dim_quantizer: TurboQuantMSE) -> None:
        """Triton compress matches PyTorch compress at each head_dim."""
        from turboquant_vllm.triton.tq4_compress import tq4_compress

        t = self._make_tensors(multi_dim_quantizer)
        dim = multi_dim_quantizer.rotation.shape[0]
        x = torch.randn(
            4, self.NUM_KV_HEADS, dim, device=t["device"], dtype=torch.float16
        )

        pt_packed, pt_norms = self._pytorch_compress(
            x, t["rotation_t"], t["boundaries"]
        )
        tr_packed, tr_norms = tq4_compress(
            x, t["rot_T_even"], t["rot_T_odd"], t["boundaries"]
        )

        assert torch.equal(pt_packed, tr_packed), (
            f"Packed bytes differ at head_dim={dim}"
        )
        torch.testing.assert_close(pt_norms, tr_norms, atol=1e-5, rtol=1e-5)

    def test_decompress_output_match(self, multi_dim_quantizer: TurboQuantMSE) -> None:
        """Triton decompress matches PyTorch decompress at each head_dim."""
        from turboquant_vllm.triton.tq4_decompress import tq4_decompress

        t = self._make_tensors(multi_dim_quantizer)
        dim = multi_dim_quantizer.rotation.shape[0]
        packed = torch.randint(
            0,
            255,
            (16, self.NUM_KV_HEADS, dim // 2),
            device=t["device"],
            dtype=torch.uint8,
        )
        norms = (
            torch.randn(
                16, self.NUM_KV_HEADS, 1, device=t["device"], dtype=torch.float32
            )
            .abs_()
            .clamp_(min=0.1)
        )

        pt_out = self._pytorch_decompress(
            packed, norms, t["centroids"], t["rotation"], torch.float32
        )
        tr_out_rot = tq4_decompress(packed, norms, t["centroids"], torch.float32)
        tr_out = (tr_out_rot.float() @ t["rotation"]).to(torch.float32)

        cos = torch.nn.functional.cosine_similarity(
            pt_out.flatten().unsqueeze(0), tr_out.flatten().unsqueeze(0)
        ).item()
        assert cos > 0.999, f"Decompress cosine {cos:.6f} < 0.999 at head_dim={dim}"

    def test_full_round_trip(self, multi_dim_quantizer: TurboQuantMSE) -> None:
        """Full compress→decompress round-trip: Triton vs PyTorch at each head_dim."""
        from turboquant_vllm.triton.tq4_compress import tq4_compress
        from turboquant_vllm.triton.tq4_decompress import tq4_decompress

        t = self._make_tensors(multi_dim_quantizer)
        dim = multi_dim_quantizer.rotation.shape[0]
        x = torch.randn(
            4, self.NUM_KV_HEADS, dim, device=t["device"], dtype=torch.float16
        )

        # PyTorch round-trip
        pt_packed, pt_norms = self._pytorch_compress(
            x, t["rotation_t"], t["boundaries"]
        )
        pt_recon = self._pytorch_decompress(
            pt_packed, pt_norms, t["centroids"], t["rotation"], torch.float32
        )

        # Triton round-trip
        tr_packed, tr_norms = tq4_compress(
            x, t["rot_T_even"], t["rot_T_odd"], t["boundaries"]
        )
        tr_recon_rot = tq4_decompress(
            tr_packed, tr_norms, t["centroids"], torch.float32
        )
        tr_recon = (tr_recon_rot.float() @ t["rotation"]).to(torch.float32)

        cos = torch.nn.functional.cosine_similarity(
            pt_recon.flatten().unsqueeze(0), tr_recon.flatten().unsqueeze(0)
        ).item()
        assert cos > 0.999, f"Round-trip cosine {cos:.6f} < 0.999 at head_dim={dim}"
