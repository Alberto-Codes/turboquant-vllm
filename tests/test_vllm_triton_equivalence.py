"""Tests for Triton vs PyTorch bit-for-bit validation.

Phase 3c.10 quality gate: the fused Triton compress/decompress kernels
must match the original multi-op PyTorch implementation exactly.
Requires vLLM to be installed.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402


class TestTritonPyTorchEquivalence:
    """Validate Triton kernels produce identical output to the old PyTorch path.

    Phase 3c.10 quality gate: the fused Triton compress/decompress kernels
    must match the original multi-op PyTorch implementation exactly.
    """

    HEAD_DIM = 128
    NUM_KV_HEADS = 8

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up quantizer primitives for both paths."""
        from turboquant_vllm.quantizer import TurboQuantMSE
        from turboquant_vllm.vllm.tq4_backend import TQ4_BITS, TQ4_SEED

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quantizer = TurboQuantMSE(self.HEAD_DIM, TQ4_BITS, seed=TQ4_SEED)
        self.rotation = quantizer.rotation.to(self.device)
        self.rotation_t = self.rotation.T.contiguous()
        self.boundaries = quantizer.codebook.boundaries.to(self.device)
        self.centroids = quantizer.codebook.centroids.to(self.device)
        self.rot_T_even = self.rotation_t[:, 0::2].contiguous()
        self.rot_T_odd = self.rotation_t[:, 1::2].contiguous()

    # --- helpers: old PyTorch path ---

    def _pytorch_compress(self, x):
        """Original PyTorch compress (pre-3c.9)."""
        N, H, D = x.shape
        flat = x.reshape(N * H, D).float()
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ self.rotation_t
        indices = torch.bucketize(rotated, self.boundaries)
        indices = indices.clamp(0, 15)
        idx_u8 = indices.to(torch.uint8)
        packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]
        return packed.reshape(N, H, D // 2), norms.reshape(N, H, 1)

    def _pytorch_decompress(self, packed, norms, dtype):
        """Original PyTorch decompress (pre-3c.8), with rotation."""
        N, H, half_D = packed.shape
        D = half_D * 2
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(N * H, D)
        flat_norms = norms.reshape(N * H, 1)
        reconstructed = self.centroids[indices]
        unrotated = reconstructed @ self.rotation
        result = unrotated * flat_norms
        return result.reshape(N, H, D).to(dtype)

    # --- compress tests ---

    def test_compress_packed_match(self):
        """Triton compress produces identical packed bytes as PyTorch."""
        from turboquant_vllm.triton.tq4_compress import tq4_compress

        x = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )
        pt_packed, pt_norms = self._pytorch_compress(x)
        tr_packed, tr_norms = tq4_compress(
            x,
            self.rot_T_even,
            self.rot_T_odd,
            self.boundaries,
        )
        assert torch.equal(pt_packed, tr_packed), "Packed bytes differ"

    def test_compress_norms_match(self):
        """Triton compress produces identical norms as PyTorch."""
        from turboquant_vllm.triton.tq4_compress import tq4_compress

        x = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )
        pt_packed, pt_norms = self._pytorch_compress(x)
        tr_packed, tr_norms = tq4_compress(
            x,
            self.rot_T_even,
            self.rot_T_odd,
            self.boundaries,
        )
        torch.testing.assert_close(pt_norms, tr_norms, atol=1e-5, rtol=1e-5)

    def test_compress_single_token(self):
        """Triton compress matches PyTorch for a single token (decode case)."""
        from turboquant_vllm.triton.tq4_compress import tq4_compress

        x = torch.randn(
            1,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )
        pt_packed, pt_norms = self._pytorch_compress(x)
        tr_packed, tr_norms = tq4_compress(
            x,
            self.rot_T_even,
            self.rot_T_odd,
            self.boundaries,
        )
        assert torch.equal(pt_packed, tr_packed), "Single-token packed bytes differ"
        torch.testing.assert_close(pt_norms, tr_norms, atol=1e-5, rtol=1e-5)

    # --- decompress tests ---

    def test_decompress_with_rotation_matches_pytorch(self):
        """Triton decompress + rotation matches old PyTorch decompress."""
        from turboquant_vllm.triton.tq4_decompress import tq4_decompress

        packed = torch.randint(
            0,
            255,
            (16, self.NUM_KV_HEADS, self.HEAD_DIM // 2),
            device=self.device,
            dtype=torch.uint8,
        )
        norms = (
            torch.randn(
                16,
                self.NUM_KV_HEADS,
                1,
                device=self.device,
                dtype=torch.float32,
            )
            .abs_()
            .clamp_(min=0.1)
        )

        # Old path: decompress with rotation
        pt_out = self._pytorch_decompress(packed, norms, torch.float32)

        # New path: Triton decompress (no rotation) + apply rotation
        tr_out_rot = tq4_decompress(packed, norms, self.centroids, torch.float32)
        tr_out = (tr_out_rot.float() @ self.rotation).to(torch.float32)

        torch.testing.assert_close(pt_out, tr_out, atol=1e-4, rtol=1e-4)

    def test_decompress_no_rotation_stays_in_rotated_space(self):
        """Triton decompress without rotation is NOT close to original data."""
        from turboquant_vllm.triton.tq4_decompress import tq4_decompress

        x = torch.randn(
            1,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )
        pt_packed, pt_norms = self._pytorch_compress(x)

        # Decompress without rotation: should be in rotated space
        tr_out_rot = tq4_decompress(pt_packed, pt_norms, self.centroids, torch.float32)
        # This should NOT match the original (it's rotated)
        cos = torch.nn.functional.cosine_similarity(
            x[0, 0].float().unsqueeze(0).to(self.device),
            tr_out_rot[0, 0].unsqueeze(0),
        ).item()
        # Rotated output should have low cosine with original
        assert cos < 0.5, f"Expected low cosine (rotated space), got {cos:.4f}"

    # --- end-to-end: pre/post-rotation equivalence ---

    def test_pre_post_rotation_attention_equivalence(self):
        """Pre-rotate Q + Triton decompress + post-rotate == old path.

        This is the core 3c.10 validation: the full attention output must
        be equivalent whether we rotate inside decompress (old) or rotate
        Q/output outside (new).
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for Flash Attention")

        from vllm.vllm_flash_attn import flash_attn_varlen_func

        from turboquant_vllm.triton.tq4_decompress import tq4_decompress

        seq_len = 64
        packed = torch.randint(
            0,
            255,
            (seq_len, self.NUM_KV_HEADS, self.HEAD_DIM // 2),
            device=self.device,
            dtype=torch.uint8,
        )
        norms = (
            torch.randn(
                seq_len,
                self.NUM_KV_HEADS,
                1,
                device=self.device,
                dtype=torch.float32,
            )
            .abs_()
            .clamp_(min=0.1)
        )
        query = torch.randn(
            1,
            32,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )

        cu_q = torch.tensor([0, 1], device=self.device, dtype=torch.int32)
        cu_k = torch.tensor([0, seq_len], device=self.device, dtype=torch.int32)

        # Old path: decompress with rotation → standard FA
        k_old = self._pytorch_decompress(packed, norms, torch.float16)
        v_old = self._pytorch_decompress(packed, norms, torch.float16)
        out_old = flash_attn_varlen_func(
            q=query,
            k=k_old,
            v=v_old,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=seq_len,
            cu_seqlens_k=cu_k,
        )

        # New path: pre-rotate Q → Triton decompress (no rotation) → FA → post-rotate
        q_rot = (query.float() @ self.rotation_t).to(torch.float16)
        k_rot = tq4_decompress(packed, norms, self.centroids, torch.float16)
        v_rot = tq4_decompress(packed, norms, self.centroids, torch.float16)
        out_new_rot = flash_attn_varlen_func(
            q=q_rot,
            k=k_rot,
            v=v_rot,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=seq_len,
            cu_seqlens_k=cu_k,
        )
        out_new = (out_new_rot.float() @ self.rotation).to(torch.float16)

        # Should be very close (not exact due to fp16 accumulation order)
        torch.testing.assert_close(out_old, out_new, atol=5e-3, rtol=5e-3)

    # --- round-trip: compress then decompress ---

    def test_full_round_trip_triton_vs_pytorch(self):
        """Full round-trip: Triton compress then decompress matches PyTorch.

        Triton compress → Triton decompress + rotation should produce the
        same reconstructed vectors as PyTorch compress → PyTorch decompress.
        """
        from turboquant_vllm.triton.tq4_compress import tq4_compress
        from turboquant_vllm.triton.tq4_decompress import tq4_decompress

        x = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            device=self.device,
            dtype=torch.float16,
        )

        # PyTorch round-trip
        pt_packed, pt_norms = self._pytorch_compress(x)
        pt_recon = self._pytorch_decompress(pt_packed, pt_norms, torch.float32)

        # Triton round-trip (compress → decompress + rotation)
        tr_packed, tr_norms = tq4_compress(
            x,
            self.rot_T_even,
            self.rot_T_odd,
            self.boundaries,
        )
        tr_recon_rot = tq4_decompress(
            tr_packed,
            tr_norms,
            self.centroids,
            torch.float32,
        )
        tr_recon = (tr_recon_rot.float() @ self.rotation).to(torch.float32)

        torch.testing.assert_close(pt_recon, tr_recon, atol=1e-4, rtol=1e-4)
