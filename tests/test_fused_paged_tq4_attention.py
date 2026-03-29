"""Tests for fused paged TQ4 decode attention kernel.

Validates that the fused paged kernel (reading compressed blocks directly
from the page table) matches the decompress-all reference path at two
precision tiers:

- Cache parity (>0.999): fused paged vs decompress-all + flash_attn
- Kernel correctness (>0.998): fused paged vs contiguous reference

All tests require CUDA (Triton does not support CPU).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.flash_attention_tq4_kv import (
    triton_flash_attention_tq4_kv,
)
from turboquant_vllm.triton.fused_paged_tq4_attention import fused_paged_tq4_decode
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

pytestmark = [pytest.mark.gpu]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
HEAD_DIM = 128
HALF_D = HEAD_DIM // 2
TQ4_BITS = 4
BLOCK_SIZE = 16  # vLLM default page size
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_paged_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    seq_lens: list[int],
    quantizer: TurboQuantMSE,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a paged KV cache using the production compress path.

    Compresses K/V via ``tq4_compress`` and packs into the same byte
    layout as ``TQ4AttentionImpl._compress_and_store``.

    Args:
        keys: ``[total_tokens, num_kv_heads, head_dim]`` fp16.
        values: ``[total_tokens, num_kv_heads, head_dim]`` fp16.
        seq_lens: List of sequence lengths.
        quantizer: TQ4 quantizer with rotation and codebook.
        device: Device string.

    Returns:
        ``(kv_cache, block_table, seq_lens_tensor)`` where:
        - kv_cache: ``[num_blocks, block_size, total_bytes]`` uint8
        - block_table: ``[num_seqs, max_blocks_per_seq]`` int32
        - seq_lens_tensor: ``[num_seqs]`` int32
    """
    total_tokens, num_kv_heads, D = keys.shape
    half_D = D // 2

    # Byte layout offsets
    k_idx_end = num_kv_heads * half_D
    k_norm_end = k_idx_end + num_kv_heads * 4
    v_idx_end = k_norm_end + num_kv_heads * half_D
    total_bytes = v_idx_end + num_kv_heads * 4

    # Pre-split rotation for compress kernel
    rot_t = quantizer.rotation.T.contiguous().to(device)
    rot_t_even = rot_t[:, 0::2].contiguous()
    rot_t_odd = rot_t[:, 1::2].contiguous()
    boundaries = quantizer.codebook.boundaries.to(device)

    # Compress K and V using production code path
    k_packed, k_norms = tq4_compress(keys, rot_t_even, rot_t_odd, boundaries)
    v_packed, v_norms = tq4_compress(values, rot_t_even, rot_t_odd, boundaries)

    # Allocate pages
    max_blocks_per_seq = max((sl + BLOCK_SIZE - 1) // BLOCK_SIZE for sl in seq_lens)
    num_seqs = len(seq_lens)
    total_blocks = sum((sl + BLOCK_SIZE - 1) // BLOCK_SIZE for sl in seq_lens)

    kv_cache = torch.zeros(
        total_blocks, BLOCK_SIZE, total_bytes, dtype=torch.uint8, device=device
    )
    block_table = torch.zeros(
        num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device
    )

    # Fill pages: assign blocks sequentially
    block_idx = 0
    token_offset = 0
    for seq_i, sl in enumerate(seq_lens):
        num_blocks_for_seq = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(num_blocks_for_seq):
            block_table[seq_i, b] = block_idx
            start = token_offset + b * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, token_offset + sl)
            num_tokens_in_block = end - start

            # Pack row: [K_idx | K_norms | V_idx | V_norms]
            for t in range(num_tokens_in_block):
                global_t = start + t
                row = kv_cache[block_idx, t]
                # K indices
                row[:k_idx_end] = k_packed[global_t].reshape(-1)
                # K norms (fp32 as bytes)
                row[k_idx_end:k_norm_end] = (
                    k_norms[global_t]
                    .reshape(num_kv_heads)
                    .contiguous()
                    .view(torch.uint8)
                )
                # V indices
                row[k_norm_end:v_idx_end] = v_packed[global_t].reshape(-1)
                # V norms (fp32 as bytes)
                row[v_idx_end:] = (
                    v_norms[global_t]
                    .reshape(num_kv_heads)
                    .contiguous()
                    .view(torch.uint8)
                )

            block_idx += 1
        token_offset += sl

    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return kv_cache, block_table, seq_lens_tensor


def _reference_decompress_and_attend(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    num_kv_heads: int,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Reference path: decompress all pages, then run attention.

    Extracts compressed data from paged cache, decompresses via
    ``tq4_decompress``, and computes attention using manual softmax.

    Returns:
        Attention output ``[num_seqs, H_Q, head_dim]`` in original space.
    """
    num_seqs, H_Q, D = q.shape
    half_D = D // 2
    k_idx_end = num_kv_heads * half_D
    k_norm_end = k_idx_end + num_kv_heads * 4
    v_idx_end = k_norm_end + num_kv_heads * half_D
    sm_scale = 1.0 / math.sqrt(D)

    outputs = []
    for seq_i in range(num_seqs):
        sl = seq_lens[seq_i]
        num_blocks = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Gather tokens from pages
        all_k_packed = []
        all_k_norms = []
        all_v_packed = []
        all_v_norms = []

        for b in range(num_blocks):
            phys = block_table[seq_i, b].item()
            start_t = b * BLOCK_SIZE
            end_t = min(start_t + BLOCK_SIZE, sl)
            for t in range(start_t, end_t):
                within = t % BLOCK_SIZE
                row = kv_cache[phys, within]  # ty: ignore[invalid-argument-type]
                all_k_packed.append(row[:k_idx_end].reshape(num_kv_heads, half_D))
                all_k_norms.append(
                    row[k_idx_end:k_norm_end]
                    .contiguous()
                    .view(torch.float32)
                    .reshape(num_kv_heads, 1)
                )
                all_v_packed.append(
                    row[k_norm_end:v_idx_end].reshape(num_kv_heads, half_D)
                )
                all_v_norms.append(
                    row[v_idx_end:]
                    .contiguous()
                    .view(torch.float32)
                    .reshape(num_kv_heads, 1)
                )

        k_packed_seq = torch.stack(all_k_packed, dim=0)  # (sl, H_KV, HALF_D)
        k_norms_seq = torch.stack(all_k_norms, dim=0)  # (sl, H_KV, 1)
        v_packed_seq = torch.stack(all_v_packed, dim=0)
        v_norms_seq = torch.stack(all_v_norms, dim=0)

        # Decompress (in rotated space)
        k_dec = tq4_decompress(k_packed_seq, k_norms_seq, centroids, dtype=q.dtype)
        v_dec = tq4_decompress(v_packed_seq, v_norms_seq, centroids, dtype=q.dtype)

        # Pre-rotate Q
        q_seq = q[seq_i]  # (H_Q, D)
        q_rot = torch.matmul(q_seq.float(), rotation.T).to(q.dtype)

        # Attention: Q_rot @ K_rot^T (GQA-aware)
        # k_dec: (sl, H_KV, D) -> (H_KV, sl, D)
        k_t = k_dec.permute(1, 0, 2)  # (H_KV, sl, D)
        v_t = v_dec.permute(1, 0, 2)  # (H_KV, sl, D)

        group_size = H_Q // num_kv_heads
        out_heads = []
        for h_q in range(H_Q):
            h_kv = h_q // group_size
            q_h = q_rot[h_q]  # (D,)
            k_h = k_t[h_kv]  # (sl, D)
            v_h = v_t[h_kv]  # (sl, D)

            scores = (q_h @ k_h.T) * sm_scale  # (sl,)
            attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            out_h = attn @ v_h  # (D,)
            out_heads.append(out_h)

        # Post-rotate output
        out_rot = torch.stack(out_heads, dim=0)  # (H_Q, D)
        out_orig = torch.matmul(out_rot.float(), rotation).to(q.dtype)
        outputs.append(out_orig)

    return torch.stack(outputs, dim=0)  # (num_seqs, H_Q, D)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flat cosine similarity."""
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


# ---------------------------------------------------------------------------
# GQA parametrization
# ---------------------------------------------------------------------------

GQA_CONFIGS = [
    pytest.param(28, 4, id="molmo2_28q_4kv"),
    pytest.param(32, 8, id="llama_32q_8kv"),
    pytest.param(32, 8, id="mistral_32q_8kv"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFusedPagedTQ4Decode:
    """Fused paged TQ4 decode kernel correctness tests."""

    @pytest.mark.parametrize(("H_Q", "H_KV"), GQA_CONFIGS)
    def test_cache_parity(
        self,
        device: str,
        tq4_quantizer: TurboQuantMSE,
        H_Q: int,
        H_KV: int,
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
                f"Layer {layer_idx}: cosine {cos:.6f} < {CACHE_PARITY_THRESHOLD}"
            )
