"""Shared paged cache builder and reference attention for fused kernel tests.

Extracted from ``test_fused_paged_tq4_attention.py`` (Test Maturity MEDIUM 5)
to be reusable by decode, INT8 prefill, and composition test modules.
"""

from __future__ import annotations

import math
import random

import torch

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEAD_DIM = 128
HALF_D = HEAD_DIM // 2
TQ4_BITS = 4
BLOCK_SIZE = 16
SEED = 42


# ---------------------------------------------------------------------------
# Paged cache builder
# ---------------------------------------------------------------------------


def build_paged_cache(
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

    # Shuffle physical block assignment to test non-contiguous access.
    phys_order = list(range(total_blocks))
    random.Random(SEED).shuffle(phys_order)

    block_idx = 0
    token_offset = 0
    for seq_i, sl in enumerate(seq_lens):
        num_blocks_for_seq = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(num_blocks_for_seq):
            phys_block = phys_order[block_idx]
            block_table[seq_i, b] = phys_block
            start = token_offset + b * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, token_offset + sl)
            num_tokens_in_block = end - start

            # Pack row: [K_idx | K_norms | V_idx | V_norms]
            for t in range(num_tokens_in_block):
                global_t = start + t
                row = kv_cache[phys_block, t]
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


# ---------------------------------------------------------------------------
# Reference attention (decompress-all path)
# ---------------------------------------------------------------------------


def reference_decompress_and_attend(
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
