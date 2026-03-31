"""Shared helpers for INT8 prefill kernel tests."""

from __future__ import annotations

import math

import torch

from tests.helpers.paged_cache import BLOCK_SIZE
from turboquant_vllm.triton.tq4_decompress import tq4_decompress


def reference_causal_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Reference causal prefill: decompress from paged cache, attend, rotate.

    Returns:
        Attention output ``[num_tokens, H_Q, D]`` in original space.
    """
    num_tokens, H_Q, D = q.shape
    half_D = D // 2
    sm_scale = 1.0 / math.sqrt(D)
    gqa_ratio = H_Q // num_kv_heads
    k_idx_end = num_kv_heads * half_D
    k_norm_end = k_idx_end + num_kv_heads * 4
    v_idx_end = k_norm_end + num_kv_heads * half_D

    # Gather and decompress from pages
    k_packed_rows, k_norm_rows, v_packed_rows, v_norm_rows = [], [], [], []
    for t in range(seq_len):
        phys = block_table[0, t // BLOCK_SIZE].item()
        row = kv_cache[phys, t % BLOCK_SIZE]  # ty: ignore[invalid-argument-type]
        k_packed_rows.append(row[:k_idx_end].reshape(num_kv_heads, half_D))
        k_norm_rows.append(
            row[k_idx_end:k_norm_end]
            .contiguous()
            .view(torch.float32)
            .reshape(num_kv_heads, 1)
        )
        v_packed_rows.append(row[k_norm_end:v_idx_end].reshape(num_kv_heads, half_D))
        v_norm_rows.append(
            row[v_idx_end:].contiguous().view(torch.float32).reshape(num_kv_heads, 1)
        )

    k_dec = tq4_decompress(
        torch.stack(k_packed_rows), torch.stack(k_norm_rows), centroids, dtype=q.dtype
    )
    v_dec = tq4_decompress(
        torch.stack(v_packed_rows), torch.stack(v_norm_rows), centroids, dtype=q.dtype
    )

    # Pre-rotate Q
    q_rot = (q.float() @ rotation.T).to(q.dtype)

    # Batched causal attention with GQA expansion
    k_exp = k_dec.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, H_Q, D)
    v_exp = v_dec.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, H_Q, D)

    q_perm = q_rot.permute(1, 0, 2).float()  # (H_Q, S, D)
    k_perm = k_exp.permute(1, 0, 2).float()
    v_perm = v_exp.permute(1, 0, 2).float()

    scores = torch.bmm(q_perm, k_perm.transpose(1, 2)) * sm_scale
    causal_mask = torch.triu(
        torch.ones(num_tokens, seq_len, device=q.device), diagonal=1
    ).bool()
    scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out_rot = torch.bmm(weights, v_perm).permute(1, 0, 2)  # (S, H_Q, D)

    return (out_rot @ rotation.float()).to(q.dtype)
