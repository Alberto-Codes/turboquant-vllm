"""Fused paged TQ4 decode attention -- decompresses directly from page table.

Phase 3a of the D9 kernel roadmap.  This kernel reads TQ4-compressed
blocks directly from vLLM's paged block table, decompresses in SRAM
(nibble unpack -> centroid gather -> norm scale), and computes FP16
Q@K^T with online softmax in a single fused pass.  No HBM writes of
decompressed cache -- HBM traffic drops from 1,160 to 136 bytes/token
(8.5x reduction).

The kernel operates entirely in **rotated space**.  The caller
pre-rotates Q by ``Pi^T`` and post-rotates the output by ``Pi``.
Decompression does NOT apply rotation (matching ``tq4_decompress.py``).

Scope: FP16 Q decode path only (``USE_INT8_QK=False``).  INT8 path
is Story 6.4.  Placeholder parameters are included for forward
compatibility but compiled out by the constexpr switch.

Attributes:
    fused_paged_tq4_decode: Python wrapper that pre-rotates Q,
        launches the fused paged kernel, and post-rotates the output.

Examples:
    ```python
    from turboquant_vllm.triton.fused_paged_tq4_attention import (
        fused_paged_tq4_decode,
    )

    out = fused_paged_tq4_decode(
        q,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        rotation,
        num_kv_heads=4,
        head_dim=128,
        block_size=16,
    )
    ```

See Also:
    :mod:`turboquant_vllm.triton.flash_attention_tq4_kv`: Contiguous
        (non-paged) reference kernel -- correctness baseline.
    :mod:`turboquant_vllm.triton.tq4_decompress`: Standalone decompress.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Autotune configs (key=["HEAD_DIM"] only -- no seq_len, no num_kv_heads)
# ---------------------------------------------------------------------------

_FUSED_DECODE_CONFIGS = [
    triton.Config({"BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BN in [16, 32, 64]
    for s in [2, 3]
    for w in [4, 8]
]

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_FUSED_DECODE_CONFIGS, key=["HEAD_DIM"])
@triton.jit
def _fused_paged_tq4_decode_kernel(
    # ── Queries (pre-rotated by Pi^T) ──
    Q_rot,
    # ── Compressed KV cache (paged) ──
    KV_cache,
    # ── Page table and sequence metadata ──
    Block_table,
    Seq_lens,
    # ── TQ4 codebook ──
    Centroids,
    # ── Output ──
    Out,
    # ── Optional INT8 path inputs (unused when USE_INT8_QK=False) ──
    Q_scale,
    QJL_S,
    QJL_signs,
    QJL_residual_norms,
    # ── Stride parameters ──
    stride_qz,
    stride_qh,
    stride_qk,
    stride_cache_block,
    stride_cache_token,
    stride_bt_seq,
    stride_bt_block,
    stride_oz,
    stride_oh,
    stride_ok,
    # ── Compile-time constants ──
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HALF_D: tl.constexpr,
    K_NORM_OFFSET: tl.constexpr,
    V_IDX_OFFSET: tl.constexpr,
    V_NORM_OFFSET: tl.constexpr,
    # ── Dual-path switch ──
    USE_INT8_QK: tl.constexpr = False,  # ty: ignore[invalid-parameter-default]
    QJL_DIM: tl.constexpr = 0,  # ty: ignore[invalid-parameter-default]
    # ── Tiling (autotuned) ──
    BLOCK_N: tl.constexpr = 32,  # ty: ignore[invalid-parameter-default]
):
    """Fused paged TQ4 decode attention kernel.

    One program per (sequence, query head).  Loops over all KV tiles,
    decompresses in-tile from paged cache, and computes attention with
    online softmax.  Output is in rotated space (caller post-rotates).
    """
    # Grid: (num_seqs, H_Q)
    off_seq = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_h_kv = off_h_q // (H_Q // H_KV)

    seq_len = tl.load(Seq_lens + off_seq)

    # Load Q row into registers (BLOCK_M=1 for decode)
    q_base = Q_rot + off_seq * stride_qz + off_h_q * stride_qh
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_base + offs_d * stride_qk)  # (HEAD_DIM,) fp16

    # fp32 online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32) + 1.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    qk_scale: tl.constexpr = sm_scale * 1.44269504

    offs_d_half = tl.arange(0, HALF_D)

    # === Main KV tile loop ===
    for start_n in range(0, seq_len, BLOCK_N):
        offs_t = start_n + tl.arange(0, BLOCK_N)
        kv_valid = offs_t < seq_len

        # -- Level 1: Block table lookup --
        logical_block = offs_t // BLOCK_SIZE
        within_block = offs_t % BLOCK_SIZE
        physical_block = tl.load(
            Block_table + off_seq * stride_bt_seq + logical_block * stride_bt_block,
            mask=kv_valid,
            other=0,
        )

        # -- Level 2: Compute base byte address per token --
        token_base = (
            physical_block * stride_cache_block + within_block * stride_cache_token
        )

        # -- K decompression --
        # K indices: offset 0 + kv_head * HALF_D
        k_idx_addr = (
            KV_cache + token_base[:, None] + off_h_kv * HALF_D + offs_d_half[None, :]
        )
        k_packed = tl.load(k_idx_addr, mask=kv_valid[:, None], other=0)

        # Nibble unpack
        k_hi = (k_packed >> 4).to(tl.int32)
        k_lo = (k_packed & 0x0F).to(tl.int32)

        # Centroid gather + interleave
        k = tl.join(tl.load(Centroids + k_hi), tl.load(Centroids + k_lo)).reshape(
            BLOCK_N, HEAD_DIM
        )

        # K norms (fp32, 4 bytes per head) -- load 4 uint8 bytes, reconstruct fp32
        k_norm_byte_addr = KV_cache + token_base + K_NORM_OFFSET + off_h_kv * 4
        b0 = tl.load(k_norm_byte_addr, mask=kv_valid, other=0).to(tl.int32)
        b1 = tl.load(k_norm_byte_addr + 1, mask=kv_valid, other=0).to(tl.int32)
        b2 = tl.load(k_norm_byte_addr + 2, mask=kv_valid, other=0).to(tl.int32)
        b3 = tl.load(k_norm_byte_addr + 3, mask=kv_valid, other=0).to(tl.int32)
        k_norm_bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        k_norms = k_norm_bits.to(tl.float32, bitcast=True)

        # Norm scale + cast to fp16
        k = (k * k_norms[:, None]).to(Q_rot.dtype.element_ty)

        # -- Q @ K^T (dot product for BLOCK_M=1) --
        qk = tl.sum(q[None, :] * k, axis=1)  # (BLOCK_N,) fp32-promoted
        qk = qk * qk_scale

        # Sequence length masking
        qk = tl.where(kv_valid, qk, float("-inf"))

        # -- Online softmax update --
        m_ij = tl.max(qk, 0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)
        acc = acc * alpha
        l_ij = tl.sum(p, 0)

        # -- V decompression --
        v_idx_addr = (
            KV_cache
            + token_base[:, None]
            + V_IDX_OFFSET
            + off_h_kv * HALF_D
            + offs_d_half[None, :]
        )
        v_packed = tl.load(v_idx_addr, mask=kv_valid[:, None], other=0)

        v_hi = (v_packed >> 4).to(tl.int32)
        v_lo = (v_packed & 0x0F).to(tl.int32)
        v = tl.join(tl.load(Centroids + v_hi), tl.load(Centroids + v_lo)).reshape(
            BLOCK_N, HEAD_DIM
        )

        # V norms
        v_norm_byte_addr = KV_cache + token_base + V_NORM_OFFSET + off_h_kv * 4
        vb0 = tl.load(v_norm_byte_addr, mask=kv_valid, other=0).to(tl.int32)
        vb1 = tl.load(v_norm_byte_addr + 1, mask=kv_valid, other=0).to(tl.int32)
        vb2 = tl.load(v_norm_byte_addr + 2, mask=kv_valid, other=0).to(tl.int32)
        vb3 = tl.load(v_norm_byte_addr + 3, mask=kv_valid, other=0).to(tl.int32)
        v_norm_bits = vb0 | (vb1 << 8) | (vb2 << 16) | (vb3 << 24)
        v_norms = v_norm_bits.to(tl.float32, bitcast=True)

        v = (v * v_norms[:, None]).to(Q_rot.dtype.element_ty)

        # -- P @ V accumulation --
        acc += tl.sum(p[:, None] * v, axis=0)

        l_i = l_i * alpha + l_ij
        m_i = m_new

    # Epilogue: normalize and store (output in rotated space)
    acc = acc / l_i
    o_base = Out + off_seq * stride_oz + off_h_q * stride_oh
    tl.store(o_base + offs_d * stride_ok, acc.to(Q_rot.dtype.element_ty))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def fused_paged_tq4_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused paged TQ4 decode attention.

    Pre-rotates Q by ``rotation^T``, launches the fused paged kernel
    that decompresses TQ4 blocks in-tile from the page table, then
    post-rotates the output by ``rotation`` to return to original space.

    Args:
        q: Query ``[num_seqs, H_Q, head_dim]`` fp16/bf16 (one token per seq).
        kv_cache: Packed paged cache ``[num_blocks, block_size, total_bytes]``
            uint8.
        block_table: Page table ``[num_seqs, max_num_blocks_per_seq]`` int32.
        seq_lens: Sequence lengths ``[num_seqs]`` int32.
        centroids: TQ4 codebook ``[16]`` fp32.
        rotation: Orthogonal rotation ``[head_dim, head_dim]`` fp32.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension (e.g. 128).
        block_size: vLLM page size (tokens per block).
        sm_scale: Softmax scale.  Defaults to ``1 / sqrt(head_dim)``.
        out: Optional pre-allocated output ``[num_seqs, H_Q, head_dim]``.

    Returns:
        Attention output ``[num_seqs, H_Q, head_dim]`` in original space.

    Note:
        INT8 placeholder parameters (``Q_scale``, ``QJL_S``, ``QJL_signs``,
        ``QJL_residual_norms``) should be passed as ``None``/zeros when
        ``USE_INT8_QK=False`` (the default for Phase 3a).
    """
    num_seqs, H_Q, D = q.shape

    assert D == head_dim
    assert H_Q % num_kv_heads == 0
    assert kv_cache.dtype == torch.uint8
    assert block_table.dtype == torch.int32
    assert seq_lens.dtype == torch.int32

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    half_D = head_dim // 2

    # Byte layout constexprs
    k_norm_offset = num_kv_heads * half_D
    v_idx_offset = k_norm_offset + num_kv_heads * 4
    v_norm_offset = v_idx_offset + num_kv_heads * half_D

    # Pre-rotate Q by Pi^T (O(num_seqs), not O(cache_len))
    q_rot = torch.matmul(q.float(), rotation.T).to(q.dtype)

    if out is None:
        out_rot = torch.empty_like(q)
    else:
        out_rot = out

    # INT8 placeholders (unused, compiled out)
    dummy = torch.empty(0, device=q.device)

    grid = (num_seqs, H_Q)

    _fused_paged_tq4_decode_kernel[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        out_rot,
        dummy,  # Q_scale
        dummy,  # QJL_S
        dummy,  # QJL_signs
        dummy,  # QJL_residual_norms
        q_rot.stride(0),
        q_rot.stride(1),
        q_rot.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        out_rot.stride(0),
        out_rot.stride(1),
        out_rot.stride(2),
        sm_scale=sm_scale,
        H_Q=H_Q,
        H_KV=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        HALF_D=half_D,
        K_NORM_OFFSET=k_norm_offset,
        V_IDX_OFFSET=v_idx_offset,
        V_NORM_OFFSET=v_norm_offset,
        USE_INT8_QK=False,
        QJL_DIM=0,
    )

    # Post-rotate: convert from rotated space back to original space
    return torch.matmul(out_rot.float(), rotation).to(q.dtype)
