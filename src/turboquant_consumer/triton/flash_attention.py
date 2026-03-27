"""Triton Flash Attention v2 -- forward-only kernel with GQA support.

Phase 1 of the fused TQ4 Flash Attention roadmap (P5). This vanilla
kernel matches SDPA output and serves as the scaffold for injecting
TQ4 decompression at K/V tile load points in Phase 2.

Supports:
    - Grouped-Query Attention (GQA) with arbitrary Q/KV head ratios
    - Causal and non-causal modes
    - Optional additive attention mask (HF-compatible)
    - fp32 online softmax accumulation for numerical stability
    - RTX 4090 (SM89) and AMD ROCm via Triton HIP backend

Algorithm:
    Implements the online softmax from FlashAttention-2 (Dao 2023).
    Three fp32 state variables per query row -- running max ``m_i``,
    running softmax denominator ``l_i``, and output accumulator ``acc``
    -- are maintained across K/V tile iterations. The correction factor
    ``alpha = exp2(m_old - m_new)`` rescales prior accumulated work when
    the running maximum increases. This is mathematically exact, not
    approximate.

Attributes:
    triton_flash_attention: Python wrapper that launches the Triton
        kernel with autotuned block sizes.

Examples:
    ```python
    from turboquant_consumer.triton.flash_attention import triton_flash_attention

    out = triton_flash_attention(q, k, v)  # non-causal
    out = triton_flash_attention(q, k, v, is_causal=True)  # prefill
    ```

See Also:
    :mod:`turboquant_consumer.triton.attention_interface`:
        HuggingFace AttentionInterface registration.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Autotune configuration space
# ---------------------------------------------------------------------------
# Covers both prefill (large BLOCK_M) and decode (small BLOCK_M).
# Pruned: 8 warps only useful for large tile products.

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [16, 64, 128]
    for BN in [32, 64]
    for s in [2, 3]
    for w in [4, 8]
    if not (w == 8 and BM < 64)
]

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N_CTX_Q", "N_CTX_KV", "HEAD_DIM"])
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Mask,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_mz,
    stride_mh,
    stride_mm,
    stride_mn,
    H_Q,
    H_KV,
    N_CTX_Q,
    N_CTX_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Flash Attention forward kernel with online softmax in fp32.

    Grid: ``(cdiv(N_CTX_Q, BLOCK_M), batch * H_Q)``.
    Each CTA loads one Q tile and streams K/V tiles through it.
    """
    # -- Program indices --
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h_q = off_hz % H_Q

    # GQA: map each Q-head to its KV-head
    off_h_kv = off_h_q // (H_Q // H_KV)

    # -- Base pointers for this (batch, head) pair --
    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_kv * stride_kh
    v_base = V + off_z * stride_vz + off_h_kv * stride_vh
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh

    # -- Block offsets --
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # -- Load Q tile [BLOCK_M, HEAD_DIM] (stays in registers) --
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    # -- fp32 online softmax state --
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Pre-multiply scale by log2(e) so we can use exp2 (single PTX instr)
    qk_scale = sm_scale * 1.44269504  # log2(e)

    # Optional mask base pointer
    if HAS_MASK:
        mask_base = Mask + off_z * stride_mz + off_h_q * stride_mh

    # KV loop upper bound
    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX_KV)
    else:
        hi = N_CTX_KV

    # === Main tile loop over K/V blocks ===
    for start_n in range(0, hi, BLOCK_N):
        kv_valid = (start_n + offs_n) < N_CTX_KV

        # -- Load K tile [BLOCK_N, HEAD_DIM] and transpose --
        k_ptrs = (
            k_base
            + (start_n + offs_n[:, None]) * stride_kn
            + offs_d[None, :] * stride_kk
        )
        k = tl.load(k_ptrs, mask=kv_valid[:, None], other=0.0)

        # Q @ K^T -> [BLOCK_M, BLOCK_N] (fp32 via Tensor Core accumulation)
        qk = tl.dot(q, tl.trans(k))

        # Scale to log2 space
        qk = qk * qk_scale

        # Additive attention mask (HF-compatible: 0 = attend, -inf = block)
        if HAS_MASK:
            m_ptrs = (
                mask_base
                + offs_m[:, None] * stride_mm
                + (start_n + offs_n[None, :]) * stride_mn
            )
            m_valid = (offs_m[:, None] < N_CTX_Q) & kv_valid[None, :]
            mask_vals = tl.load(m_ptrs, mask=m_valid, other=0.0)
            # Convert natural-space mask to log2 space
            qk = qk + mask_vals * 1.44269504

        # Causal mask: position i can only attend to positions <= i
        if IS_CAUSAL:
            causal = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal, qk, float("-inf"))

        # Out-of-bounds KV positions -> -inf
        qk = tl.where(kv_valid[None, :], qk, float("-inf"))

        # -- Online softmax update --
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        # Rescale prior accumulated output
        acc = acc * alpha[:, None]

        # -- Load V tile [BLOCK_N, HEAD_DIM] --
        v_ptrs = (
            v_base
            + (start_n + offs_n[:, None]) * stride_vn
            + offs_d[None, :] * stride_vk
        )
        v = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)

        # Accumulate P @ V (fp16/bf16 matmul into fp32 accumulator)
        l_ij = tl.sum(p, 1)
        p_cast = p.to(v.dtype)
        acc = tl.dot(p_cast, v, acc)

        # Update running softmax statistics
        l_i = l_i * alpha + l_ij
        m_i = m_new

    # -- Epilogue: normalize and store --
    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=offs_m[:, None] < N_CTX_Q)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    is_causal: bool = False,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention using Triton Flash Attention.

    Args:
        q: Query tensor ``[batch, num_q_heads, seq_q, head_dim]``.
        k: Key tensor ``[batch, num_kv_heads, seq_kv, head_dim]``.
        v: Value tensor ``[batch, num_kv_heads, seq_kv, head_dim]``.
        sm_scale: Softmax scale factor. Defaults to ``1 / sqrt(head_dim)``.
        is_causal: Apply causal masking. Only valid when ``seq_q >= seq_kv``
            (prefill). For decode (``seq_q == 1``), forced to ``False``.
        attention_mask: Optional additive mask
            ``[batch, 1|heads, seq_q, seq_kv]``. Values of 0 mean attend,
            large negative values mean block. Mutually exclusive with
            ``is_causal`` in practice (HF sets ``is_causal`` only when
            ``attention_mask is None``).

    Returns:
        Attention output ``[batch, num_q_heads, seq_q, head_dim]``.
    """
    B, H_Q, N_Q, D = q.shape
    _, H_KV, N_KV, _ = k.shape

    assert q.dtype == k.dtype == v.dtype, "Q, K, V must have the same dtype"
    assert q.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Triton FA requires fp16 or bf16, got {q.dtype}"
    assert H_Q % H_KV == 0, f"Q heads ({H_Q}) must be divisible by KV heads ({H_KV})"
    assert k.shape[2] == v.shape[2], "K and V must have the same sequence length"
    assert q.shape[3] == k.shape[3] == v.shape[3], "Head dimensions must match"

    # Single query token never needs causal masking
    if is_causal and N_Q == 1:
        is_causal = False

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)

    # Handle optional mask
    has_mask = attention_mask is not None
    if has_mask:
        assert attention_mask is not None  # for type narrowing
        mask = attention_mask
        stride_mz, stride_mh, stride_mm, stride_mn = mask.stride()
        # Handle broadcast: size-1 dims have nonzero stride in PyTorch
        # but must be treated as stride=0 for correct broadcast behavior.
        if mask.shape[0] == 1:
            stride_mz = 0
        if mask.shape[1] == 1:
            stride_mh = 0
    else:
        mask = q  # dummy pointer, never dereferenced
        stride_mz = stride_mh = stride_mm = stride_mn = 0

    # Grid: one CTA per (Q-block, batch*q_head)
    def grid(META: dict) -> tuple[int, int]:
        """Compute kernel launch grid from autotuned block size.

        Returns:
            ``(num_q_blocks, batch * num_q_heads)`` grid dimensions.
        """
        return (triton.cdiv(N_Q, META["BLOCK_M"]), B * H_Q)

    _fwd_kernel[grid](
        q,
        k,
        v,
        out,
        mask,
        sm_scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        stride_mz,
        stride_mh,
        stride_mm,
        stride_mn,
        H_Q,
        H_KV,
        N_Q,
        N_KV,
        HEAD_DIM=D,
        IS_CAUSAL=is_causal,
        HAS_MASK=has_mask,
    )

    return out
