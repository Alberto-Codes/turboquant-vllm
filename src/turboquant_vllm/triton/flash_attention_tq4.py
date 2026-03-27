"""Fused TQ4 Flash Attention -- K decompression inside the FA inner loop.

Phase 2 of the P5 roadmap. Replaces the standard K tile load with:
nibble unpack -> centroid gather -> interleave -> norm scale. The query
is pre-rotated by ``Pi^T`` outside the kernel. Values remain standard
fp16/bf16.

The fp32 online softmax ``(m_i, l_i, acc)`` state machine prevents the
0.023/layer cosine drift that killed the Q@K^T-only kernel (Key Lesson #7).

Attributes:
    triton_flash_attention_tq4: Python wrapper that pre-rotates Q and
        launches the fused TQ4 kernel.

Examples:
    ```python
    from turboquant_vllm.triton.flash_attention_tq4 import (
        triton_flash_attention_tq4,
    )

    out = triton_flash_attention_tq4(
        q,
        k_packed,
        k_norms,
        centroids,
        rotation_matrix,
        v,
    )
    ```

See Also:
    :mod:`turboquant_vllm.triton.flash_attention`: Phase 1 vanilla kernel.
    :mod:`turboquant_vllm.quantizer`: TurboQuantMSE rotation + quantization.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Autotune configuration space
# ---------------------------------------------------------------------------

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


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N_CTX_Q", "HEAD_DIM"])
@triton.jit
def _fwd_tq4_kernel(
    Q_rot,
    K_packed,
    K_norms,
    Centroids,
    V,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kpz,
    stride_kph,
    stride_kpn,
    stride_kpd,
    stride_knz,
    stride_knh,
    stride_knn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    H_Q,
    H_KV,
    N_CTX_Q,
    N_CTX_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Fused TQ4 Flash Attention forward kernel.

    K tiles are decompressed inline from nibble-packed uint8 indices via
    centroid gather. V tiles are loaded as standard fp16/bf16. Online
    softmax maintained in fp32 throughout.

    Autotuned on ``(N_CTX_Q, HEAD_DIM)`` to avoid per-decode-step retuning.
    """
    HALF_D: tl.constexpr = HEAD_DIM // 2

    # -- Program indices --
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h_q = off_hz % H_Q
    off_h_kv = off_h_q // (H_Q // H_KV)

    # -- Base pointers --
    q_base = Q_rot + off_z * stride_qz + off_h_q * stride_qh
    kp_base = K_packed + off_z * stride_kpz + off_h_kv * stride_kph
    kn_base = K_norms + off_z * stride_knz + off_h_kv * stride_knh
    v_base = V + off_z * stride_vz + off_h_kv * stride_vh
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh

    # -- Block offsets --
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_d_half = tl.arange(0, HALF_D)

    # -- Load Q_rot tile [BLOCK_M, HEAD_DIM] --
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    # -- fp32 online softmax state --
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    # KV loop bound
    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX_KV)
    else:
        hi = N_CTX_KV

    # === Main tile loop ===
    for start_n in range(0, hi, BLOCK_N):
        kv_valid = (start_n + offs_n) < N_CTX_KV

        # -- TQ4 K decompression --
        # Load nibble-packed indices: [BLOCK_N, HALF_D] uint8
        kp_ptrs = (
            kp_base
            + (start_n + offs_n[:, None]) * stride_kpn
            + offs_d_half[None, :] * stride_kpd
        )
        packed = tl.load(kp_ptrs, mask=kv_valid[:, None], other=0)

        # Nibble unpack: two 4-bit indices per byte
        hi_idx = (packed >> 4).to(tl.int32)
        lo_idx = (packed & 0x0F).to(tl.int32)

        # Centroid gather: [BLOCK_N, HALF_D] fp32
        k_hi = tl.load(Centroids + hi_idx)
        k_lo = tl.load(Centroids + lo_idx)

        # Interleave to [BLOCK_N, HEAD_DIM]: even=hi, odd=lo
        k = tl.join(k_hi, k_lo).reshape(BLOCK_N, HEAD_DIM)

        # Load norms: [BLOCK_N] fp32
        kn_ptrs = kn_base + (start_n + offs_n) * stride_knn
        norms = tl.load(kn_ptrs, mask=kv_valid, other=0.0)

        # Scale by norms + cast to input dtype for Tensor Core dot
        k = (k * norms[:, None]).to(Q_rot.dtype.element_ty)

        # Q_rot @ K_decompressed^T -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale

        # Causal mask
        if IS_CAUSAL:
            causal = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal, qk, float("-inf"))

        # OOB mask
        qk = tl.where(kv_valid[None, :], qk, float("-inf"))

        # -- Online softmax update --
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        acc = acc * alpha[:, None]

        # -- V tile (standard, not compressed) --
        v_ptrs = (
            v_base
            + (start_n + offs_n[:, None]) * stride_vn
            + offs_d[None, :] * stride_vk
        )
        v = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)

        # P @ V
        l_ij = tl.sum(p, 1)
        p_cast = p.to(v.dtype)
        acc = tl.dot(p_cast, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_new

    # -- Epilogue --
    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Q_rot.dtype.element_ty), mask=offs_m[:, None] < N_CTX_Q)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def triton_flash_attention_tq4(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fused TQ4 Flash Attention with compressed K and standard V.

    Pre-rotates Q by ``rotation^T``, then launches the fused kernel that
    decompresses nibble-packed K indices inline via centroid gather.

    Args:
        q: Query ``[batch, H_Q, seq_q, head_dim]`` fp16/bf16.
        k_packed: Nibble-packed key indices ``[batch, H_KV, seq_kv, head_dim//2]`` uint8.
        k_norms: Key norms ``[batch, H_KV, seq_kv]`` or ``[..., 1]`` fp32.
        centroids: Lloyd-Max codebook ``[16]`` fp32 (for 4-bit).
        rotation: Orthogonal rotation matrix ``[head_dim, head_dim]`` fp32.
        v: Values ``[batch, H_KV, seq_kv, head_dim]`` fp16/bf16.
        sm_scale: Softmax scale. Defaults to ``1 / sqrt(head_dim)``.
        is_causal: Apply causal masking.

    Returns:
        Attention output ``[batch, H_Q, seq_q, head_dim]``.
    """
    B, H_Q, N_Q, D = q.shape
    _, H_KV, N_KV, HALF_D = k_packed.shape

    assert HALF_D == D // 2, f"Packed dim {HALF_D} != head_dim//2 ({D // 2})"
    assert H_Q % H_KV == 0, f"Q heads ({H_Q}) must be divisible by KV heads ({H_KV})"
    assert k_packed.dtype == torch.uint8, "k_packed must be uint8"
    assert k_norms.dtype == torch.float32, "k_norms must be float32"
    assert centroids.dtype == torch.float32, "centroids must be float32"

    # Squeeze trailing 1 from norms if present
    if k_norms.dim() == 4 and k_norms.shape[-1] == 1:
        k_norms = k_norms.squeeze(-1)

    if is_causal and N_Q == 1:
        is_causal = False

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Pre-rotate Q: q_rot = q @ Pi^T
    q_rot = torch.matmul(q.float(), rotation.T).to(q.dtype)

    out = torch.empty_like(q)

    def grid(META: dict) -> tuple[int, int]:
        """Compute launch grid from autotuned block size.

        Returns:
            ``(num_q_blocks, batch * H_Q)`` grid dimensions.
        """
        return (triton.cdiv(N_Q, META["BLOCK_M"]), B * H_Q)

    _fwd_tq4_kernel[grid](
        q_rot,
        k_packed,
        k_norms,
        centroids,
        v,
        out,
        sm_scale,
        *q_rot.stride(),
        *k_packed.stride(),
        *k_norms.stride(),
        *v.stride(),
        *out.stride(),
        H_Q,
        H_KV,
        N_Q,
        N_KV,
        HEAD_DIM=D,
        IS_CAUSAL=is_causal,
    )

    return out
