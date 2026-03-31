"""Fused TQ4 Flash Attention -- both K and V decompressed in the inner loop.

Phase 3 of the P5 roadmap. Both K and V tiles are decompressed inline
from nibble-packed uint8 indices. The query is pre-rotated by ``Pi^T``
and the output is post-rotated by ``Pi`` outside the kernel (since K and
V share the same rotation matrix). Non-power-of-two HEAD_DIM (e.g., 96)
is supported via padded tl.arange + masking. Autotune configs include
BLOCK_M values 16, 32, 64, 128 to cover head_dim up to 256.

Attributes:
    triton_flash_attention_tq4_kv: Python wrapper that pre-rotates Q,
        launches the fused kernel, and post-rotates the output.

Examples:
    ```python
    from turboquant_vllm.triton.flash_attention_tq4_kv import (
        triton_flash_attention_tq4_kv,
    )

    out = triton_flash_attention_tq4_kv(
        q,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        centroids,
        rotation,
        sm_scale=None,
    )
    ```

See Also:
    :mod:`turboquant_vllm.triton.flash_attention_tq4`: Phase 2 (K-only).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    """Round up to the nearest power of two (for Triton tl.arange).

    Args:
        n: The number to round up.

    Returns:
        The smallest power of two >= ``n``.
    """
    return 1 << (n - 1).bit_length() if n > 0 else 1


# ---------------------------------------------------------------------------
# Autotune
# ---------------------------------------------------------------------------

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [16, 32, 64, 128]
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
def _fwd_tq4_kv_kernel(
    Q_rot,
    K_packed,
    K_norms,
    V_packed,
    V_norms,
    Centroids,
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
    stride_vpz,
    stride_vph,
    stride_vpn,
    stride_vpd,
    stride_vnz,
    stride_vnh,
    stride_vnn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    H_Q,
    H_KV,
    N_CTX_Q,
    N_CTX_KV,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    HALF_D_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Fused TQ4 K+V Flash Attention kernel.

    Both K and V tiles decompressed inline from nibble-packed indices.
    Output is in rotated space (caller applies post-rotation ``@ Pi``).
    Autotuned on ``(N_CTX_Q, HEAD_DIM)``.

    Supports non-power-of-two head dimensions (e.g. 96) by padding
    ``tl.arange`` to the next power of two and masking out-of-bounds lanes.
    """
    HALF_D: tl.constexpr = HEAD_DIM // 2

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h_q = off_hz % H_Q
    off_h_kv = off_h_q // (H_Q // H_KV)

    # Base pointers
    q_base = Q_rot + off_z * stride_qz + off_h_q * stride_qh
    kp_base = K_packed + off_z * stride_kpz + off_h_kv * stride_kph
    kn_base = K_norms + off_z * stride_knz + off_h_kv * stride_knh
    vp_base = V_packed + off_z * stride_vpz + off_h_kv * stride_vph
    vn_base = V_norms + off_z * stride_vnz + off_h_kv * stride_vnh
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh

    # Block offsets (pad HEAD_DIM/HALF_D for non-power-of-two dims)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    d_mask = offs_d < HEAD_DIM
    offs_d_half = tl.arange(0, HALF_D_PAD)
    d_half_mask = offs_d_half < HALF_D

    # Load Q_rot tile
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX_Q) & d_mask[None, :], other=0.0)

    # fp32 online softmax state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PAD], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX_KV)
    else:
        hi = N_CTX_KV

    # === Main tile loop ===
    for start_n in range(0, hi, BLOCK_N):
        kv_valid = (start_n + offs_n) < N_CTX_KV

        # -- K decompression (same as Phase 2) --
        kp_ptrs = (
            kp_base
            + (start_n + offs_n[:, None]) * stride_kpn
            + offs_d_half[None, :] * stride_kpd
        )
        k_packed = tl.load(
            kp_ptrs, mask=kv_valid[:, None] & d_half_mask[None, :], other=0
        )
        k_hi = (k_packed >> 4).to(tl.int32)
        k_lo = (k_packed & 0x0F).to(tl.int32)
        k = tl.join(tl.load(Centroids + k_hi), tl.load(Centroids + k_lo)).reshape(
            BLOCK_N, HEAD_DIM_PAD
        )
        # Zero padded lanes so they don't contribute to dot products
        k = tl.where(d_mask[None, :], k, 0.0)
        kn_ptrs = kn_base + (start_n + offs_n) * stride_knn
        k_norms = tl.load(kn_ptrs, mask=kv_valid, other=0.0)
        k = (k * k_norms[:, None]).to(Q_rot.dtype.element_ty)

        # Q_rot @ K^T
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale

        if IS_CAUSAL:
            causal = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal, qk, float("-inf"))
        qk = tl.where(kv_valid[None, :], qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        acc = acc * alpha[:, None]

        # -- V decompression (Phase 3: same pattern as K) --
        vp_ptrs = (
            vp_base
            + (start_n + offs_n[:, None]) * stride_vpn
            + offs_d_half[None, :] * stride_vpd
        )
        v_packed = tl.load(
            vp_ptrs, mask=kv_valid[:, None] & d_half_mask[None, :], other=0
        )
        v_hi = (v_packed >> 4).to(tl.int32)
        v_lo = (v_packed & 0x0F).to(tl.int32)
        v = tl.join(tl.load(Centroids + v_hi), tl.load(Centroids + v_lo)).reshape(
            BLOCK_N, HEAD_DIM_PAD
        )
        # Zero padded lanes so they don't contribute to dot products
        v = tl.where(d_mask[None, :], v, 0.0)
        vn_ptrs = vn_base + (start_n + offs_n) * stride_vnn
        v_norms = tl.load(vn_ptrs, mask=kv_valid, other=0.0)
        v = (v * v_norms[:, None]).to(Q_rot.dtype.element_ty)

        # P @ V
        l_ij = tl.sum(p, 1)
        p_cast = p.to(v.dtype)
        acc = tl.dot(p_cast, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_new

    # Epilogue (output is in rotated space -- caller post-rotates)
    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(
        o_ptrs,
        acc.to(Q_rot.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX_Q) & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def triton_flash_attention_tq4_kv(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    sm_scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fused TQ4 Flash Attention with both K and V compressed.

    Pre-rotates Q by ``rotation^T``, launches the kernel that decompresses
    both K and V inline, then post-rotates the output by ``rotation`` to
    return to the original coordinate space. Non-power-of-two head
    dimensions (e.g., 96) are handled via padded tile loads and boundary
    masking inside the kernel.  Non-pow2 dims incur ~5-15 % throughput
    penalty due to wasted lanes in padded tiles (e.g., head_dim=96 pads
    to 128, wasting 25 % of memory bandwidth on K/V loads).

    Args:
        q: Query ``[batch, H_Q, seq_q, head_dim]`` fp16/bf16.
        k_packed: Nibble-packed key indices ``[batch, H_KV, seq_kv, D//2]`` uint8.
        k_norms: Key norms ``[batch, H_KV, seq_kv]`` or ``[..., 1]`` fp32.
        v_packed: Nibble-packed value indices ``[batch, H_KV, seq_kv, D//2]`` uint8.
        v_norms: Value norms ``[batch, H_KV, seq_kv]`` or ``[..., 1]`` fp32.
        centroids: Shared Lloyd-Max codebook ``[16]`` fp32.
        rotation: Shared orthogonal rotation ``[head_dim, head_dim]`` fp32.
        sm_scale: Softmax scale. Defaults to ``1 / sqrt(head_dim)``.
        is_causal: Apply causal masking.

    Returns:
        Attention output ``[batch, H_Q, seq_q, head_dim]`` in original space.
    """
    B, H_Q, N_Q, D = q.shape
    _, H_KV, N_KV, HALF_D = k_packed.shape

    assert D % 2 == 0, f"HEAD_DIM must be even, got {D}"
    assert HALF_D == D // 2
    assert H_Q % H_KV == 0
    assert k_packed.dtype == torch.uint8
    assert v_packed.dtype == torch.uint8
    assert k_norms.dtype == torch.float32
    assert v_norms.dtype == torch.float32

    # Squeeze trailing 1 from norms
    if k_norms.dim() == 4 and k_norms.shape[-1] == 1:
        k_norms = k_norms.squeeze(-1)
    if v_norms.dim() == 4 and v_norms.shape[-1] == 1:
        v_norms = v_norms.squeeze(-1)

    if is_causal and N_Q == 1:
        is_causal = False

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Pre-rotate Q
    q_rot = torch.matmul(q.float(), rotation.T).to(q.dtype)

    out_rot = torch.empty_like(q)

    def grid(META: dict) -> tuple[int, int]:
        """Compute launch grid.

        Returns:
            ``(num_q_blocks, batch * H_Q)`` grid dimensions.
        """
        return (triton.cdiv(N_Q, META["BLOCK_M"]), B * H_Q)

    HEAD_DIM_PAD = _next_pow2(D)
    HALF_D_PAD = _next_pow2(D // 2)
    assert HALF_D_PAD * 2 == HEAD_DIM_PAD, (
        f"Padding invariant violated: 2*HALF_D_PAD ({2 * HALF_D_PAD}) "
        f"!= HEAD_DIM_PAD ({HEAD_DIM_PAD}) — tl.join reshape requires this"
    )

    _fwd_tq4_kv_kernel[grid](
        q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        centroids,
        out_rot,
        sm_scale,
        *q_rot.stride(),
        *k_packed.stride(),
        *k_norms.stride(),
        *v_packed.stride(),
        *v_norms.stride(),
        *out_rot.stride(),
        H_Q,
        H_KV,
        N_Q,
        N_KV,
        HEAD_DIM=D,
        HEAD_DIM_PAD=HEAD_DIM_PAD,
        HALF_D_PAD=HALF_D_PAD,
        IS_CAUSAL=is_causal,
    )

    # Post-rotate: convert from rotated space back to original space
    return torch.matmul(out_rot.float(), rotation).to(q.dtype)
