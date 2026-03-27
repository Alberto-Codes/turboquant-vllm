"""HuggingFace AttentionInterface registration for Triton Flash Attention.

Registers attention backends that resolve at forward time via
``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]``.

Two backends:

- ``triton_fa``: Phase 1 vanilla kernel (standard fp16 K/V).
- ``triton_fa_tq4_kv``: Phase 3 fused TQ4 kernel (compressed K+V read
  from ``CompressedDynamicCache`` via side-channel cache reference).

Attributes:
    triton_fa_forward: Phase 1 vanilla attention function.
    triton_fa_tq4_kv_forward: Phase 3 fused TQ4 K+V attention function.
    register_triton_fa: Register vanilla backend.
    install_triton_fa: Activate vanilla backend on a model.
    install_fused_tq4_kv: Activate fused TQ4 K+V backend with cache stash.

Examples:
    ```python
    from turboquant_vllm.triton.attention_interface import install_fused_tq4_kv

    install_fused_tq4_kv(model, compressed_cache)
    output = model.generate(inputs)
    ```

See Also:
    :mod:`turboquant_vllm.triton.flash_attention`: Phase 1 kernel.
    :mod:`turboquant_vllm.triton.flash_attention_tq4_kv`: Phase 3 kernel.
    :mod:`turboquant_vllm.kv_cache`: CompressedDynamicCache storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from turboquant_vllm.triton.flash_attention import triton_flash_attention
from turboquant_vllm.triton.flash_attention_tq4_kv import (
    triton_flash_attention_tq4_kv,
)

if TYPE_CHECKING:
    from turboquant_vllm.kv_cache import CompressedDynamicCache


def triton_fa_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """HF-compatible attention forward using Triton Flash Attention.

    Signature matches ``transformers.integrations.sdpa_attention.sdpa_attention_forward``.
    Handles GQA natively (no KV repeat expansion needed).

    Args:
        module: The attention layer module. Used to read ``is_causal`` attribute.
        query: ``[batch, num_q_heads, seq_q, head_dim]``.
        key: ``[batch, num_kv_heads, seq_kv, head_dim]``.
        value: ``[batch, num_kv_heads, seq_kv, head_dim]``.
        attention_mask: Optional additive mask ``[batch, 1|heads, seq_q, seq_kv]``.
        dropout: Dropout rate (must be 0 -- inference only).
        scaling: Softmax scale. Defaults to ``1 / sqrt(head_dim)``.

    Other Parameters:
        is_causal (bool | None): Override causal mode. If ``None``, inferred
            from ``query.shape[2]`` and ``module.is_causal``.
        **kwargs: Additional model-specific arguments (ignored).

    Returns:
        ``(output, None)`` where output is ``[batch, seq_q, num_q_heads, head_dim]``
        (transposed to match HF convention).
    """
    # Determine causal mode (same logic as HF SDPA backend)
    is_causal_raw = kwargs.pop("is_causal", None)
    is_causal_flag: bool
    if is_causal_raw is None:
        is_causal_flag = bool(
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )
    else:
        is_causal_flag = bool(is_causal_raw)

    out = triton_flash_attention(
        query,
        key,
        value,
        sm_scale=scaling,
        is_causal=is_causal_flag,
        attention_mask=attention_mask if not is_causal_flag else None,
    )

    # Transpose to [batch, seq, heads, head_dim] per HF convention
    out = out.transpose(1, 2).contiguous()
    return out, None


def register_triton_fa() -> None:
    """Register ``triton_fa`` as a global attention backend in HuggingFace.

    Safe to call multiple times -- overwrites the previous registration.
    """
    ALL_ATTENTION_FUNCTIONS.register("triton_fa", triton_fa_forward)


def install_triton_fa(model: torch.nn.Module) -> None:
    """Register the backend and activate it on *model*.

    Changes ``model.config._attn_implementation`` to ``"triton_fa"``.
    The model resolves the attention function at forward time, so this
    takes effect on the next forward call.

    Args:
        model: A HuggingFace model with a ``config`` attribute.

    Raises:
        AttributeError: If *model* has no ``config`` attribute.
    """
    register_triton_fa()
    config = getattr(model, "config", None)
    if config is None:
        msg = "Model has no config attribute"
        raise AttributeError(msg)
    config._attn_implementation = "triton_fa"


# ---------------------------------------------------------------------------
# Phase 3: Fused TQ4 K+V attention (Approach B — cache side-channel)
# ---------------------------------------------------------------------------


def triton_fa_tq4_kv_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Fused TQ4 K+V attention via cache side-channel.

    Reads compressed K/V from the ``CompressedDynamicCache`` stashed on
    ``module._tq4_cache`` (ignoring the decompressed *key*/*value* args).
    Falls back to vanilla Triton FA if no cache reference is found.

    Args:
        module: Attention layer with ``layer_idx`` and ``_tq4_cache`` attrs.
        query: ``[batch, H_Q, seq_q, head_dim]`` (RoPE already applied).
        key: ``[batch, H_KV, seq_kv, head_dim]`` — ignored when fused.
        value: ``[batch, H_KV, seq_kv, head_dim]`` — ignored when fused.
        attention_mask: Optional additive mask.
        dropout: Must be 0 (inference only).
        scaling: Softmax scale.

    Other Parameters:
        is_causal (bool | None): Override causal mode.
        **kwargs: Additional model-specific arguments (ignored).

    Returns:
        ``(output, None)`` with output ``[batch, seq_q, H_Q, head_dim]``.
    """
    cache: CompressedDynamicCache | None = getattr(module, "_tq4_cache", None)
    layer_idx: int | None = getattr(module, "layer_idx", None)

    # Fallback: no cache stash → use vanilla Triton FA with decompressed K/V
    if cache is None or layer_idx is None:
        return triton_fa_forward(
            module, query, key, value, attention_mask, dropout, scaling, **kwargs
        )

    # Read compressed K/V from cache side-channel
    k_packed, k_norms, v_packed, v_norms = cache.get_compressed(layer_idx)
    rotation = cache.rotation.to(device=query.device)
    centroids = cache.centroids.to(device=query.device)

    # Determine causal mode
    is_causal_raw = kwargs.pop("is_causal", None)
    is_causal_flag: bool
    if is_causal_raw is None:
        is_causal_flag = bool(
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )
    else:
        is_causal_flag = bool(is_causal_raw)

    out = triton_flash_attention_tq4_kv(
        query,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        centroids,
        rotation,
        sm_scale=scaling,
        is_causal=is_causal_flag,
    )

    out = out.transpose(1, 2).contiguous()
    return out, None


def install_fused_tq4_kv(model: torch.nn.Module, cache: CompressedDynamicCache) -> None:
    """Activate fused TQ4 K+V attention on *model* with cache side-channel.

    Registers the ``triton_fa_tq4_kv`` backend, stashes *cache* on each
    attention layer as ``module._tq4_cache``, sets the model's
    ``_attn_implementation``, and enables ``fused_mode`` on the cache
    to skip wasted decompression (P5b optimization).

    Args:
        model: HuggingFace model with attention layers that have ``layer_idx``.
        cache: CompressedDynamicCache instance that stores compressed K/V.

    Raises:
        AttributeError: If *model* has no ``config`` attribute.
    """
    ALL_ATTENTION_FUNCTIONS.register("triton_fa_tq4_kv", triton_fa_tq4_kv_forward)

    config = getattr(model, "config", None)
    if config is None:
        msg = "Model has no config attribute"
        raise AttributeError(msg)
    config._attn_implementation = "triton_fa_tq4_kv"

    # Enable fused mode: skip decompression in cache.update()
    cache.fused_mode = True

    # Stash cache reference on each attention layer
    for module in model.modules():
        if hasattr(module, "layer_idx"):
            object.__setattr__(module, "_tq4_cache", cache)


def uninstall_fused_tq4_kv(model: torch.nn.Module) -> None:
    """Remove fused TQ4 attention and restore SDPA.

    Removes ``_tq4_cache`` from attention layers, disables ``fused_mode``
    on the cache, and resets ``_attn_implementation`` to ``"sdpa"``.

    Args:
        model: Model previously configured with ``install_fused_tq4_kv``.
    """
    config = getattr(model, "config", None)
    if config is not None:
        config._attn_implementation = "sdpa"

    for module in model.modules():
        if hasattr(module, "_tq4_cache"):
            cache = getattr(module, "_tq4_cache", None)
            if cache is not None and hasattr(cache, "fused_mode"):
                cache.fused_mode = False  # type: ignore[union-attr]
            if hasattr(module, "_tq4_cache"):
                delattr(module, "_tq4_cache")
