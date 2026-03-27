"""HuggingFace AttentionInterface registration for Triton Flash Attention.

Registers ``triton_fa`` as a custom attention backend that can be selected
when loading a model or installed post-load. The backend resolves at forward
time (not init time) via ``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]``,
so changing the config after loading takes effect immediately.

Attributes:
    triton_fa_forward: HF-compatible attention function.
    register_triton_fa: Register the backend globally.
    install_triton_fa: Register and activate on a specific model.

Examples:
    ```python
    from turboquant_consumer.triton.attention_interface import install_triton_fa

    install_triton_fa(model)
    output = model.generate(inputs)
    ```

See Also:
    :mod:`turboquant_consumer.triton.flash_attention`:
        The underlying Triton kernel.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from turboquant_consumer.triton.flash_attention import triton_flash_attention


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
