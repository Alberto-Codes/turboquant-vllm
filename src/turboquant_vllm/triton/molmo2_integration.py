"""Fused TurboQuant attention integration for Molmo2 models.

Patches Molmo2 attention layers to compute Q @ K^T directly from
nibble-packed 4-bit compressed keys using the fused Triton kernel.
Keys are never materialized as full fp16 tensors during attention.

Values are stored uncompressed in fp16 (the softmax @ V path benefits
less from compression and doesn't need a fused kernel).

Attributes:
    FusedTurboQuantRunner: High-level runner that patches a Molmo2 model,
        generates text, and cleans up.
    install_fused_attention: Low-level function to patch attention layers.

Examples:
    ```python
    runner = FusedTurboQuantRunner(model, processor, bits=4)
    text, stats = runner.generate(
        prompt="Describe this scene.",
        video_path="/path/to/video.mp4",
        max_new_tokens=256,
    )
    ```

See Also:
    :mod:`turboquant_vllm.triton.fused_qk_attention`: The Triton kernel.
    :mod:`turboquant_vllm.kv_cache`: Unfused CompressedDynamicCache.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn
from transformers import DynamicCache

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.fused_qk_attention import fused_qk_scores


class CompressedKVStore(DynamicCache):
    """KV store with compressed keys and standard values.

    Keys are compressed into nibble-packed uint8 indices + fp32 norms
    in side storage for the fused Triton kernel. Values and all
    DynamicLayer bookkeeping are managed by the base ``DynamicCache``
    via the overridden ``update()`` method.

    This cache is passed as ``past_key_values`` to ``model.generate()``.

    Attributes:
        quantizer (TurboQuantMSE): The TQ4 quantizer instance.
        rotation_T (torch.Tensor): Transposed rotation matrix for
            query pre-rotation, shape ``(head_dim, head_dim)``.
        centroids (torch.Tensor): Lloyd-Max centroid values,
            shape ``(n_levels,)``.

    Examples:
        ```python
        store = CompressedKVStore(quantizer=tq)
        store.compress_and_store_key(key_states, layer_idx=0)
        ```
    """

    def __init__(self, quantizer: TurboQuantMSE) -> None:
        """Initialize the compressed KV store.

        Args:
            quantizer: TurboQuantMSE instance for key compression.
        """
        super().__init__()
        self.quantizer = quantizer
        self.rotation_T = quantizer.rotation.T.contiguous()
        self.centroids = quantizer.codebook.centroids.contiguous()

        self._packed_indices: list[torch.Tensor | None] = []
        self._norms: list[torch.Tensor | None] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress keys on write, store values normally via DynamicCache.

        Overrides ``DynamicCache.update()`` to intercept key storage.
        Keys are nibble-packed into compressed side storage for the
        fused Triton kernel. Values and all ``DynamicLayer`` bookkeeping
        (seq_length, layer creation, offloading) are handled by the
        base class, keeping ``model.generate()`` happy.

        Args:
            key_states: Key tensor, shape
                ``(batch, n_kv_heads, seq_len, head_dim)``.
            value_states: Value tensor, same shape as key_states.
            layer_idx: Transformer layer index.
            cache_kwargs: Additional cache arguments (passed to base).

        Returns:
            Tuple of (full_keys, full_values) from the base class.
            The returned keys are uncompressed (for compatibility);
            the fused kernel reads from compressed side storage instead.
        """
        # 1. Compress keys into side storage
        indices, norms = self.quantizer.quantize(key_states.float())
        indices = indices.to(torch.uint8)
        packed = (indices[..., 0::2] << 4) | indices[..., 1::2]
        norms = norms.float().squeeze(-1)

        while len(self._packed_indices) <= layer_idx:
            self._packed_indices.append(None)
            self._norms.append(None)

        if self._packed_indices[layer_idx] is None:
            self._packed_indices[layer_idx] = packed
            self._norms[layer_idx] = norms
        else:
            self._packed_indices[layer_idx] = torch.cat(
                [self._packed_indices[layer_idx], packed], dim=2
            )
            self._norms[layer_idx] = torch.cat([self._norms[layer_idx], norms], dim=2)

        # 2. Let DynamicCache handle values + DynamicLayer bookkeeping.
        #    Keys are passed through for seq_length tracking.
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def get_compressed_key(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compressed key data for a layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (packed_indices, norms).
        """
        return self._packed_indices[layer_idx], self._norms[layer_idx]


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine component of RoPE.
        sin: Sine component of RoPE.

    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension and negate it.

        Args:
            x: Input tensor.

        Returns:
            Tensor with second half negated and swapped with first half.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA.

    Args:
        hidden_states: KV tensor, shape ``(B, n_kv_heads, S, D)``.
        n_rep: Number of times to repeat each KV head.

    Returns:
        Expanded tensor, shape ``(B, n_kv_heads * n_rep, S, D)``.
    """
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, n_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


def _make_fused_forward(
    attn_module: nn.Module,
    store: CompressedKVStore,
    layer_idx: int,
) -> Any:
    """Create a fused attention forward for one Molmo2 attention layer.

    Args:
        attn_module: The original Molmo2Attention module.
        store: The CompressedKVStore for compressed key storage.
        layer_idx: This layer's index.

    Returns:
        A replacement forward function.
    """
    # Capture from the attention module
    head_dim = int(attn_module.head_dim)  # ty: ignore[invalid-argument-type]
    n_heads = int(attn_module.num_heads)  # ty: ignore[invalid-argument-type]
    n_kv_heads = int(attn_module.num_key_value_heads)  # ty: ignore[invalid-argument-type]
    n_kv_groups = n_heads // n_kv_heads
    fused_dims = attn_module.fused_dims
    q_norm = attn_module.q_norm
    k_norm = attn_module.k_norm
    qk_norm_type = getattr(attn_module, "qk_norm_type", None)

    # Capture from the store
    rotation_T = store.rotation_T  # [head_dim, head_dim]
    centroids = store.centroids  # [n_levels]
    scale = 1.0 / math.sqrt(head_dim)

    def fused_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fused TurboQuant attention forward for Molmo2.

        Replaces the standard Q @ K^T with a Triton kernel that reads
        nibble-packed 4-bit key indices directly. Cache bookkeeping is
        handled by ``store.update()`` (DynamicCache base class).

        Args:
            hidden_states: Input tensor.
            position_embeddings: Tuple of (cos, sin) for RoPE.
            attention_mask: Causal attention mask.
            past_key_values: The CompressedKVStore (ignored, we use
                the captured ``store`` reference).
            cache_position: Position indices for static cache.

        Other Parameters:
            **kwargs: Passed through for HuggingFace forward compatibility.

        Returns:
            Tuple of (attn_output, None).
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        # QKV projection (fused linear)
        qkv = attn_module.att_proj(hidden_states)  # ty: ignore[call-non-callable]
        query_states, key_states, value_states = qkv.split(fused_dims, dim=-1)

        # QK norm (before reshape for non-qwen3, after for qwen3)
        if q_norm is not None and k_norm is not None and qk_norm_type != "qwen3":
            query_states = q_norm(query_states)  # ty: ignore[call-non-callable]
            key_states = k_norm(key_states)  # ty: ignore[call-non-callable]

        # Reshape to [batch, seq, heads, head_dim]
        value_states = value_states.view(hidden_shape)
        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)

        if q_norm is not None and k_norm is not None and qk_norm_type == "qwen3":
            query_states = q_norm(query_states)  # ty: ignore[call-non-callable]
            key_states = k_norm(key_states)  # ty: ignore[call-non-callable]

        # Transpose to [batch, heads, seq, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # --- Cache update: compress keys + store values via DynamicCache ---
        # store.update() overrides DynamicCache.update() to compress keys
        # into side storage while letting the base class handle values
        # and all DynamicLayer bookkeeping (seq_length, layer creation).
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        _, full_values = store.update(key_states, value_states, layer_idx, cache_kwargs)

        # --- Fused attention scores ---
        # Pre-rotate query: q_rot = Q @ rotation_T
        # Use model dtype (bf16) for pre-rotation to match the precision
        # behavior of the unfused bf16 attention path. fp32 pre-rotation
        # produces scores that compound differently across 36 layers.
        device = query_states.device
        q_rot = (
            query_states
            @ rotation_T.to(device=device, dtype=query_states.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        ).float()

        # Get compressed keys for this layer
        packed_indices, norms = store.get_compressed_key(layer_idx)

        # Fused Triton kernel: Q_rot @ compressed_K^T
        attn_weights = fused_qk_scores(
            q_rot,
            packed_indices,
            norms,
            centroids.to(device),
            scale,
            n_q_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )

        # Cast scores to model dtype before mask + softmax to match
        # the precision behavior of the original bf16 attention path.
        # Without this, fp32 kernel scores compound differently through
        # softmax across 36 layers, causing output degradation.
        attn_weights = attn_weights.to(query_states.dtype)

        # Apply causal mask
        if attention_mask is not None:
            kv_len = packed_indices.shape[2]
            q_len = query_states.shape[2]
            causal_mask = attention_mask[:, :, :q_len, :kv_len]
            attn_weights = attn_weights + causal_mask

        # Softmax (compute in fp32 for numerical stability, cast back)
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Values: GQA expand and matmul
        full_values_expanded = _repeat_kv(full_values, n_kv_groups)
        attn_output = torch.matmul(attn_weights, full_values_expanded)

        # Transpose heads back: [B, heads, S, D] → [B, S, heads, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn_module.attn_out(attn_output)  # ty: ignore[call-non-callable]

        return attn_output, None

    return fused_forward


def install_fused_attention(
    model: nn.Module,
    bits: int = 4,
    *,
    seed: int = 42,
) -> CompressedKVStore:
    """Patch all Molmo2 text attention layers to use fused TurboQuant.

    Args:
        model: A loaded Molmo2 model.
        bits: Quantization bits per coordinate (default 4 for nibble packing).
        seed: Random seed for reproducibility.

    Returns:
        A CompressedKVStore to pass as ``past_key_values`` to
        ``model.generate()``.
    """
    # Detect head_dim from model config
    config = model.config
    text_config = getattr(config, "text_config", config)
    head_dim = getattr(text_config, "head_dim", 128)

    # Create quantizer and store
    tq = TurboQuantMSE(head_dim, bits, seed=seed)
    kv_store = CompressedKVStore(tq)

    # Find and patch text attention layers
    patched = 0
    for name, module in model.named_modules():
        if hasattr(module, "att_proj") and hasattr(module, "attn_out"):
            layer_idx = getattr(module, "layer_idx", patched)
            module._original_forward = module.forward
            module.forward = _make_fused_forward(module, kv_store, layer_idx)
            patched += 1

    print(f"  Installed fused TurboQuant TQ{bits} on {patched} attention layers")
    return kv_store


def uninstall_fused_attention(model: nn.Module) -> None:
    """Restore original attention forwards.

    Args:
        model: The patched Molmo2 model.
    """
    restored = 0
    for _name, module in model.named_modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
            restored += 1
    if restored:
        print(f"  Restored {restored} original attention forwards")


class FusedTurboQuantRunner:
    """High-level runner for fused TurboQuant inference on Molmo2.

    Patches the model, runs inference, and cleans up. Handles both
    text-only and video inputs.

    Attributes:
        model (nn.Module): The Molmo2 model.
        processor (Any): The Molmo2 processor.
        bits (int): Quantization bit width.

    Examples:
        ```python
        runner = FusedTurboQuantRunner(model, processor, bits=4)
        text, stats = runner.generate("Describe this.", max_new_tokens=256)
        print(text)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        bits: int = 4,
        *,
        seed: int = 42,
    ) -> None:
        """Initialize the runner.

        Args:
            model: A loaded Molmo2 model.
            processor: The corresponding Molmo2 processor.
            bits: Quantization bits (default 4 for nibble packing).
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.processor = processor
        self.bits = bits
        self.seed = seed

    def generate(
        self,
        prompt: str,
        video_path: str | None = None,
        max_new_tokens: int = 256,
    ) -> tuple[str, dict]:
        """Generate text with fused TurboQuant attention.

        Args:
            prompt: Text prompt.
            video_path: Optional path to a video file.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (generated_text, stats_dict).
        """
        import time

        # Build input
        content: list[dict] = []
        if video_path:
            content.append({"type": "video", "video": video_path})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        input_len = inputs["input_ids"].shape[-1]

        # Install fused attention
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        store = install_fused_attention(self.model, self.bits, seed=self.seed)

        # Generate
        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(  # ty: ignore[call-non-callable]
                **inputs,
                past_key_values=store,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.perf_counter() - t0
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Decode
        generated_ids = output_ids[0, input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)

        output_len = len(generated_ids)
        tok_per_sec = output_len / elapsed if elapsed > 0 else 0

        # Uninstall
        uninstall_fused_attention(self.model)

        stats = {
            "input_tokens": input_len,
            "output_tokens": output_len,
            "tokens_per_sec": round(tok_per_sec, 1),
            "elapsed_s": round(elapsed, 2),
            "vram_peak_mib": round(vram_peak, 1),
            "bits": self.bits,
            "kv_seq_len": store.get_seq_length(),
        }

        return text, stats
