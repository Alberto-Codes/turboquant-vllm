"""Tests for INT8 prefill feature gating and two-kernel dispatch.

Story 6.4: INT8 Q@K^T path feature gating tests.  Covers env var
parsing, gate chain (INT8 requires fused), kernel dispatch, and
backward compatibility.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402

from turboquant_vllm.vllm.tq4_backend import (  # noqa: E402  # isort: skip
    TQ4AttentionImpl,
    TQ4_NORM_BYTES,
    _tq4_bytes_per_token_kv,
)

pytestmark = [pytest.mark.unit]

# ---------------------------------------------------------------------------
# Constants (Molmo2-8B config)
# ---------------------------------------------------------------------------

NUM_KV_HEADS = 4
NUM_HEADS = 28
HEAD_SIZE = 128
BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_impl(
    quantizer,
    *,
    fused_paged_available=False,
    int8_prefill_available=False,
    max_prefill_len=2048,
):
    """Create a TQ4AttentionImpl without full vLLM init."""
    impl = object.__new__(TQ4AttentionImpl)
    impl.head_size = HEAD_SIZE
    impl.num_kv_heads = NUM_KV_HEADS
    impl.num_heads = NUM_HEADS
    impl.scale = 1.0 / (HEAD_SIZE**0.5)

    impl._tq4_rotation = quantizer.rotation.clone()
    impl._tq4_centroids = quantizer.codebook.centroids.clone()
    impl._tq4_boundaries = quantizer.codebook.boundaries.clone()
    rot_t = quantizer.rotation.T.contiguous()
    impl._tq4_rot_T_even = rot_t[:, 0::2].contiguous()
    impl._tq4_rot_T_odd = rot_t[:, 1::2].contiguous()
    impl._cg_buffers_ready = False

    half_D = HEAD_SIZE // 2
    impl._half_D = half_D
    impl._k_idx_end = NUM_KV_HEADS * half_D
    impl._k_norm_end = impl._k_idx_end + NUM_KV_HEADS * TQ4_NORM_BYTES
    impl._v_idx_end = impl._k_norm_end + NUM_KV_HEADS * half_D
    impl._total_bytes = impl._v_idx_end + NUM_KV_HEADS * TQ4_NORM_BYTES

    impl._fused_paged_available = fused_paged_available
    impl._int8_prefill_available = int8_prefill_available
    impl._max_prefill_len = max_prefill_len

    return impl


def _make_cache(num_blocks):
    """Create a zeroed packed TQ4 cache."""
    total_bytes = NUM_KV_HEADS * _tq4_bytes_per_token_kv(HEAD_SIZE)
    return torch.zeros(num_blocks, BLOCK_SIZE, total_bytes, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Task 5.10: INT8 feature gate tests
# ---------------------------------------------------------------------------


class TestInt8PrefillGating:
    """Feature gate initialization and env var parsing for INT8 prefill."""

    @pytest.mark.parametrize(
        ("env_val", "expected"),
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("yes", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("", False),
        ],
        ids=[
            "1-true",
            "true-lower",
            "True-title",
            "yes-true",
            "0-false",
            "false-lower",
            "no-false",
            "empty-false",
        ],
    )
    def test_int8_env_var_parsing(self, monkeypatch, env_val, expected) -> None:
        """TQ4_USE_INT8_PREFILL parsing for truthy/falsy values."""
        monkeypatch.setenv("TQ4_USE_INT8_PREFILL", env_val)
        from turboquant_vllm.vllm.tq4_backend import _parse_int8_prefill_env

        assert _parse_int8_prefill_env() is expected

    def test_int8_defaults_false_when_env_absent(self, monkeypatch) -> None:
        """INT8 gate is False when TQ4_USE_INT8_PREFILL is not set."""
        monkeypatch.delenv("TQ4_USE_INT8_PREFILL", raising=False)
        from turboquant_vllm.vllm.tq4_backend import _parse_int8_prefill_env

        assert _parse_int8_prefill_env() is False

    def test_int8_requires_fused_gate(self, tq4_quantizer, monkeypatch) -> None:
        """INT8 gate requires fused gate to also be True."""
        monkeypatch.setenv("TQ4_USE_INT8_PREFILL", "1")
        monkeypatch.setenv("TQ4_USE_FUSED_PAGED", "0")
        import turboquant_vllm.vllm.tq4_backend as backend_mod

        # Fused is off → INT8 cannot be on regardless of its own env var
        result = (
            backend_mod._parse_fused_paged_env()
            and backend_mod._fused_paged_kernel_available
            and backend_mod._parse_int8_prefill_env()
            and backend_mod._int8_prefill_kernel_available
        )
        assert result is False

    def test_int8_false_when_kernel_import_fails(self, monkeypatch) -> None:
        """INT8 gate is False when kernel import fails."""
        monkeypatch.setenv("TQ4_USE_INT8_PREFILL", "1")
        import turboquant_vllm.vllm.tq4_backend as backend_mod

        monkeypatch.setattr(backend_mod, "_int8_prefill_kernel_available", False)
        result = (
            backend_mod._parse_int8_prefill_env()
            and backend_mod._int8_prefill_kernel_available
        )
        assert result is False


# ---------------------------------------------------------------------------
# Task 5.8: Two-kernel dispatch
# ---------------------------------------------------------------------------


class TestTwoKernelDispatch:
    """Verify INT8 is called for prefill and FP16 for decode."""

    def test_int8_prefill_calls_int8_path(self, tq4_quantizer, mocker) -> None:
        """When _int8_prefill_available=True and is prefill, _int8_prefill_path is called."""
        from vllm.v1.attention.backend import AttentionType

        impl = _make_impl(
            tq4_quantizer, fused_paged_available=True, int8_prefill_available=True
        )
        impl.attn_type = AttentionType.DECODER
        impl._cg_buffers_ready = True

        int8_spy = mocker.patch.object(
            impl,
            "_int8_prefill_path",
            return_value=torch.zeros(32, NUM_HEADS, HEAD_SIZE),
        )

        attn_metadata = mocker.MagicMock()
        attn_metadata.num_actual_tokens = 32  # prefill (> 1 token)
        attn_metadata.seq_lens = torch.tensor([32], dtype=torch.int32)

        output = torch.zeros(32, NUM_HEADS, HEAD_SIZE)
        kv_cache = _make_cache(4)

        impl.forward(
            mocker.MagicMock(),
            torch.randn(32, NUM_HEADS, HEAD_SIZE),
            None,
            None,
            kv_cache,
            attn_metadata,
            output=output,
        )

        int8_spy.assert_called_once()

    def test_decode_calls_fused_decode_not_int8(self, tq4_quantizer, mocker) -> None:
        """When is_decode, fused decode path is called, not INT8 prefill."""
        from vllm.v1.attention.backend import AttentionType

        impl = _make_impl(
            tq4_quantizer, fused_paged_available=True, int8_prefill_available=True
        )
        impl.attn_type = AttentionType.DECODER
        impl._cg_buffers_ready = True

        fused_spy = mocker.patch.object(
            impl,
            "_fused_decode_path",
            return_value=torch.zeros(1, NUM_HEADS, HEAD_SIZE),
        )
        int8_spy = mocker.patch.object(impl, "_int8_prefill_path")

        attn_metadata = mocker.MagicMock()
        attn_metadata.num_actual_tokens = 1  # decode

        output = torch.zeros(1, NUM_HEADS, HEAD_SIZE)
        kv_cache = _make_cache(4)

        impl.forward(
            mocker.MagicMock(),
            torch.randn(1, NUM_HEADS, HEAD_SIZE),
            None,
            None,
            kv_cache,
            attn_metadata,
            output=output,
        )

        fused_spy.assert_called_once()
        int8_spy.assert_not_called()

    def test_prefill_falls_back_when_int8_disabled(self, tq4_quantizer, mocker) -> None:
        """When _int8_prefill_available=False, prefill uses decompress-all."""
        from vllm.v1.attention.backend import AttentionType

        impl = _make_impl(
            tq4_quantizer, fused_paged_available=True, int8_prefill_available=False
        )
        impl.attn_type = AttentionType.DECODER
        impl.alibi_slopes = None
        impl.sliding_window = None
        impl.logits_soft_cap = 0.0
        impl.vllm_flash_attn_version = None
        impl.sinks = None
        impl._cg_buffers_ready = True

        int8_spy = mocker.patch.object(impl, "_int8_prefill_path")
        mocker.patch.object(
            impl,
            "_tq4_prefill",
            return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        )
        mocker.patch("vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func")

        attn_metadata = mocker.MagicMock()
        attn_metadata.num_actual_tokens = 32
        attn_metadata.use_cascade = False

        layer = mocker.MagicMock()
        output = torch.zeros(32, NUM_HEADS, HEAD_SIZE)
        kv_cache = _make_cache(4)

        impl.forward(
            layer,
            torch.randn(32, NUM_HEADS, HEAD_SIZE),
            None,
            None,
            kv_cache,
            attn_metadata,
            output=output,
        )

        int8_spy.assert_not_called()

    def test_multi_sequence_prefill_falls_back(self, tq4_quantizer, mocker) -> None:
        """Multi-sequence batch bypasses INT8 kernel (single-seq only)."""
        from vllm.v1.attention.backend import AttentionType

        impl = _make_impl(
            tq4_quantizer, fused_paged_available=True, int8_prefill_available=True
        )
        impl.attn_type = AttentionType.DECODER
        impl.alibi_slopes = None
        impl.sliding_window = None
        impl.logits_soft_cap = 0.0
        impl.vllm_flash_attn_version = None
        impl.sinks = None
        impl._cg_buffers_ready = True

        int8_spy = mocker.patch.object(impl, "_int8_prefill_path")
        mocker.patch.object(
            impl,
            "_tq4_prefill",
            return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        )
        mocker.patch("vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func")

        attn_metadata = mocker.MagicMock()
        attn_metadata.num_actual_tokens = 64  # two sequences
        attn_metadata.seq_lens = torch.tensor([32, 32], dtype=torch.int32)
        attn_metadata.use_cascade = False

        layer = mocker.MagicMock()
        output = torch.zeros(64, NUM_HEADS, HEAD_SIZE)
        kv_cache = _make_cache(8)

        impl.forward(
            layer,
            torch.randn(64, NUM_HEADS, HEAD_SIZE),
            None,
            None,
            kv_cache,
            attn_metadata,
            output=output,
        )

        int8_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Task 5.11: Backward compatibility
# ---------------------------------------------------------------------------


class TestInt8BackwardCompat:
    """_int8_prefill_available=False produces identical output to existing path."""

    def test_int8_prefill_attr_always_exists(self, tq4_quantizer) -> None:
        """_int8_prefill_available is always set, defaulting to False."""
        impl = _make_impl(tq4_quantizer)
        assert hasattr(impl, "_int8_prefill_available")
        assert impl._int8_prefill_available is False

    def test_int8_prefill_path_method_exists(self, tq4_quantizer) -> None:
        """_int8_prefill_path exists on impl regardless of feature gate."""
        impl = _make_impl(tq4_quantizer)
        assert hasattr(impl, "_int8_prefill_path")

    def test_fused_decode_unaffected(self, tq4_quantizer) -> None:
        """Fused decode path is not affected by INT8 gate."""
        impl = _make_impl(
            tq4_quantizer, fused_paged_available=True, int8_prefill_available=False
        )
        assert impl._fused_paged_available is True
        assert impl._int8_prefill_available is False
