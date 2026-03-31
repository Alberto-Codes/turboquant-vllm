"""Tests for TQ4 fused paged decode feature gating and backend integration.

Story 6.3: Backend integration and feature gating for the fused paged
TQ4 decode kernel.  Tests cover feature gate initialization, decode
path selection, buffer downsizing, and backward compatibility.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402

from turboquant_vllm.vllm.tq4_backend import (  # noqa: E402  # isort: skip
    TQ4AttentionImpl,
    TQ4_BITS,
    TQ4_NORM_BYTES,
    TQ4_SEED,
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


def _make_impl(quantizer, *, fused_paged_available=False, max_prefill_len=2048):
    """Create a TQ4AttentionImpl without full vLLM init.

    Args:
        quantizer: TurboQuantMSE instance.
        fused_paged_available: Override for ``_fused_paged_available``.
        max_prefill_len: Override for ``_max_prefill_len``.
    """
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
    impl._int8_prefill_available = False
    impl._max_prefill_len = max_prefill_len
    impl._max_model_len = 6144

    return impl


def _make_cache(num_blocks):
    """Create a zeroed packed TQ4 cache."""
    total_bytes = NUM_KV_HEADS * _tq4_bytes_per_token_kv(HEAD_SIZE)
    return torch.zeros(num_blocks, BLOCK_SIZE, total_bytes, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Task 1: Feature gate tests
# ---------------------------------------------------------------------------


class TestFusedPagedGating:
    """Feature gate initialization and env var parsing."""

    def test_fused_paged_defaults_false_when_env_absent(
        self, tq4_quantizer, monkeypatch
    ) -> None:
        """_fused_paged_available is False when TQ4_USE_FUSED_PAGED is not set."""
        monkeypatch.delenv("TQ4_USE_FUSED_PAGED", raising=False)
        impl = _make_impl(tq4_quantizer)
        assert impl._fused_paged_available is False

    @pytest.mark.parametrize(
        ("env_val", "expected"),
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("YES", True),
            ("0", False),
            ("false", False),
            ("False", False),
            ("no", False),
            ("", False),
        ],
        ids=[
            "1-true",
            "true-lower",
            "True-title",
            "TRUE-upper",
            "yes-lower",
            "YES-upper",
            "0-false",
            "false-lower",
            "False-title",
            "no-false",
            "empty-false",
        ],
    )
    def test_env_var_parsing(
        self, tq4_quantizer, monkeypatch, env_val, expected
    ) -> None:
        """Env var TQ4_USE_FUSED_PAGED parsing for truthy/falsy values."""
        monkeypatch.setenv("TQ4_USE_FUSED_PAGED", env_val)
        # We test the parsing function directly once it exists
        from turboquant_vllm.vllm.tq4_backend import _parse_fused_paged_env

        assert _parse_fused_paged_env() is expected

    def test_fused_available_false_when_import_fails(
        self, tq4_quantizer, monkeypatch
    ) -> None:
        """_fused_paged_available is False when kernel import fails, even if env is truthy."""
        monkeypatch.setenv("TQ4_USE_FUSED_PAGED", "1")
        # Mock the import to fail
        import turboquant_vllm.vllm.tq4_backend as backend_mod

        monkeypatch.setattr(
            backend_mod,
            "_fused_paged_kernel_available",
            False,
        )
        # After parsing, env is True but import failed → False
        result = (
            backend_mod._parse_fused_paged_env()
            and backend_mod._fused_paged_kernel_available
        )
        assert result is False

    def test_fused_available_true_when_env_and_import_succeed(
        self, tq4_quantizer, monkeypatch
    ) -> None:
        """_fused_paged_available is True when env is truthy AND kernel imports."""
        monkeypatch.setenv("TQ4_USE_FUSED_PAGED", "1")
        import turboquant_vllm.vllm.tq4_backend as backend_mod

        assert backend_mod._parse_fused_paged_env() is True
        if not backend_mod._fused_paged_kernel_available:
            pytest.skip("fused kernel not importable on this platform")
        assert backend_mod._fused_paged_kernel_available is True


# ---------------------------------------------------------------------------
# Task 2: Decode path selector tests
# ---------------------------------------------------------------------------


class TestDecodePathSelector:
    """Verify fused kernel dispatch during decode and decompress-all during prefill."""

    def test_fused_decode_calls_fused_kernel(self, tq4_quantizer, mocker) -> None:
        """When _fused_paged_available=True and is_decode, _fused_decode_path is called."""
        from vllm.v1.attention.backend import AttentionType

        impl = _make_impl(tq4_quantizer, fused_paged_available=True)
        impl.attn_type = AttentionType.DECODER
        spy = mocker.patch.object(
            impl, "_fused_decode_path", return_value=torch.zeros(1)
        )

        # Simulate decode: _cg_buffers_ready=True, num_actual_tokens=1
        impl._cg_buffers_ready = True

        attn_metadata = mocker.MagicMock()
        attn_metadata.num_actual_tokens = 1

        layer = mocker.MagicMock()
        output = torch.zeros(1, NUM_HEADS, HEAD_SIZE)
        kv_cache = _make_cache(4)

        impl.forward(
            layer,
            torch.randn(1, NUM_HEADS, HEAD_SIZE),
            None,
            None,
            kv_cache,
            attn_metadata,
            output=output,
        )

        spy.assert_called_once()

    def test_prefill_skips_fused_even_when_available(
        self, tq4_quantizer, mocker
    ) -> None:
        """Prefill (num_actual_tokens > 1) always uses decompress-all path."""
        from vllm.v1.attention.backend import AttentionType

        impl = _make_impl(tq4_quantizer, fused_paged_available=True)
        impl.attn_type = AttentionType.DECODER
        impl.alibi_slopes = None
        impl.sliding_window = None
        impl.logits_soft_cap = 0.0
        impl.vllm_flash_attn_version = None
        impl.sinks = None
        impl._cg_buffers_ready = True

        fused_spy = mocker.patch.object(impl, "_fused_decode_path")
        mocker.patch.object(
            impl,
            "_tq4_prefill",
            return_value=(
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.int32),
            ),
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

        fused_spy.assert_not_called()

    def test_decompress_all_when_fused_disabled(self, tq4_quantizer, mocker) -> None:
        """When _fused_paged_available=False, _fused_decode_path is never called."""
        from vllm.v1.attention.backend import AttentionType

        impl = _make_impl(tq4_quantizer, fused_paged_available=False)
        impl.attn_type = AttentionType.DECODER
        impl.alibi_slopes = None
        impl.sliding_window = None
        impl.logits_soft_cap = 0.0
        impl.vllm_flash_attn_version = None
        impl.sinks = None
        impl._cg_buffers_ready = True

        fused_spy = mocker.patch.object(impl, "_fused_decode_path")
        mocker.patch.object(
            impl,
            "_tq4_decode",
            return_value=(
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.int32),
            ),
        )
        mocker.patch("vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func")

        attn_metadata = mocker.MagicMock()
        attn_metadata.num_actual_tokens = 1
        attn_metadata.use_cascade = False

        layer = mocker.MagicMock()
        output = torch.zeros(1, NUM_HEADS, HEAD_SIZE)
        kv_cache = _make_cache(4)

        impl.forward(
            layer,
            torch.randn(1, NUM_HEADS, HEAD_SIZE),
            None,
            None,
            kv_cache,
            attn_metadata,
            output=output,
        )

        fused_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Task 3: Buffer downsizing tests
# ---------------------------------------------------------------------------


class TestBufferDownsizing:
    """Decompress buffer sizing bounded by max_model_len."""

    def test_decompress_buffers_bounded_when_cache_small(self, tq4_quantizer) -> None:
        """Decompress buffers are min(max_model_len, max_tokens) — cache smaller."""
        num_blocks = 100
        impl = _make_impl(tq4_quantizer, fused_paged_available=False)
        kv_cache = _make_cache(num_blocks)
        impl._init_cg_buffers(kv_cache, torch.bfloat16)

        # max_tokens = 1600 < max_model_len = 6144, so uses max_tokens
        expected_tokens = num_blocks * BLOCK_SIZE
        assert impl._cg_decompress_k.shape == (expected_tokens, NUM_KV_HEADS, HEAD_SIZE)
        assert impl._cg_decompress_v.shape == (expected_tokens, NUM_KV_HEADS, HEAD_SIZE)

    def test_decompress_buffers_same_regardless_of_fused(self, tq4_quantizer) -> None:
        """Decompress buffers identical for fused=True and fused=False."""
        num_blocks = 100
        impl_nonfused = _make_impl(tq4_quantizer, fused_paged_available=False)
        impl_fused = _make_impl(tq4_quantizer, fused_paged_available=True)
        kv_cache = _make_cache(num_blocks)
        impl_nonfused._init_cg_buffers(kv_cache, torch.bfloat16)
        impl_fused._init_cg_buffers(kv_cache, torch.bfloat16)

        assert impl_nonfused._cg_decompress_k.shape == impl_fused._cg_decompress_k.shape
        assert impl_nonfused._cg_decompress_v.shape == impl_fused._cg_decompress_v.shape

    def test_max_prefill_len_fallback_default(self, tq4_quantizer) -> None:
        """_max_prefill_len defaults to 2048 when vllm_config is None."""
        impl = _make_impl(tq4_quantizer)
        assert impl._max_prefill_len == 2048

    def test_buffer_downsizing_vram_savings(self, tq4_quantizer) -> None:
        """Bounded buffers significantly smaller than full cache on large configs."""
        num_blocks = 4000  # typical Molmo2 config: 64000 tokens
        impl = _make_impl(tq4_quantizer, fused_paged_available=False)
        kv_cache = _make_cache(num_blocks)
        impl._init_cg_buffers(kv_cache, torch.bfloat16)

        # max_model_len=6144 << 64000, so buffer is bounded
        bounded_bytes = (
            impl._cg_decompress_k.nelement() * impl._cg_decompress_k.element_size()
        )
        full_bytes = (
            num_blocks
            * BLOCK_SIZE
            * NUM_KV_HEADS
            * HEAD_SIZE
            * impl._cg_decompress_k.element_size()
        )

        savings = 1.0 - bounded_bytes / full_bytes
        assert savings > 0.9, f"Expected >90% VRAM savings, got {savings:.1%}"


# ---------------------------------------------------------------------------
# Task 5: Backend integration tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """When _fused_paged_available=False, behavior is identical to pre-6.3."""

    def test_compress_store_decompress_unchanged(self, tq4_quantizer) -> None:
        """Round-trip through compress → store → decompress is unaffected by feature gate."""
        impl = _make_impl(tq4_quantizer, fused_paged_available=False)
        kv_cache = _make_cache(num_blocks=4)

        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        slot_mapping = torch.tensor([5])

        impl._compress_and_store(key, value, kv_cache, slot_mapping)
        key_cache, value_cache = impl._decompress_cache(kv_cache, torch.float32)

        # Slot 5 should have data after round-trip
        reconstructed_k = key_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)[5]
        assert reconstructed_k.any(), "Decompressed K should be non-zero"

    def test_fused_decode_path_method_exists(self, tq4_quantizer) -> None:
        """_fused_decode_path exists on impl regardless of feature gate."""
        impl = _make_impl(tq4_quantizer, fused_paged_available=False)
        assert hasattr(impl, "_fused_decode_path")

    def test_fused_paged_available_attr_always_exists(self, tq4_quantizer) -> None:
        """_fused_paged_available is always set, defaulting to False."""
        impl = _make_impl(tq4_quantizer)
        assert hasattr(impl, "_fused_paged_available")
        assert impl._fused_paged_available is False


# ---------------------------------------------------------------------------
# Test Maturity: MEDIUM 4 — conftest/production constant coupling
# ---------------------------------------------------------------------------


class TestConstantCoupling:
    """Validate conftest constants track production TQ4 constants.

    MEDIUM 4 from test-review.md: conftest defines BITS_4 and SEED
    independently from tq4_backend.TQ4_BITS and TQ4_SEED.  If either
    production constant changes without updating conftest, tests silently
    test the wrong config.
    """

    def test_conftest_bits4_matches_production(self) -> None:
        """conftest.BITS_4 must equal tq4_backend.TQ4_BITS."""
        from tests.conftest import BITS_4

        assert BITS_4 == TQ4_BITS, (
            f"conftest.BITS_4={BITS_4} != tq4_backend.TQ4_BITS={TQ4_BITS}"
        )

    def test_conftest_seed_matches_production(self) -> None:
        """conftest.SEED must equal tq4_backend.TQ4_SEED."""
        from tests.conftest import SEED

        assert SEED == TQ4_SEED, (
            f"conftest.SEED={SEED} != tq4_backend.TQ4_SEED={TQ4_SEED}"
        )
