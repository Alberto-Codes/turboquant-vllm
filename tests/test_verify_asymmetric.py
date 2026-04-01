"""Unit tests for asymmetric (--k-bits/--v-bits) CLI argument parsing."""

from __future__ import annotations

import pytest

from turboquant_vllm.verify import COMPRESSION_QUALITY_THRESHOLD, main

from .test_verify import _make_result

pytestmark = [pytest.mark.unit]


class TestVerifyAsymmetricArgparse:
    """Validate --k-bits/--v-bits CLI argument parsing."""

    def test_k_bits_v_bits_parsed(self, mocker) -> None:
        """--k-bits and --v-bits should be passed to _run_verification."""
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(k_bits=4, v_bits=3),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--k-bits", "4", "--v-bits", "3"])
        assert exc_info.value.code == 0
        spy.assert_called_once_with(
            "test/m", 4, COMPRESSION_QUALITY_THRESHOLD, k_bits=4, v_bits=3
        )

    def test_bits_with_k_bits_errors(self, mocker) -> None:
        """--bits and --k-bits together should error (ambiguous)."""
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4", "--k-bits", "3"])
        assert exc_info.value.code != 0

    def test_bits_with_v_bits_errors(self, mocker) -> None:
        """--bits and --v-bits together should error (ambiguous)."""
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4", "--v-bits", "3"])
        assert exc_info.value.code != 0

    def test_no_bits_no_kv_bits_errors(self, mocker) -> None:
        """Omitting all bit-width args should error."""
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m"])
        assert exc_info.value.code != 0

    def test_k_bits_only_without_v_bits_errors(self, mocker) -> None:
        """--k-bits alone (without --v-bits) should error."""
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--k-bits", "4"])
        assert exc_info.value.code != 0

    def test_v_bits_only_without_k_bits_errors(self, mocker) -> None:
        """--v-bits alone (without --k-bits) should error."""
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--v-bits", "3"])
        assert exc_info.value.code != 0

    def test_backward_compat_bits_only(self, mocker) -> None:
        """--bits alone should still work (backward compat)."""
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 0
        _, args, kwargs = spy.mock_calls[0]
        assert kwargs["k_bits"] is None
        assert kwargs["v_bits"] is None
