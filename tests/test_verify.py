"""Unit tests for the verify CLI module."""

from __future__ import annotations

import gc
import json

import pytest
import torch

from turboquant_vllm.verify import (
    CACHE_PARITY_THRESHOLD,
    VALIDATED_MODELS,
    _format_human_summary,
    _run_verification,
    main,
)

pytestmark = [pytest.mark.unit]


def _make_result(
    *,
    model: str = "test/model",
    bits: int = 4,
    per_layer_cosine: list[float] | None = None,
    min_cosine: float | None = 0.9995,
    threshold: float = CACHE_PARITY_THRESHOLD,
    validation: str = "VALIDATED",
    family_name: str | None = "Molmo2",
    status: str | None = None,
) -> dict:
    """Build a synthetic verification result dict."""
    if per_layer_cosine is None:
        per_layer_cosine = [0.9995, 0.9993]
    if min_cosine is None:
        min_cosine = min(per_layer_cosine)
    if status is None:
        status = "PASS" if min_cosine >= threshold else "FAIL"
    result = {
        "model": model,
        "bits": bits,
        "status": status,
        "validation": validation,
        "threshold": threshold,
        "per_layer_cosine": per_layer_cosine,
        "min_cosine": min_cosine,
        "versions": {
            "turboquant_vllm": "1.1.1",
            "transformers": "4.57.0",
            "torch": "2.6.0",
        },
    }
    if family_name is not None:
        result["family_name"] = family_name
    return result


class TestVerifyArgparse:
    def test_model_and_bits_required(self, mocker):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_model_flag_parsed(self, mocker):
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(model="allenai/Molmo2-4B"),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "allenai/Molmo2-4B", "--bits", "4"])
        assert exc_info.value.code == 0
        spy.assert_called_once_with("allenai/Molmo2-4B", 4, CACHE_PARITY_THRESHOLD)

    def test_bits_flag_parsed(self, mocker):
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(bits=3),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "3"])
        assert exc_info.value.code == 0
        spy.assert_called_once_with("test/m", 3, CACHE_PARITY_THRESHOLD)

    def test_threshold_default(self, mocker):
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4"])
        _, args, _ = spy.mock_calls[0]
        assert args[2] == CACHE_PARITY_THRESHOLD

    def test_threshold_custom(self, mocker):
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(threshold=0.998),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--threshold", "0.998"])
        spy.assert_called_once_with("test/m", 4, 0.998)

    def test_json_flag_default_false(self, mocker, capsys):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4"])
        captured = capsys.readouterr()
        # Without --json, stdout should NOT be valid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(captured.out)

    def test_json_flag_enables_json_output(self, mocker, capsys):
        result = _make_result()
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=result,
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["model"] == "test/model"


class TestVerifyOutput:
    def test_json_has_all_required_fields(self, mocker, capsys):
        result = _make_result()
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=result,
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        required_fields = {
            "model",
            "bits",
            "status",
            "validation",
            "threshold",
            "per_layer_cosine",
            "min_cosine",
            "versions",
        }
        assert required_fields.issubset(parsed.keys())

    def test_json_versions_has_three_keys(self, mocker, capsys):
        result = _make_result()
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=result,
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert set(parsed["versions"].keys()) == {
            "turboquant_vllm",
            "transformers",
            "torch",
        }

    def test_pass_exits_zero(self, mocker):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(min_cosine=0.9995, status="PASS"),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 0

    def test_fail_exits_one(self, mocker):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(
                min_cosine=0.997,
                per_layer_cosine=[0.998, 0.997],
                status="FAIL",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 1

    def test_human_summary_to_stderr_with_json(self, mocker, capsys):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        assert "Model:" in captured.err
        assert "Result:" in captured.err

    def test_human_summary_to_stdout_without_json(self, mocker, capsys):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4"])
        captured = capsys.readouterr()
        assert "Model:" in captured.out
        assert "Result:" in captured.out

    def test_human_summary_format(self):
        result = _make_result(
            model="allenai/Molmo2-4B",
            bits=4,
            per_layer_cosine=[0.9995, 0.9993, 0.9991],
            min_cosine=0.9991,
            validation="VALIDATED",
            family_name="Molmo2",
        )
        summary = _format_human_summary(result)
        assert "allenai/Molmo2-4B" in summary
        assert "VALIDATED" in summary
        assert "Molmo2" in summary
        assert "0.9991" in summary
        assert "PASS" in summary

    def test_human_summary_many_layers_truncated(self):
        cosines = [0.999 + i * 0.0001 for i in range(32)]
        result = _make_result(
            per_layer_cosine=cosines,
            min_cosine=min(cosines),
        )
        summary = _format_human_summary(result)
        assert "more layers" in summary


class TestValidatedModels:
    def test_molmo2_exact_match(self):
        assert "molmo2" in VALIDATED_MODELS
        assert VALIDATED_MODELS["molmo2"] == "Molmo2"

    def test_unvalidated_for_unknown_type(self):
        assert "llama" not in VALIDATED_MODELS
        assert "mistral" not in VALIDATED_MODELS

    def test_no_substring_match(self):
        # "molmo2" should not match "molmo2-extended" or "xmolmo2"
        assert "molmo2-extended" not in VALIDATED_MODELS
        assert "xmolmo2" not in VALIDATED_MODELS

    def test_display_name_mapping(self):
        for model_type, display_name in VALIDATED_MODELS.items():
            assert isinstance(model_type, str)
            assert isinstance(display_name, str)
            assert len(display_name) > 0

    def test_validated_result_field(self, mocker, capsys):
        result = _make_result(validation="VALIDATED", family_name="Molmo2")
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=result,
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["validation"] == "VALIDATED"

    def test_unvalidated_result_field(self, mocker, capsys):
        result = _make_result(validation="UNVALIDATED", family_name=None)
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=result,
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["validation"] == "UNVALIDATED"


class TestVerifyThreshold:
    def test_default_threshold_value(self):
        assert CACHE_PARITY_THRESHOLD == 0.999

    def test_pass_above_threshold(self, mocker):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(min_cosine=0.9995, status="PASS"),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 0

    def test_fail_below_threshold(self, mocker):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(
                min_cosine=0.998,
                per_layer_cosine=[0.9985, 0.998],
                status="FAIL",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 1

    def test_custom_threshold_pass(self, mocker):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(
                threshold=0.998,
                min_cosine=0.9985,
                status="PASS",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4", "--threshold", "0.998"])
        assert exc_info.value.code == 0

    def test_custom_threshold_fail(self, mocker):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(
                threshold=0.9999,
                min_cosine=0.9995,
                per_layer_cosine=[0.9995],
                status="FAIL",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4", "--threshold", "0.9999"])
        assert exc_info.value.code == 1

    def test_exact_threshold_passes(self, mocker):
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(
                threshold=0.999,
                min_cosine=0.999,
                per_layer_cosine=[0.999],
                status="PASS",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 0


@pytest.mark.gpu
@pytest.mark.slow
class TestVerifyGPU:
    """GPU smoke tests for _run_verification on real hardware."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_run_verification_molmo2_passes(self) -> None:
        """End-to-end verification of Molmo2-4B on real GPU."""
        # 0.99 = compression quality tier for random Gaussian data at 4-bit
        # (verify.py's default 0.999 is cache parity tier — wrong for this comparison)
        result = _run_verification("allenai/Molmo2-4B", bits=4, threshold=0.99)
        assert result["status"] == "PASS"
        assert result["min_cosine"] > 0.99
        assert result["validation"] == "VALIDATED"
        gc.collect()
        torch.cuda.empty_cache()
