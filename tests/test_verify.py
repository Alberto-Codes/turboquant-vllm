"""Unit tests for the verify CLI module."""

from __future__ import annotations

import gc
import json
from types import SimpleNamespace

import pytest
import torch

from turboquant_vllm.verify import (
    COMPRESSION_QUALITY_THRESHOLD,
    VALIDATED_MODELS,
    _detect_model_config,
    _format_human_summary,
    _run_verification,
    main,
)

pytestmark = [pytest.mark.unit]


def _make_result(
    *,
    model: str = "test/model",
    bits: int = 4,
    k_bits: int | None = None,
    v_bits: int | None = None,
    per_layer_cosine: list[float] | None = None,
    min_cosine: float | None = 0.9995,
    threshold: float = COMPRESSION_QUALITY_THRESHOLD,
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
    resolved_k = k_bits if k_bits is not None else bits
    resolved_v = v_bits if v_bits is not None else bits
    result = {
        "model": model,
        "bits": bits,
        "k_bits": resolved_k,
        "v_bits": resolved_v,
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
    def test_model_and_bits_required(self, mocker) -> None:
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_model_flag_parsed(self, mocker) -> None:
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(model="allenai/Molmo2-4B"),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "allenai/Molmo2-4B", "--bits", "4"])
        assert exc_info.value.code == 0
        spy.assert_called_once_with(
            "allenai/Molmo2-4B",
            4,
            COMPRESSION_QUALITY_THRESHOLD,
            k_bits=None,
            v_bits=None,
        )

    def test_bits_flag_parsed(self, mocker) -> None:
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(bits=3),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "3"])
        assert exc_info.value.code == 0
        spy.assert_called_once_with(
            "test/m", 3, COMPRESSION_QUALITY_THRESHOLD, k_bits=None, v_bits=None
        )

    def test_threshold_default(self, mocker) -> None:
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4"])
        _, args, _ = spy.mock_calls[0]
        assert args[2] == COMPRESSION_QUALITY_THRESHOLD

    def test_threshold_custom(self, mocker) -> None:
        spy = mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(threshold=0.998),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--threshold", "0.998"])
        spy.assert_called_once_with("test/m", 4, 0.998, k_bits=None, v_bits=None)

    def test_json_flag_default_false(self, mocker, capsys) -> None:
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

    def test_json_flag_enables_json_output(self, mocker, capsys) -> None:
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
    def test_json_has_all_required_fields(self, mocker, capsys) -> None:
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
            "k_bits",
            "v_bits",
            "status",
            "validation",
            "threshold",
            "per_layer_cosine",
            "min_cosine",
            "versions",
        }
        assert required_fields.issubset(parsed.keys())

    def test_json_versions_has_three_keys(self, mocker, capsys) -> None:
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

    def test_pass_exits_zero(self, mocker) -> None:
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(min_cosine=0.9995, status="PASS"),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 0

    def test_fail_exits_one(self, mocker) -> None:
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(
                min_cosine=0.985,
                per_layer_cosine=[0.986, 0.985],
                status="FAIL",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 1

    def test_human_summary_to_stderr_with_json(self, mocker, capsys) -> None:
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        assert "Model:" in captured.err
        assert "Result:" in captured.err

    def test_human_summary_to_stdout_without_json(self, mocker, capsys) -> None:
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(),
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4"])
        captured = capsys.readouterr()
        assert "Model:" in captured.out
        assert "Result:" in captured.out

    def test_human_summary_format(self) -> None:
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

    def test_human_summary_many_layers_truncated(self) -> None:
        cosines = [0.999 + i * 0.0001 for i in range(32)]
        result = _make_result(
            per_layer_cosine=cosines,
            min_cosine=min(cosines),
        )
        summary = _format_human_summary(result)
        assert "more layers" in summary


class TestValidatedModels:
    def test_molmo2_exact_match(self) -> None:
        assert "molmo2" in VALIDATED_MODELS
        assert VALIDATED_MODELS["molmo2"] == "Molmo2"

    def test_mistral_exact_match(self) -> None:
        assert "mistral" in VALIDATED_MODELS
        assert VALIDATED_MODELS["mistral"] == "Mistral"

    def test_llama_exact_match(self) -> None:
        assert "llama" in VALIDATED_MODELS
        assert VALIDATED_MODELS["llama"] == "Llama"

    def test_qwen2_exact_match(self) -> None:
        assert "qwen2" in VALIDATED_MODELS
        assert VALIDATED_MODELS["qwen2"] == "Qwen2.5"

    def test_phi3_exact_match(self) -> None:
        assert "phi3" in VALIDATED_MODELS
        assert VALIDATED_MODELS["phi3"] == "Phi"

    def test_gemma2_exact_match(self) -> None:
        assert "gemma2" in VALIDATED_MODELS
        assert VALIDATED_MODELS["gemma2"] == "Gemma 2"

    def test_gemma3_exact_match(self) -> None:
        assert "gemma3" in VALIDATED_MODELS
        assert VALIDATED_MODELS["gemma3"] == "Gemma 3"

    def test_unvalidated_for_unknown_type(self) -> None:
        assert "gpt2" not in VALIDATED_MODELS

    def test_no_substring_match(self) -> None:
        # "molmo2" should not match "molmo2-extended" or "xmolmo2"
        assert "molmo2-extended" not in VALIDATED_MODELS
        assert "xmolmo2" not in VALIDATED_MODELS

    def test_display_name_mapping(self) -> None:
        for model_type, display_name in VALIDATED_MODELS.items():
            assert isinstance(model_type, str)
            assert isinstance(display_name, str)
            assert len(display_name) > 0

    def test_validated_result_field(self, mocker, capsys) -> None:
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

    def test_unvalidated_result_field(self, mocker, capsys) -> None:
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
    def test_default_threshold_value(self) -> None:
        assert COMPRESSION_QUALITY_THRESHOLD == 0.99

    def test_pass_above_threshold(self, mocker) -> None:
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(min_cosine=0.9995, status="PASS"),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 0

    def test_fail_below_threshold(self, mocker) -> None:
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=_make_result(
                min_cosine=0.985,
                per_layer_cosine=[0.986, 0.985],
                status="FAIL",
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["--model", "test/m", "--bits", "4"])
        assert exc_info.value.code == 1

    def test_custom_threshold_pass(self, mocker) -> None:
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

    def test_custom_threshold_fail(self, mocker) -> None:
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

    def test_exact_threshold_passes(self, mocker) -> None:
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


class TestDetectModelConfig:
    """Tests for _detect_model_config head_dim resolution and division guard."""

    @staticmethod
    def _cfg(
        *,
        head_dim: int | None = 128,
        num_heads: int = 8,
        hidden_size: int = 1024,
        num_kv_heads: int = 8,
        num_layers: int = 32,
        has_head_dim: bool = True,
    ) -> SimpleNamespace:
        attrs: dict = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "num_hidden_layers": num_layers,
        }
        if has_head_dim:
            attrs["head_dim"] = head_dim
        return SimpleNamespace(config=SimpleNamespace(**attrs))

    def test_explicit_head_dim(self) -> None:
        result = _detect_model_config(self._cfg(head_dim=128))
        assert result["head_dim"] == 128

    def test_head_dim_none_falls_back(self) -> None:
        result = _detect_model_config(
            self._cfg(head_dim=None, hidden_size=1024, num_heads=8)
        )
        assert result["head_dim"] == 128

    def test_num_heads_zero_with_explicit_head_dim(self) -> None:
        result = _detect_model_config(self._cfg(head_dim=128, num_heads=0))
        assert result["head_dim"] == 128

    def test_num_heads_zero_without_head_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="num_attention_heads=0"):
            _detect_model_config(self._cfg(head_dim=None, num_heads=0))

    def test_explicit_head_dim_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="head_dim=0"):
            _detect_model_config(self._cfg(head_dim=0))

    def test_explicit_head_dim_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="head_dim=-1"):
            _detect_model_config(self._cfg(head_dim=-1))

    def test_head_dim_absent_falls_back(self) -> None:
        result = _detect_model_config(
            self._cfg(has_head_dim=False, hidden_size=1024, num_heads=8)
        )
        assert result["head_dim"] == 128

    def test_vlm_nested_text_config(self) -> None:
        text_cfg = SimpleNamespace(
            hidden_size=1024,
            num_attention_heads=8,
            head_dim=128,
            num_key_value_heads=8,
            num_hidden_layers=32,
        )
        model = SimpleNamespace(config=SimpleNamespace(text_config=text_cfg))
        result = _detect_model_config(model)
        assert result["head_dim"] == 128


@pytest.mark.gpu
@pytest.mark.slow
class TestVerifyGPU:
    """GPU smoke tests for _run_verification on real hardware."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_run_verification_molmo2_passes(self) -> None:
        """End-to-end verification of Molmo2-4B on real GPU."""
        try:
            result = _run_verification("allenai/Molmo2-4B", bits=4, threshold=0.99)
            assert result["status"] == "PASS"
            assert result["min_cosine"] >= 0.99
            assert result["validation"] == "VALIDATED"
        finally:
            gc.collect()
            torch.cuda.empty_cache()


# TestHFTokenPassthrough extracted to test_verify_token.py (GH-67 test maturity)
