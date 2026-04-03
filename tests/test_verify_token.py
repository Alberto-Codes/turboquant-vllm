"""Tests for HF_TOKEN passthrough in verify CLI (extracted from test_verify.py)."""

from __future__ import annotations

import pytest

from turboquant_vllm.verify import _run_verification

pytestmark = [pytest.mark.unit]


class TestHFTokenPassthrough:
    """Verify that HF_TOKEN is forwarded to all from_pretrained calls."""

    def test_token_passed_to_auto_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AutoConfig.from_pretrained receives token= from HF_TOKEN env var."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_sentinel")
        captured: dict[str, object] = {}

        def spy(*args: object, **kwargs: object) -> object:
            captured["token"] = kwargs.get("token")
            raise RuntimeError("stop after capture")

        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", spy)
        with pytest.raises(RuntimeError, match="stop after capture"):
            _run_verification("fake/model", bits=4, threshold=0.99)

        assert captured["token"] == "hf_test_sentinel"

    def test_token_none_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """token= is None when HF_TOKEN is not in the environment."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        captured: dict[str, object] = {}

        def spy(*args: object, **kwargs: object) -> object:
            captured["token"] = kwargs.get("token")
            raise RuntimeError("stop after capture")

        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", spy)
        with pytest.raises(RuntimeError, match="stop after capture"):
            _run_verification("fake/model", bits=4, threshold=0.99)

        assert captured["token"] is None
