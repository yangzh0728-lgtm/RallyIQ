"""Pipeline profiles: own footage turns the stroke residual on; broadcast/unset leave defaults."""
import json

import pytest

from rallyiq.pipeline.profiles import config_from_env, profile_config


def test_own_profile_turns_residual_on(monkeypatch):
    monkeypatch.delenv("RALLYIQ_PIPELINE_PROFILE", raising=False)
    monkeypatch.delenv("RALLYIQ_PIPELINE_CONFIG", raising=False)
    assert config_from_env() == {}
    assert config_from_env(profile="broadcast") == {}
    assert config_from_env({"demo": True}, "own") == {"demo": True, "strokes": {"residual": "auto"}}
    monkeypatch.setenv("RALLYIQ_PIPELINE_PROFILE", "own")
    assert config_from_env()["strokes"]["residual"] == "auto"


def test_json_override_and_unknown(monkeypatch, tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"strokes": {"sequence": False}}))
    monkeypatch.setenv("RALLYIQ_PIPELINE_PROFILE", "own")
    monkeypatch.setenv("RALLYIQ_PIPELINE_CONFIG", str(p))
    assert config_from_env()["strokes"] == {"residual": "auto", "sequence": False}
    with pytest.raises(ValueError):
        profile_config("nope")
