from pathlib import Path

import pytest

from precedent import config


def test_require_env_names_the_missing_variable(monkeypatch):
    monkeypatch.delenv("PRECEDENT_TEST_VAR", raising=False)
    with pytest.raises(config.ConfigError, match="PRECEDENT_TEST_VAR"):
        config.require_env("PRECEDENT_TEST_VAR")


def test_require_env_rejects_an_empty_value(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TEST_VAR", "")
    with pytest.raises(config.ConfigError, match="PRECEDENT_TEST_VAR"):
        config.require_env("PRECEDENT_TEST_VAR")


def test_require_env_returns_the_value(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TEST_VAR", "abc")
    assert config.require_env("PRECEDENT_TEST_VAR") == "abc"


def test_data_dir_prefers_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    assert config.data_dir() == tmp_path


def test_data_dir_expands_the_default(monkeypatch):
    monkeypatch.delenv("PRECEDENT_DATA_DIR", raising=False)
    assert config.data_dir() == Path(config.DEFAULT_DATA_DIR).expanduser()
    assert "~" not in str(config.data_dir())


def test_references_dir_resolves_in_a_checkout():
    assert (config.references_dir() / "README.md").is_file()


def test_sample_dir_is_the_checkout_sample_when_present(monkeypatch, tmp_path):
    (tmp_path / "data" / "sample").mkdir(parents=True)
    monkeypatch.setattr(config, "_REPOSITORY_ROOT", tmp_path)
    assert config.sample_dir() == tmp_path / "data" / "sample"


def test_sample_dir_falls_back_to_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "cache"))
    assert config.sample_dir() == tmp_path / "cache"


def test_unset_model_and_snapshot_settings_are_none():
    assert config.PARSER_MODEL_ID is None
    assert config.SNAPSHOT_RELEASES_URL is None
