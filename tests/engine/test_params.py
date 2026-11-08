import json

import pytest
from pydantic import ValidationError

from precedent.config import references_dir
from precedent.engine.params import EngineParams, load_engine_params

ENGINE_JSON = json.loads((references_dir() / "precedent_params.json").read_text())["engine"]


def test_engine_params_hold_the_proposed_values():
    p = load_engine_params()
    assert (
        p.top_k,
        p.min_neff,
        p.shrink_k,
        p.story_window_sessions,
        p.nearest_shown,
        p.card_rows_per_type,
    ) == (30, 8.0, 10.0, 3, 5, 8)
    assert p.asset_horizons == (1, 5, 20) and p.sequence_horizons == (5, 20, 60)
    assert p.wilson_z == pytest.approx(1.2816) and p.market == "SPY"
    assert (p.placebo_samples, p.placebo_seed) == (500, 0)
    assert p.embedding_weight == 0.5 and ENGINE_JSON["embedding_weight"] == 0.5
    assert len(p.asset_panel) == 22 and p.asset_panel[0] == "SPY" and "TLT" in p.asset_panel


def test_params_are_frozen():
    with pytest.raises(ValidationError):
        load_engine_params().top_k = 3


def test_unknown_engine_param_is_rejected(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"engine": {**ENGINE_JSON, "top_kk": 3}}))
    with pytest.raises(ValidationError):
        load_engine_params(tmp_path / "p.json")


def test_a_missing_engine_param_is_rejected(tmp_path):
    partial = {k: v for k, v in ENGINE_JSON.items() if k != "top_k"}
    (tmp_path / "p.json").write_text(json.dumps({"engine": partial}))
    with pytest.raises(ValidationError):
        load_engine_params(tmp_path / "p.json")


def test_only_the_engine_section_is_read(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"gdelt": {"anything": 1}, "engine": ENGINE_JSON}))
    assert load_engine_params(tmp_path / "p.json") == EngineParams(**ENGINE_JSON)


def test_a_file_without_an_engine_section_is_an_error(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"gdelt": {}}))
    with pytest.raises(ValueError, match="engine"):
        load_engine_params(tmp_path / "p.json")


@pytest.mark.parametrize("weight", [-0.01, 1.01])
def test_embedding_weight_outside_the_unit_interval_is_rejected(tmp_path, weight):
    (tmp_path / "p.json").write_text(
        json.dumps({"engine": {**ENGINE_JSON, "embedding_weight": weight}})
    )
    with pytest.raises(ValidationError):
        load_engine_params(tmp_path / "p.json")
