"""Research: the ablation ladder. Each rung adds factor panels to the price features; every
(rung, fold) pair is one trial, and the run manifest counts them all.

A rung is one YAML file (`references/rungs/*.yaml`) naming the panel-key prefixes it adds on top
of the price features, so a research track adds a rung by adding a file. Folds are purged
walk-forward segments (`purged_splits`); the purge gap must cover the label's look-ahead, which
is derived from the label expression rather than set by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import DataLoader

from precedent.spine.qlib_dataset import make_alpha_handler
from precedent.spine.rank_ic import daily_rank_ic, ic_summary

LABEL_NAME = "LABEL0"
EARLY_STOPPING_ROUNDS = 10
RESULT_COLUMNS = ["rung", "fold", "rank_ic_mean", "icir", "n_days"]
FoldSegments = Mapping[str, tuple[pd.Timestamp, pd.Timestamp]]
_REF_CALL = re.compile(r"\bRef\s*\(")
_INTEGER_LITERAL = re.compile(r"([+-]?)\s*(\d+)")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class Rung(BaseModel):
    """One rung of the ladder: which factor panels (by key prefix) it adds to price features."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    panel_prefixes: list[str]


def load_rungs(rungs_dir: Path) -> list[Rung]:
    """Read every `*.yaml` in `rungs_dir` (sorted by file name) as one validated rung."""
    paths = sorted(rungs_dir.glob("*.yaml"))
    if not paths:
        raise ValueError(f"no rung files (*.yaml) in {rungs_dir}")
    rungs = []
    for path in paths:
        try:
            rungs.append(Rung.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))
        except (ValidationError, yaml.YAMLError) as error:
            raise ValueError(f"invalid rung file {path.name}: {error}") from error
    names = [rung.name for rung in rungs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"rung names must be unique, repeated: {duplicates}")
    return rungs


def label_horizon_of(label_expr: str) -> int:
    """The furthest look-ahead, in sessions, of a label expression: the most negative `Ref` shift.

    `Ref($open, -2)/Ref($open, -1) - 1` gives 2. An expression with no forward `Ref` gives 0.
    """
    horizon = 0
    for match in _REF_CALL.finditer(label_expr):
        depth, last_comma, position = 1, None, match.end()
        while position < len(label_expr) and depth:
            char = label_expr[position]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 1:
                last_comma = position
            position += 1
        if depth or last_comma is None:
            raise ValueError(f"cannot read the Ref(...) call at {match.start()} in {label_expr!r}")
        shift = label_expr[last_comma + 1 : position - 1].strip()
        parsed = _INTEGER_LITERAL.fullmatch(shift)
        if parsed is None:
            raise ValueError(
                f"Ref shift must be an integer literal, got {shift!r} in {label_expr!r}"
            )
        if parsed.group(1) == "-":
            horizon = max(horizon, int(parsed.group(2)))
    return horizon


def panel_sha256(panel: pd.DataFrame) -> str:
    """SHA-256 of a panel, independent of row and column order (sorted, exact-float CSV)."""
    canonical = panel.sort_index().sort_index(axis=1)
    return hashlib.sha256(canonical.to_csv(float_format="%.17g").encode("utf-8")).hexdigest()


def code_commit(repo_dir: Path) -> str:
    """The `git rev-parse HEAD` of the checkout holding `repo_dir`, or "unknown" outside git."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not _GIT_COMMIT.fullmatch(commit):
        return "unknown"
    return commit


def _purge_gaps(sessions: pd.DatetimeIndex, fold: FoldSegments) -> dict[str, int]:
    """Sessions strictly between train and valid, and between valid and test."""

    def position_at_or_before(day: pd.Timestamp) -> int:
        return int(sessions.searchsorted(day, side="right")) - 1

    def position_at_or_after(day: pd.Timestamp) -> int:
        return int(sessions.searchsorted(day, side="left"))

    return {
        "train_to_valid": position_at_or_after(fold["valid"][0])
        - position_at_or_before(fold["train"][1])
        - 1,
        "valid_to_test": position_at_or_after(fold["test"][0])
        - position_at_or_before(fold["valid"][1])
        - 1,
    }


def check_purge_gap(sessions: pd.DatetimeIndex, fold: FoldSegments, label_horizon: int) -> None:
    """Raise unless both purge gaps in `fold` are at least `label_horizon` sessions."""
    short = {
        boundary: gap
        for boundary, gap in _purge_gaps(sessions, fold).items()
        if gap < label_horizon
    }
    if short:
        raise ValueError(
            f"purge gap {short} is shorter than the label horizon of {label_horizon} sessions; "
            "a training label would overlap the next segment"
        )


def fit_fold_predictions(
    handler: DataHandlerLP, fold: FoldSegments, lgb_params: Mapping[str, Any]
) -> pd.DataFrame:
    """Fit LightGBM on `train` (early stopping on `valid`) and predict `test`.

    Returns a frame indexed by (datetime, instrument) with columns `pred` and `label`.
    Training and validation use the learn view (labels rank-normalised); the test features use
    the inference view and the label is the raw forward return.
    """
    dataset = DatasetH(handler, segments={name: tuple(span) for name, span in fold.items()})
    columns = ["feature", "label"]
    train, valid = dataset.prepare(["train", "valid"], col_set=columns, data_key=handler.DK_L)
    test = dataset.prepare("test", col_set=columns, data_key=handler.DK_I)
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        train["feature"],
        train["label"][LABEL_NAME],
        eval_X=valid["feature"],
        eval_y=valid["label"][LABEL_NAME],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return pd.DataFrame(
        {"pred": model.predict(test["feature"]), "label": test["label"][LABEL_NAME].to_numpy()},
        index=test.index,
    )


def _panels_for(rung: Rung, factor_panels: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for prefix in rung.panel_prefixes:
        if not any(key.startswith(prefix) for key in factor_panels):
            raise ValueError(
                f"rung {rung.name!r} adds prefix {prefix!r} but no supplied panel key starts "
                f"with it (panels: {sorted(factor_panels)})"
            )
    return {
        key: panel
        for key, panel in factor_panels.items()
        if any(key.startswith(prefix) for prefix in rung.panel_prefixes)
    }


def _price_index(
    price_loader: DataLoader, start: pd.Timestamp, end: pd.Timestamp
) -> pd.MultiIndex:
    """The (datetime, instrument) rows the price loader has over `[start, end]`."""
    handler, _ = make_alpha_handler(price_loader, {}, start, end)
    return handler.fetch(col_set="__raw", data_key=handler.DK_R).index


def _count_rows_without_price_row(
    panel: pd.DataFrame, price_index: pd.MultiIndex, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    """Panel rows inside `[start, end]` that have no price row (the join drops them)."""
    dates = panel.index.get_level_values("datetime")
    in_window = panel.index[(dates >= start) & (dates <= end)]
    return int((~in_window.isin(price_index)).sum())


def _write_text_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def run_ablation(
    rungs: Sequence[Rung],
    price_loader: DataLoader,
    factor_panels: Mapping[str, pd.DataFrame],
    folds: Sequence[FoldSegments],
    out_dir: Path,
    lgb_params: Mapping[str, Any],
    *,
    parser_version: str,
    label_expr: str,
    rungs_dir: Path,
) -> pd.DataFrame:
    """Run every rung on every fold; write `ablation.csv` and `run_manifest.json` to `out_dir`.

    Each (rung, fold) is one trial with its own handler over `[train start, test end]` of the
    fold. Raises `ValueError` before fitting anything if any fold's purge gap is shorter than the
    label horizon derived from `label_expr`, or a rung names a prefix no panel carries. Factor rows
    with no price row are counted once per panel over the union window of all folds, not per trial.
    """
    if not rungs or not folds:
        raise ValueError("run_ablation needs at least one rung and one fold")
    horizon = label_horizon_of(label_expr)
    selected = {rung.name: _panels_for(rung, factor_panels) for rung in rungs}
    window_start = min(fold["train"][0] for fold in folds)
    window_end = max(fold["test"][1] for fold in folds)
    price_index = _price_index(price_loader, window_start, window_end)
    sessions = pd.DatetimeIndex(price_index.get_level_values("datetime").unique().sort_values())
    for fold in folds:
        check_purge_gap(sessions, fold, horizon)
    orphan_rows = {
        key: _count_rows_without_price_row(panel, price_index, window_start, window_end)
        for key, panel in sorted(factor_panels.items())
    }
    rows: list[dict[str, Any]] = []
    for rung in rungs:
        for fold_index, fold in enumerate(folds):
            handler, _ = make_alpha_handler(
                price_loader, selected[rung.name], fold["train"][0], fold["test"][1]
            )
            predictions = fit_fold_predictions(handler, fold, lgb_params)
            summary = ic_summary(daily_rank_ic(predictions["pred"], predictions["label"]))
            rows.append(
                {
                    "rung": rung.name,
                    "fold": fold_index,
                    "rank_ic_mean": summary["mean"],
                    "icir": summary["icir"],
                    "n_days": summary["n_days"],
                }
            )
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    manifest = {
        "code_commit": code_commit(Path(__file__).resolve().parent),
        "parser_version": parser_version,
        "label_expr": label_expr,
        "label_horizon": horizon,
        "panel_sha256": {key: panel_sha256(panel) for key, panel in sorted(factor_panels.items())},
        "rungs": [rung.model_dump() for rung in rungs],
        "rung_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(rungs_dir.glob("*.yaml"))
        },
        "folds": [
            {name: [day.isoformat() for day in span] for name, span in fold.items()}
            for fold in folds
        ],
        "lgb_params": dict(lgb_params),
        "trial_count": len(rows),
        "events_without_price_row": orphan_rows,
        "events_without_price_row_total": sum(orphan_rows.values()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomically(out_dir / "ablation.csv", result.to_csv(index=False))
    _write_text_atomically(
        out_dir / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return result
