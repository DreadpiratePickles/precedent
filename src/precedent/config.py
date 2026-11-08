"""The only module that names a model id, a base URL or a data-dir default."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = "~/.cache/precedent"
SEC_SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"
GDELT_MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
POLYMARKET_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2026-10-21"
ANTHROPIC_TYPER_MODEL = "claude-haiku-5-5"
# PRECEDENT_MODEL_BASE_URL (env) is reserved for the parser runtime (Task 18); nothing reads it yet.
PARSER_MODEL_ID: str | None = None  # set after decision D2
SNAPSHOT_RELEASES_URL: str | None = None  # set by the owner when the GitHub repository exists

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PACKAGE_DIR.parents[1]


class ConfigError(RuntimeError):
    """A required setting is missing or empty."""


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ConfigError(f"environment variable {name} is required and is not set")
    return value


def data_dir() -> Path:
    return Path(os.environ.get("PRECEDENT_DATA_DIR") or DEFAULT_DATA_DIR).expanduser()


def release_base_url() -> str | None:
    """Where library snapshots are published, or None until the owner sets it."""
    return SNAPSHOT_RELEASES_URL


def references_dir() -> Path:
    packaged = _PACKAGE_DIR / "references"
    return packaged if packaged.is_dir() else _REPOSITORY_ROOT / "references"


def sample_dir() -> Path:
    checkout_sample = _REPOSITORY_ROOT / "data" / "sample"
    return checkout_sample if checkout_sample.is_dir() else data_dir()
