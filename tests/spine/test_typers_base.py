"""Tests for the typer protocol value object and the fake typer."""
import pytest
from pydantic import ValidationError

from precedent.spine.event_key import EventKey
from precedent.spine.typers.base import TYPING_FIELDS, TypedText
from precedent.spine.typers.fake import FakeTyper

KEY = EventKey("central_bank", "central_bank", ("country",), "US", "mid", True, 1)


def test_typed_text_requires_a_key_exactly_when_typed():
    with pytest.raises(ValidationError):
        TypedText(status="typed", key=None, reason="", evidence_span="cuts rates", typer_version="v")
    with pytest.raises(ValidationError):
        TypedText(status="typed", key=KEY, reason="", evidence_span=None, typer_version="v")
    with pytest.raises(ValidationError):
        TypedText(status="untyped", key=KEY, reason="", evidence_span=None, typer_version="v")
    for status in ("untyped", "refused_invalid"):
        with pytest.raises(ValidationError):
            TypedText(status=status, key=None, reason="", evidence_span="x", typer_version="v")
    typed = TypedText(status="typed", key=KEY, reason="r", evidence_span="cuts", typer_version="v")
    assert typed.key == KEY


def test_typed_text_is_frozen_and_forbids_extra_fields():
    typed = TypedText(status="untyped", key=None, reason="r", evidence_span=None, typer_version="v")
    with pytest.raises(ValidationError):
        typed.reason = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TypedText(status="untyped", key=None, reason="r", evidence_span=None, typer_version="v",
                  extra=1)  # type: ignore[call-arg]


def test_typing_fields_are_the_event_key_fields():
    assert set(TYPING_FIELDS) == set(EventKey.__dataclass_fields__)


def test_fake_typer_returns_the_mapping_and_untypes_the_rest():
    mapped = TypedText(status="typed", key=KEY, reason="r", evidence_span="x", typer_version="v")
    fake = FakeTyper({"a": mapped})
    assert fake.type_text("a") == mapped
    missing = fake.type_text("b")
    assert (missing.status, missing.key, missing.reason) == ("untyped", None, "not in mapping")
    assert missing.typer_version == fake.version == "fake-v1"
