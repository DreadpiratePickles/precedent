import pytest

from precedent.engine.keys import key_from_dict, key_from_flags, match_fields, typed_as_line

GOOD = {
    "event_type": "central_bank",
    "actor_kind": "central_bank",
    "severity_band": "high",
    "is_scheduled": False,
}


def test_engine_reexports_the_spine_key():
    from precedent.engine import keys
    from precedent.spine import event_key

    assert keys.EventKey is event_key.EventKey
    assert keys.build_event_key is event_key.build_event_key
    assert keys.severity_band is event_key.severity_band


def test_key_from_flags_defaults_and_validation():
    k = key_from_flags("central_bank", "central_bank", "high", False, country_group="US")
    assert (k.impacted_kinds, k.direction_sign) == ((), 0)
    with pytest.raises(ValueError, match="unknown event_type"):
        key_from_flags("weather", "government", "high", False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actor_kind": "alien"}, "unknown actor_kind"),
        ({"severity_band": "extreme"}, "unknown severity_band"),
        ({"country_group": "Mars"}, "unknown country_group"),
        ({"impacted_kinds": ("bond",)}, "unknown impacted_kinds"),
        ({"direction_sign": 2}, "unknown direction_sign"),
    ],
)
def test_key_from_flags_validates_every_enum_like_field(overrides, message):
    with pytest.raises(ValueError, match=message):
        key_from_flags(**{**GOOD, **overrides})


def test_key_from_flags_sorts_and_dedupes_impacted_kinds():
    k = key_from_flags(**GOOD, impacted_kinds=("sector", "company", "sector"))
    assert k.impacted_kinds == ("company", "sector")


def test_key_from_dict_rejects_unknown_fields():
    assert key_from_dict(GOOD) == key_from_flags("central_bank", "central_bank", "high", False)
    with pytest.raises(ValueError, match="colour"):
        key_from_dict({**GOOD, "colour": "red"})


def test_key_from_dict_rejects_a_missing_required_field():
    with pytest.raises(ValueError, match="event_type"):
        key_from_dict({k: v for k, v in GOOD.items() if k != "event_type"})


def test_key_from_dict_accepts_impacted_kinds_as_a_list_and_validates():
    k = key_from_dict({**GOOD, "impacted_kinds": ["index", "company"], "direction_sign": -1})
    assert (k.impacted_kinds, k.direction_sign) == (("company", "index"), -1)
    with pytest.raises(ValueError, match="unknown event_type"):
        key_from_dict({**GOOD, "event_type": "weather"})


def test_typed_as_line():
    k = key_from_flags("central_bank", "central_bank", "high", False, country_group="US")
    assert typed_as_line(k) == "central_bank / central_bank / US / severity high / unscheduled"
    scheduled = key_from_flags("earnings", "company", "low", True)
    assert typed_as_line(scheduled) == "earnings / company / none / severity low / scheduled"


def test_match_fields_order():
    k = key_from_flags("central_bank", "central_bank", "high", False, country_group="US")
    assert match_fields(k) == ("central_bank", "central_bank", "US", "high", False)
