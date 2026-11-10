"""PEP 440 ordering and downgrade refusal for the gradual fleet rollout."""

import pytest

from lobes.rollout._versions import (
    DowngradeRefused,
    check_target,
    is_downgrade,
    parse_version,
)


def test_dev_release_sorts_before_release_of_same_base():
    assert parse_version("0.92.0.dev694") < parse_version("0.92.0")


def test_documented_ordering_chain():
    chain = ["0.92.0.dev694", "0.93.0", "0.94.0", "0.94.1"]
    parsed = [parse_version(v) for v in chain]
    assert parsed == sorted(parsed)
    assert parse_version("0.92.0.dev694") < parse_version("0.94.0") < parse_version("0.94.1")


def test_dev_before_pre_before_release():
    assert (
        parse_version("1.0.0.dev1")
        < parse_version("1.0.0a1")
        < parse_version("1.0.0b1")
        < parse_version("1.0.0rc1")
        < parse_version("1.0.0")
    )


def test_post_release_sorts_after_release():
    assert parse_version("1.0.0") < parse_version("1.0.0.post1") < parse_version("1.0.1")


def test_numeric_components_compare_as_integers():
    assert parse_version("0.9.0") < parse_version("0.10.0")
    assert parse_version("0.94.0") < parse_version("0.94.1")


def test_trailing_zero_release_components_are_equal():
    assert parse_version("1.0") == parse_version("1.0.0")
    assert not parse_version("1.0") < parse_version("1.0.0")


def test_dev_release_of_later_base_is_after_earlier_release():
    assert parse_version("0.94.0.dev1") > parse_version("0.93.9")


def test_malformed_version_is_rejected():
    with pytest.raises(ValueError):
        parse_version("not-a-version")


def test_is_downgrade_true_when_target_is_older():
    assert is_downgrade("0.93.0", "0.94.0") is True
    assert is_downgrade("0.92.0.dev694", "0.94.0") is True


def test_is_downgrade_false_for_upgrade_or_same():
    assert is_downgrade("0.94.1", "0.94.0") is False
    assert is_downgrade("0.94.0", "0.94.0") is False


def test_check_target_refuses_downgrade_by_default():
    with pytest.raises(DowngradeRefused):
        check_target("0.93.0", "0.94.0")


def test_check_target_allows_downgrade_when_opted_in():
    check_target("0.93.0", "0.94.0", allow_downgrade=True)


def test_check_target_allows_upgrade():
    check_target("0.94.1", "0.94.0")


def test_downgrade_refused_is_a_value_error():
    assert issubclass(DowngradeRefused, ValueError)
