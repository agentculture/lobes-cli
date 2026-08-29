"""Buildability guard for a variation's pinned MODEL_GEAR_VERSION (t10).

Every test here is hermetic: no network, no docker, no GPU. The offline
version-shape layer (:func:`is_dev_version`, :func:`check_buildability` with
no *index_query*) needs nothing but the version string. The live-index layer
is tested against a FAKE :data:`~lobes.runtime._buildability.IndexQuery`
callable — never :func:`~lobes.runtime._buildability.default_pypi_index_query`,
which is the one function in the module that touches the network and is
deliberately exercised nowhere in this suite.
"""

from __future__ import annotations

import pytest

from lobes.cli._errors import ModelGearError
from lobes.runtime._buildability import (
    PYPI_PACKAGE,
    BuildabilityResult,
    assert_buildable,
    check_buildability,
    check_lock_buildability,
    is_dev_version,
)
from lobes.runtime._lock import DeploymentLock

# --- is_dev_version: offline shape detection ---------------------------------


@pytest.mark.parametrize(
    "version",
    [
        "0.67.0.dev12",  # this repo's own publish.yml shape (.dev<run_number>)
        "0.67.0.dev0",
        "1.2.3dev5",  # bare PEP 440 form, no dot
        "1.2.3.dev",  # a dev segment with no number is still valid PEP 440
    ],
)
def test_dev_version_shapes_detected(version: str) -> None:
    assert is_dev_version(version) is True


@pytest.mark.parametrize(
    "version",
    [
        "0.67.0",
        "1.2.3",
        "0.67.0rc1",  # a release candidate is not a dev release
        "0.67.0.post1",
        "",
    ],
)
def test_released_or_non_dev_shapes_not_flagged(version: str) -> None:
    assert is_dev_version(version) is False


def test_is_dev_version_handles_none_without_raising() -> None:
    assert is_dev_version(None) is False  # type: ignore[arg-type]


# --- check_buildability: offline layer (no index_query) ----------------------


def test_empty_version_is_unversioned_not_released() -> None:
    result = check_buildability("")
    assert result.risk == "unversioned"
    assert result.installable is None
    assert "MODEL_GEAR_VERSION" in result.message


def test_none_version_is_unversioned() -> None:
    result = check_buildability(None)
    assert result.risk == "unversioned"


def test_released_version_classified_low_risk_offline() -> None:
    result = check_buildability("0.67.0")
    assert result.risk == "released"
    assert result.installable is None  # no index_query supplied -> not checked
    assert "0.67.0" in result.message


def test_dev_version_classified_ephemeral_offline_with_no_network() -> None:
    result = check_buildability("0.67.0.dev12")
    assert result.risk == "ephemeral_dev"
    assert result.installable is None
    assert "TestPyPI" in result.message or "dev" in result.message.lower()


def test_offline_layer_never_calls_an_index_query() -> None:
    calls = []

    def _boom(package: str, version: str) -> bool:
        calls.append((package, version))
        raise AssertionError("index_query must not be called when not supplied")

    # Sanity: check_buildability with index_query=None (the default) never
    # touches the fake — proving the offline layer is genuinely optional.
    check_buildability("0.67.0.dev1")
    assert calls == []


# --- check_buildability: injectable live-index layer (fake query) ------------


def test_fake_index_query_installable_true_is_reported() -> None:
    def _fake(package: str, version: str) -> bool:
        assert package == PYPI_PACKAGE
        assert version == "0.67.0"
        return True

    result = check_buildability("0.67.0", index_query=_fake)
    assert result.installable is True
    assert result.risk == "released"


def test_fake_index_query_installable_false_produces_clear_unbuildable_message() -> None:
    def _fake_gone(package: str, version: str) -> bool:
        return False

    result = check_buildability("0.67.0.dev99", index_query=_fake_gone)
    assert result.installable is False
    assert result.risk == "ephemeral_dev"
    # The message must name the exact pin and the Dockerfile step it breaks,
    # not a vague "something is wrong" — this is what a caller surfaces
    # verbatim instead of a pip traceback three layers into a docker build.
    assert "lobes-cli==0.67.0.dev99" in result.message
    assert "Dockerfile.gateway" in result.message
    assert "pip install" in result.message


def test_index_query_receives_the_configured_package_name() -> None:
    seen = {}

    def _fake(package: str, version: str) -> bool:
        seen["package"] = package
        return True

    check_buildability("1.0.0", package="my-other-pkg", index_query=_fake)
    assert seen["package"] == "my-other-pkg"


# --- assert_buildable: the early-failure boundary -----------------------------


def test_assert_buildable_raises_clearly_on_uninstallable_pin() -> None:
    result = BuildabilityResult(
        version="0.67.0.dev99",
        risk="ephemeral_dev",
        installable=False,
        message="lobes-cli==0.67.0.dev99 is not installable ... Dockerfile.gateway",
    )
    with pytest.raises(ModelGearError) as excinfo:
        assert_buildable(result)
    assert "0.67.0.dev99" in str(excinfo.value)


def test_assert_buildable_raises_on_unversioned() -> None:
    result = check_buildability("")
    with pytest.raises(ModelGearError):
        assert_buildable(result)


def test_assert_buildable_does_not_raise_when_installable_true() -> None:
    result = BuildabilityResult(version="0.67.0", risk="released", installable=True, message="ok")
    assert_buildable(result)  # must not raise


def test_assert_buildable_does_not_raise_on_unchecked_dev_version() -> None:
    # installable is None (no live check performed) — a warning-shaped fact,
    # not a proven failure, so this must not raise. The risk classification
    # is still visible on the result for a caller that wants to warn.
    result = check_buildability("0.67.0.dev1")
    assert result.installable is None
    assert_buildable(result)  # must not raise


def test_assert_buildable_error_names_the_pin_not_a_generic_message() -> None:
    result = check_buildability("0.67.0.dev5", index_query=lambda p, v: False)
    with pytest.raises(ModelGearError) as excinfo:
        assert_buildable(result)
    message = str(excinfo.value)
    assert "0.67.0.dev5" in message
    assert "docker build" in message.lower() or "pip install" in message.lower()


# --- check_lock_buildability: reads lobes_version off a captured lock --------


def test_check_lock_buildability_reads_version_from_the_lock() -> None:
    lock = DeploymentLock(
        variation="spark",
        env={},
        lobes_version="0.67.0.dev3",
    )
    result = check_lock_buildability(lock)
    assert result.version == "0.67.0.dev3"
    assert result.risk == "ephemeral_dev"


def test_check_lock_buildability_with_no_lobes_version_is_unversioned() -> None:
    lock = DeploymentLock(variation="spark", env={})
    result = check_lock_buildability(lock)
    assert result.risk == "unversioned"


def test_check_lock_buildability_forwards_a_fake_index_query() -> None:
    lock = DeploymentLock(variation="spark", env={}, lobes_version="0.67.0")

    def _fake(package: str, version: str) -> bool:
        return version == "0.67.0"

    result = check_lock_buildability(lock, index_query=_fake)
    assert result.installable is True


def test_check_lock_buildability_can_feed_assert_buildable_end_to_end() -> None:
    """The whole guard chained: a stale dev pin on a lock is caught, and
    early — no docker, no build, just the lock plus a fake index answer."""
    lock = DeploymentLock(variation="spark", env={}, lobes_version="0.55.0.dev7")

    def _gone(package: str, version: str) -> bool:
        return False  # this dev build no longer resolves anywhere

    result = check_lock_buildability(lock, index_query=_gone)
    with pytest.raises(ModelGearError) as excinfo:
        assert_buildable(result)
    assert "0.55.0.dev7" in str(excinfo.value)
