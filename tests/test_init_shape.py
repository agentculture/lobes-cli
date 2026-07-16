"""Tests for ``lobes init --shape`` (brain-shapes t4).

``--shape`` selects the deployment-shape axis (brain-shapes t1-t3: which roles
a box HOSTS) at scaffold time, composed over whichever per-machine
:class:`~lobes.profiles.schema.Profile` detection/``--profile`` resolves
(issue #110), via :func:`lobes.profiles.shape_render.render_shape` — never a
re-implementation of that composition. Bare ``lobes init`` (no ``--shape``)
resolves the ``machine-as-brain`` identity shape — a strict no-op over the
profile (the invariant ``tests/test_shape_goldens.py`` pins) — so the default
path renders exactly as it did before this flag existed: zero new required
decisions.

Card detection is injected via ``monkeypatch.setattr(_detect, "detect_card",
...)`` — this repo's offline-probe idiom (see ``tests/conftest.py`` and
``tests/test_init_profile.py``) — so none of these touch real hardware.
"""

from __future__ import annotations

import json

import yaml

from lobes.cli import main
from lobes.profiles.loader import resolve_profile
from lobes.profiles.render import profile_env
from lobes.profiles.shape_render import render_shape
from lobes.profiles.shapes import builtin_shape_names, resolve_shape
from lobes.runtime import _compose, _detect, _env


class _ComposeTagLoader(yaml.SafeLoader):
    """A SafeLoader that tolerates docker-compose's `!reset` / `!override` merge tags.

    PyYAML rejects unknown custom tags; compose override files use them. `!reset`
    clears an attribute (its value is ignored by compose), so we resolve it to a
    sentinel that records the tag was used; `!override` keeps its underlying value.
    """


def _construct_reset(loader, node):  # noqa: ANN001 — pyyaml constructor signature
    return {"__reset__": True}


def _construct_override(loader, node):  # noqa: ANN001 — pyyaml constructor signature
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


_ComposeTagLoader.add_constructor("!reset", _construct_reset)
_ComposeTagLoader.add_constructor("!override", _construct_override)


def _load_shape_override(target) -> dict:
    """Parse ``docker-compose.shape.yml`` with the merge-tag-tolerant loader."""
    return yaml.load(
        (target / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8"),
        Loader=_ComposeTagLoader,
    )


def _fake_card(
    resolved: str,
    *,
    device_name: str | None = "NVIDIA GB10",
    compute_capability: str | None = "sm_121",
    total_memory_gb: float | None = 119.7,
) -> _detect.DetectedCard:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name=device_name,
        compute_capability=compute_capability,
        total_memory_gb=total_memory_gb,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


def _patch_detect(monkeypatch, card: _detect.DetectedCard) -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda: card)


# --- acceptance 2: bare init == explicit machine-as-brain, byte-identical ---


def test_bare_init_env_equals_explicit_machine_as_brain_shape(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    bare = tmp_path / "bare"
    explicit = tmp_path / "explicit"
    assert main(["init", str(bare), "--apply"]) == 0
    assert main(["init", "--shape", "machine-as-brain", str(explicit), "--apply"]) == 0
    assert (bare / ".env").read_text() == (explicit / ".env").read_text()
    assert (bare / "docker-compose.yml").read_text() == (
        explicit / "docker-compose.yml"
    ).read_text()


def _numerically_equal(a: str | None, b: str) -> bool:
    """Same tolerant comparison lobes.cli._commands.init._values_equal applies:
    a resolved float restating a template default in fewer digits (0.3 vs the
    shipped 0.30) is the SAME value, not a mismatch."""
    if a == b:
        return True
    try:
        return float(a) == float(b)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def test_bare_init_env_matches_pre_shape_profile_env(tmp_path, monkeypatch) -> None:
    """The rendered .env carries every value plain profile_env(profile) would —
    the pre-change rendering, from before --shape existed at all — modulo the
    pre-existing numeric-formatting tolerance (see _values_equal in init.py)."""
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    env = _env.read_env_file(target / ".env")
    expected = profile_env(resolve_profile("thor"))
    for key, value in expected.items():
        assert _numerically_equal(env.get(key), value), f"{key}: {env.get(key)!r} != {value!r}"


def test_bare_init_dry_run_names_default_shape(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Shape: machine-as-brain" in out
    assert not target.exists()


def test_bare_init_apply_json_reports_default_shape(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape"] == "machine-as-brain"


def test_single_topology_is_untouched_by_shape_machinery(tmp_path, monkeypatch) -> None:
    """--single (no --shape given at all) never resolves a shape or calls
    detection — mirrors test_single_topology_never_calls_detection."""
    calls = []
    monkeypatch.setattr(_detect, "detect_card", lambda: calls.append(1) or _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", "--single", str(target), "--apply"])
    assert rc == 0
    assert calls == []


# --- shape selection changes the rendered .env ------------------------------


def test_shape_spark_lobe_drops_senses_and_reclaims_cortex_budget(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target), "--apply"])
    assert rc == 0
    env = _env.read_env_file(target / ".env")
    assert env["MULTIMODAL_FEASIBLE"] == "false"
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.44"


def test_shape_thor_lobe_drops_cortex_and_reclaims_senses_budget(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "thor-lobe", str(target), "--apply"])
    assert rc == 0
    env = _env.read_env_file(target / ".env")
    assert env["PRIMARY_FEASIBLE"] == "false"
    assert env["MULTIMODAL_GPU_MEM_UTIL"] == "0.3"


def test_scaffolded_env_matches_render_shape_directly(tmp_path, monkeypatch) -> None:
    """The scaffolded .env carries every key render_shape() itself would produce
    — init never reimplements the shape x profile composition."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    env = _env.read_env_file(target / ".env")
    expected = render_shape(resolve_shape("spark-lobe"), resolve_profile("spark")).env
    for key, value in expected.items():
        assert env.get(key) == value, f"{key}: {env.get(key)!r} != {value!r}"


def test_shape_composes_with_explicit_profile_override(tmp_path, monkeypatch) -> None:
    """--shape composes over WHICHEVER profile is resolved, including a forced
    --profile — the two axes are independent."""
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", "--profile", "spark", str(target), "--apply"])
    assert rc == 0
    env = _env.read_env_file(target / ".env")
    assert env["MULTIMODAL_FEASIBLE"] == "false"
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.44"
    assert env["PRIMARY_KV_CACHE_DTYPE"] == "fp8"  # spark's own default, not thor's


# --- unknown --shape is a user error naming the valid (sorted) shapes ------


def test_unknown_shape_is_user_error_naming_valid_shapes(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "not-a-real-shape", str(target), "--apply"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown shape" in err
    assert "machine-as-brain, orin-small, spark-lobe, thor-lobe" in err
    assert not target.exists()  # aborts before any scaffolding


def test_unknown_shape_dry_run_also_errors_before_writing(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "not-a-real-shape", str(target)])
    assert rc == 1
    assert not target.exists()


def test_builtin_shape_names_are_sorted() -> None:
    assert list(builtin_shape_names()) == sorted(builtin_shape_names())
    assert builtin_shape_names() == (
        "machine-as-brain",
        "orin-small",
        "spark-lobe",
        "thor-lobe",
        "thor-muse",
    )


# --- --shape x --single: user error (shapes are a fleet-scaffold axis) -----


def test_shape_with_single_is_a_user_error(tmp_path, capsys) -> None:
    target = tmp_path / "x"
    rc = main(["init", "--single", "--shape", "spark-lobe", str(target), "--apply"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--shape" in err and "--single" in err
    assert not target.exists()


def test_shape_with_single_errors_even_for_the_default_shape_name(tmp_path, capsys) -> None:
    # The recommendation this encodes: shapes are a fleet-scaffold axis, so
    # ANY explicit --shape (even spelling out the default) conflicts with
    # --single, which never runs profile/shape resolution at all.
    target = tmp_path / "x"
    rc = main(["init", "--single", "--shape", "machine-as-brain", str(target), "--apply"])
    assert rc == 1
    assert not target.exists()


# --- --shape x --audio: harmless / idempotent -------------------------------


def test_shape_with_audio_scaffolds_both_normally(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", "--audio", str(target), "--apply"])
    assert rc == 0
    assert (target / "docker-compose.audio.yml").is_file()
    env = _env.read_env_file(target / ".env")
    assert env["MULTIMODAL_FEASIBLE"] == "false"
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.44"
    assert "CHATTERBOX_PORT" in env  # audio keys still appended, unaffected by shape


def test_default_shape_with_audio_is_unaffected(tmp_path, monkeypatch) -> None:
    """machine-as-brain (the default) hosts stt/tts too, exactly like every
    built-in shape — --audio remains the sole switch that scaffolds the
    overlay; passing both is harmless and changes nothing about the audio
    scaffold compared to --audio alone."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    with_shape = tmp_path / "with-shape"
    without_shape = tmp_path / "without-shape"
    assert main(["init", "--audio", str(without_shape), "--apply"]) == 0
    assert main(["init", "--shape", "machine-as-brain", "--audio", str(with_shape), "--apply"]) == 0
    assert (with_shape / ".env").read_text() == (without_shape / ".env").read_text()
    assert (with_shape / "docker-compose.audio.yml").read_text() == (
        without_shape / "docker-compose.audio.yml"
    ).read_text()


# --- dry-run: zero bytes on disk, --apply required -------------------------


def test_shape_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target)])
    assert rc == 0
    assert not target.exists()


def test_shape_dry_run_json_reports_shape_and_hosts(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape"] == "spark-lobe"
    assert set(payload["shape_hosts"]) == {"cortex", "embedder", "reranker", "stt", "tts"}
    assert not target.exists()


def test_shape_dry_run_text_names_the_shape(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Shape: spark-lobe" in out
    assert "Re-run with --apply to write." in out


# --- apply requires the flag; re-running with the previous shape restores
# the previous rendering byte-for-byte -------------------------------------


def test_apply_is_required_to_commit_a_shape(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target)]) == 0
    assert not target.exists()
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    assert target.exists()


def test_reapplying_the_same_shape_is_byte_for_byte_idempotent(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    first = (target / ".env").read_text()
    # Switch away to a different shape...
    assert main(["init", "--shape", "machine-as-brain", str(target), "--apply", "--force"]) == 0
    assert (target / ".env").read_text() != first  # sanity: it actually changed
    # ...then switch back to the original shape — byte-for-byte restore.
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply", "--force"]) == 0
    assert (target / ".env").read_text() == first


# --- t4b: the shape compose override — a dropped lobe must not RUN ----------
# The gap: rendering MULTIMODAL_FEASIBLE=false is honest, but the base compose
# keeps every core service unconditional, so `lobes fleet up` boots the dropped
# lobe anyway (proven live on the GB10). The override profile-disables it.


def test_spark_lobe_writes_shape_override_disabling_multimodal(tmp_path, monkeypatch) -> None:
    """Acceptance 1: the override parks vllm-multimodal in an inert profile and
    !resets the gateway depends_on to exclude it."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    override_path = target / _compose.SHAPE_OVERLAY
    assert override_path.is_file()
    doc = _load_shape_override(target)
    services = doc["services"]
    # vllm-multimodal (senses) is dropped → parked in the inert shape-dropped profile.
    assert services["vllm-multimodal"]["profiles"] == ["shape-dropped"]
    # cortex/embedder/reranker stay hosted → NOT in the override's service set.
    assert "vllm-primary" not in services
    assert "vllm-embed" not in services
    assert "vllm-rerank" not in services
    # The gateway's depends_on is cleared with the !reset tag (sentinel), so it no
    # longer references the now-profile-disabled vllm-multimodal.
    assert services["gateway"]["depends_on"] == {"__reset__": True}
    # The raw text carries the literal merge tag + a compose-version note.
    raw = override_path.read_text(encoding="utf-8")
    assert "depends_on: !reset" in raw
    assert "v2.24" in raw


def test_thor_lobe_shape_override_disables_primary(tmp_path, monkeypatch) -> None:
    """Acceptance 1 mirror: thor-lobe drops cortex → disables vllm-primary."""
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "thor-lobe", str(target), "--apply"]) == 0
    doc = _load_shape_override(target)
    services = doc["services"]
    assert services["vllm-primary"]["profiles"] == ["shape-dropped"]
    assert "vllm-multimodal" not in services
    assert services["gateway"]["depends_on"] == {"__reset__": True}


def test_shape_override_disabled_service_matches_render_api(tmp_path, monkeypatch) -> None:
    """The disabled service is DERIVED from the render API (ROLE_SERVICE over the
    dropped role), never hardcoded — the override's service set equals exactly the
    core services render_shape would NOT run."""
    from lobes.profiles.schema import ROLES
    from lobes.profiles.shape_render import ROLE_SERVICE
    from lobes.profiles.shapes import OPT_IN_CORE_ROLES

    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    shape = resolve_shape("spark-lobe")
    # Opt-in core roles (muse) are excluded: their services are already parked
    # behind their own compose profile, so the override never names them.
    expected_dropped = {
        ROLE_SERVICE[r] for r in ROLES if r not in OPT_IN_CORE_ROLES and not shape.hosts_role(r)
    }
    doc = _load_shape_override(target)
    disabled = {
        svc
        for svc, body in doc["services"].items()
        if isinstance(body, dict) and body.get("profiles") == ["shape-dropped"]
    }
    assert disabled == expected_dropped


def test_machine_as_brain_writes_no_shape_override(tmp_path, monkeypatch) -> None:
    """Acceptance 2: the whole-brain shape drops nothing → no override file at all."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "machine-as-brain", str(target), "--apply"]) == 0
    assert not (target / _compose.SHAPE_OVERLAY).exists()


def test_bare_init_writes_no_shape_override(tmp_path, monkeypatch) -> None:
    """Acceptance 2: a bare `lobes init` (no --shape) writes no override either."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    assert not (target / _compose.SHAPE_OVERLAY).exists()


def test_bare_scaffold_is_byte_identical_and_override_absent(tmp_path, monkeypatch) -> None:
    """Acceptance 2 (hard): the bare scaffold is byte-identical to explicit
    machine-as-brain, and NEITHER grows a shape override."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    bare = tmp_path / "bare"
    explicit = tmp_path / "explicit"
    assert main(["init", str(bare), "--apply"]) == 0
    assert main(["init", "--shape", "machine-as-brain", str(explicit), "--apply"]) == 0
    assert sorted(p.name for p in bare.iterdir()) == sorted(p.name for p in explicit.iterdir())
    assert not (bare / _compose.SHAPE_OVERLAY).exists()
    assert not (explicit / _compose.SHAPE_OVERLAY).exists()


def test_reinit_machine_as_brain_removes_stale_override(tmp_path, monkeypatch) -> None:
    """Acceptance 3: re-init to machine-as-brain over a mesh-shape scaffold REMOVES
    the stale override (else boot keeps skipping the re-hosted lobe)."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    assert (target / _compose.SHAPE_OVERLAY).is_file()
    assert main(["init", "--shape", "machine-as-brain", str(target), "--apply", "--force"]) == 0
    assert not (target / _compose.SHAPE_OVERLAY).exists()


def test_reinit_machine_as_brain_dry_run_reports_removal(tmp_path, monkeypatch, capsys) -> None:
    """Acceptance 3 (dry-run): the plan reports the stale override would be REMOVED,
    and leaves it untouched on disk."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    capsys.readouterr()  # drop the apply output
    rc = main(["init", "--shape", "machine-as-brain", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert _compose.SHAPE_OVERLAY in out
    assert "REMOVED" in out
    assert (target / _compose.SHAPE_OVERLAY).is_file()  # dry-run touched nothing


def test_shape_dry_run_reports_override_would_be_written(tmp_path, monkeypatch, capsys) -> None:
    """The dry-run plan mentions the override that would be written; zero bytes on disk."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert _compose.SHAPE_OVERLAY in out
    assert "vllm-multimodal" in out
    assert not target.exists()


def test_shape_dry_run_json_reports_override_plan(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape_override"]["action"] == "write"
    assert payload["shape_override"]["disables"] == ["vllm-multimodal"]
    assert payload["shape_override"]["file"] == _compose.SHAPE_OVERLAY
    assert not target.exists()


def test_shape_apply_json_reports_override_written(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "thor-lobe", str(target), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape_override"]["written"] is True
    assert payload["shape_override"]["disables"] == ["vllm-primary"]


def test_reapplying_shape_is_idempotent_including_override(tmp_path, monkeypatch) -> None:
    """Re-applying the same mesh shape restores the override byte-for-byte too."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    first = (target / _compose.SHAPE_OVERLAY).read_text()
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply", "--force"]) == 0
    assert (target / _compose.SHAPE_OVERLAY).read_text() == first
