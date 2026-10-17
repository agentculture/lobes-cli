"""t11: page-cache clear + start-counter reset at the _compose chokepoint."""

from __future__ import annotations

import types

import pytest

from lobes.cli import main
from lobes.runtime import _compose, _machine


@pytest.fixture
def rec(monkeypatch):
    events: list[str] = []
    result = {"ok": True}

    def fake_clear():
        events.append("clear")
        if result["ok"]:
            return _machine.MachineResult(True)
        return _machine.MachineResult(False, "not_granted")

    monkeypatch.setattr(_machine, "drop_caches", fake_clear)

    def fake_run(argv, **kw):
        events.append("docker " + " ".join(argv[2:]))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_compose, "_run", fake_run)
    monkeypatch.setattr(_compose, "_cleared_after_down", False)
    # the clear is only attempted on spark/thor/orin (review-b #10)
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: "spark")
    return events, result


def _deploy(tmp_path):
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  vllm-primary:\n    environment:\n      - MG_LOG_NAME=primary\n"
        "  vllm-hand:\n    environment:\n      - MG_LOG_NAME=hand\n",
        encoding="utf-8",
    )
    state = tmp_path / "logs" / "lane-state"
    state.mkdir(parents=True)
    for n in ("primary", "hand"):
        (state / f"{n}.starts").write_text("1\n", encoding="utf-8")
    return state


def test_down_then_clear(rec, tmp_path):
    events, _ = rec
    _compose.compose_down(tmp_path)
    assert events == ["docker down", "clear"]


def test_clear_before_up(rec, tmp_path):
    events, _ = rec
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]
    events.clear()
    _compose.compose_up_build(tmp_path)
    assert events == ["clear", "docker up -d --build"]


def test_switch_pair_clears_once(rec, tmp_path):
    events, _ = rec
    _compose.compose_down(tmp_path)
    _compose.compose_up_detached(tmp_path)
    assert events == ["docker down", "clear", "docker up -d"]


def test_run_compose_up_and_down(rec, tmp_path):
    events, _ = rec
    _compose.run_compose(tmp_path, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    assert events == ["clear", "docker up -d --no-deps vllm-hand"]
    events.clear()
    _compose.run_compose(tmp_path, ["docker", "compose", "stop", "vllm-hand"])
    assert events == ["docker stop vllm-hand"]  # stop never clears


def test_ungranted_skips_with_one_line_and_continues(rec, tmp_path, capsys):
    events, result = rec
    result["ok"] = False
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]
    err = capsys.readouterr().err
    assert err.count("page-cache clear skipped (not_granted)") == 1
    assert err.count("\n") == 1


def test_targeted_up_resets_only_that_lane(rec, tmp_path):
    state = _deploy(tmp_path)
    _compose.run_compose(tmp_path, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    assert not (state / "hand.starts").exists()
    assert (state / "primary.starts").exists()


def test_whole_project_up_resets_all(rec, tmp_path):
    state = _deploy(tmp_path)
    _compose.compose_up_detached(tmp_path)
    assert list(state.glob("*.starts")) == []


def test_log_dir_from_env_and_missing_ok(rec, tmp_path):
    (tmp_path / ".env").write_text("MODEL_GEAR_LOG_DIR=elsewhere\n", encoding="utf-8")
    d = tmp_path / "elsewhere" / "lane-state"
    d.mkdir(parents=True)
    (d / "x.starts").write_text("1\n", encoding="utf-8")
    _compose.compose_up_detached(tmp_path)
    assert not (d / "x.starts").exists()
    _compose.reset_start_counters(tmp_path / "nonexistent")  # no raise


def test_dry_run_shows_step_and_runs_nothing(rec, tmp_path, capsys):
    events, _ = rec
    _compose.write_scaffold(tmp_path, force=True)
    for argv in (["serve"], ["stop"]):
        assert main([*argv, "--compose-dir", str(tmp_path)]) == 0
        assert "page cache" in capsys.readouterr().out
    assert events == []


# --- review-b #10: card-gated clear + per-process detection cache ----------


@pytest.mark.parametrize("card", ["unknown", "dgx-station", None])
def test_clear_skipped_silently_off_supported_cards(rec, monkeypatch, capsys, card):
    events, _ = rec
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: card)
    probes: list[list[str]] = []
    monkeypatch.setattr(_machine, "_run", lambda argv, **k: probes.append(argv))
    assert _compose.clear_page_cache() is False
    assert events == [] and probes == []  # no drop_caches, no `sudo -n -l`
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("card", ["spark", "thor", "orin"])
def test_clear_attempted_on_supported_cards(rec, monkeypatch, card):
    events, _ = rec
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: card)
    assert _compose.clear_page_cache() is True
    assert events == ["clear"]


def test_card_detection_cached_per_process(monkeypatch):
    calls: list[int] = []

    def fake_detect():
        calls.append(1)
        return types.SimpleNamespace(resolved="orin")

    monkeypatch.setattr(_machine, "detect_card", fake_detect)
    assert [_machine.detected_card() for _ in range(5)] == ["orin"] * 5
    _machine._current_cli()  # resolves through the same cache
    assert len(calls) == 1


def test_card_detection_failure_is_none_and_rate_limited(monkeypatch):
    calls: list[int] = []
    now = {"t": 1000.0}
    monkeypatch.setattr(_machine, "_clock", lambda: now["t"])

    def boom():
        calls.append(1)
        raise RuntimeError("nvidia-smi exploded")

    monkeypatch.setattr(_machine, "detect_card", boom)
    assert _machine.detected_card() is None
    assert _machine.detected_card() is None
    assert len(calls) == 1  # inside the retry interval: no re-probe


def _flaky(results):
    calls: list[int] = []

    def detect():
        calls.append(1)
        r = results[min(len(calls), len(results)) - 1]
        if isinstance(r, Exception):
            raise r
        return types.SimpleNamespace(resolved=r)

    return detect, calls


@pytest.mark.parametrize("first", [RuntimeError("nvidia-smi not ready"), "unknown"])
def test_failed_detection_is_retried_after_the_interval(monkeypatch, first):
    """review2-b #6: a boot-time failure must not stick for the watcher's lifetime."""
    now = {"t": 1000.0}
    monkeypatch.setattr(_machine, "_clock", lambda: now["t"])
    detect, calls = _flaky([first, "thor"])
    monkeypatch.setattr(_machine, "detect_card", detect)
    assert _machine.detected_card() in (None, "unknown")
    now["t"] += _machine.DETECT_RETRY_S - 1
    assert _machine.detected_card() in (None, "unknown")
    assert len(calls) == 1
    now["t"] += 2
    assert _machine.detected_card() == "thor"
    now["t"] += 10 * _machine.DETECT_RETRY_S
    assert _machine.detected_card() == "thor"
    assert len(calls) == 2  # a success is cached for good


# --- review-b #3: no lifecycle clear while another lane is mid-boot --------


#: The project every 3-column ``_ps`` row is stamped with (see ``_project_env``).
_TEST_PROJECT = "lobes-test"


@pytest.fixture(autouse=True)
def _project_env(monkeypatch):
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", _TEST_PROJECT)


def _with_project(stdout: str) -> str:
    """Prefix a ``service\tname\tstatus`` row with the test project label."""
    rows = []
    for line in stdout.splitlines():
        rows.append(line if line.count("\t") >= 3 else f"{_TEST_PROJECT}\t{line}")
    return "".join(r + "\n" for r in rows)


def _ps(monkeypatch, stdout: str | None):
    calls: list[list[str]] = []

    def probe(argv, **k):
        calls.append(list(argv))
        if stdout is None:
            return None
        return types.SimpleNamespace(returncode=0, stdout=_with_project(stdout), stderr="")

    monkeypatch.setattr(_compose, "_probe", probe)
    return calls


def test_up_defers_clear_while_another_lane_starts(rec, tmp_path, monkeypatch, capsys):
    events, _ = rec
    _deploy(tmp_path)
    calls = _ps(
        monkeypatch,
        "vllm-primary\tmodel-gear-vllm-primary\tUp 2 minutes (health: starting)\n"
        "vllm-hand\tmodel-gear-vllm-hand\tUp 3 hours (healthy)\n",
    )
    _compose.run_compose(tmp_path, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    assert events == ["docker up -d --no-deps vllm-hand"]  # no clear, verb still runs
    assert calls == [_compose.lane_health_argv(tmp_path)]
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "deferred" in err and "primary" in err


def test_up_clears_when_no_lane_is_starting(rec, tmp_path, monkeypatch):
    events, _ = rec
    _deploy(tmp_path)
    _ps(monkeypatch, "vllm-primary\tmodel-gear-vllm-primary\tUp 2 hours (healthy)\n")
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]


def test_gated_lane_starting_does_not_defer_the_clear(rec, tmp_path, monkeypatch):
    events, _ = rec
    state = _deploy(tmp_path)
    (state / "primary.gated").write_text('{"lane": "primary"}', encoding="utf-8")
    _ps(monkeypatch, "vllm-primary\tmodel-gear-vllm-primary\tUp 2 minutes (health: starting)\n")
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]


def test_unreadable_docker_still_clears(rec, tmp_path, monkeypatch):
    events, _ = rec
    _ps(monkeypatch, None)
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]


def test_lane_health_maps_services_to_log_names(tmp_path, monkeypatch):
    _deploy(tmp_path)
    _ps(
        monkeypatch,
        "vllm-primary\tmodel-gear-vllm-primary\tUp 1 minute (health: starting)\n"
        "vllm-hand\tmodel-gear-vllm-hand\tUp 2 hours (unhealthy)\n"
        "\tmodel-gear-loose\tUp 5 seconds\n",
    )
    assert _compose.lane_health(tmp_path) == {
        "primary": "starting",
        "hand": "running",
        "model-gear-loose": "running",
    }


def test_lane_health_none_on_docker_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _compose,
        "_probe",
        lambda argv, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="denied"),
    )
    assert _compose.lane_health(tmp_path) is None


# --- review2-b #5: LOBES_PAGE_CACHE_CLEAR = auto | always | never ------------


@pytest.mark.parametrize("card", ["unknown", None])
def test_always_attempts_the_clear_on_an_undetected_card(rec, tmp_path, monkeypatch, card):
    events, _ = rec
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: card)
    monkeypatch.setenv("LOBES_PAGE_CACHE_CLEAR", "always")
    assert _compose.clear_page_cache(tmp_path) is True
    assert events == ["clear"]


def test_always_from_the_deployment_env(rec, tmp_path, monkeypatch):
    events, _ = rec
    monkeypatch.delenv("LOBES_PAGE_CACHE_CLEAR", raising=False)
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: "unknown")
    (tmp_path / ".env").write_text("LOBES_PAGE_CACHE_CLEAR=always\n", encoding="utf-8")
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]


@pytest.mark.parametrize("card", ["spark", "unknown"])
def test_never_skips_the_clear_everywhere(rec, tmp_path, monkeypatch, capsys, card):
    events, _ = rec
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: card)
    (tmp_path / ".env").write_text("LOBES_PAGE_CACHE_CLEAR=never\n", encoding="utf-8")
    monkeypatch.delenv("LOBES_PAGE_CACHE_CLEAR", raising=False)
    probes = _ps(monkeypatch, "")
    _compose.compose_up_detached(tmp_path)
    _compose.compose_down(tmp_path)
    assert events == ["docker up -d", "docker down"]
    assert probes == []  # no docker ps either
    assert capsys.readouterr().err == ""


def test_process_env_wins_over_the_deployment_env(rec, tmp_path, monkeypatch):
    events, _ = rec
    (tmp_path / ".env").write_text("LOBES_PAGE_CACHE_CLEAR=never\n", encoding="utf-8")
    monkeypatch.setenv("LOBES_PAGE_CACHE_CLEAR", "always")
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: "unknown")
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]


@pytest.mark.parametrize("value", ["auto", "", "bogus"])
def test_auto_and_unknown_values_keep_the_card_gate(rec, tmp_path, monkeypatch, capsys, value):
    events, _ = rec
    monkeypatch.setenv("LOBES_PAGE_CACHE_CLEAR", value)
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: "unknown")
    assert _compose.clear_page_cache(tmp_path) is False
    assert events == []
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: "orin")
    assert _compose.clear_page_cache(tmp_path) is True


def test_always_ungranted_on_an_undetected_card_names_the_sudo_rule(
    rec, tmp_path, monkeypatch, capsys
):
    events, result = rec
    result["ok"] = False
    monkeypatch.setenv("LOBES_PAGE_CACHE_CLEAR", "always")
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: "unknown")
    monkeypatch.setattr(_machine, "_current_cli", lambda: None)
    assert _compose.clear_page_cache(tmp_path) is False
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "vm.drop_caches" in err and "LOBES_PAGE_CACHE_CLEAR=never" in err


# --- review2-b #1: lanes are found by compose project, not by name prefix ----


def test_lane_health_argv_filters_by_the_compose_project_label(tmp_path, monkeypatch):
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    d = tmp_path / ".lobes"
    d.mkdir()
    argv = _compose.lane_health_argv(d)
    assert argv[:2] == ["docker", "ps"]
    assert "label=com.docker.compose.project" in argv
    assert not any(a.startswith("name=") for a in argv)


@pytest.mark.parametrize(
    "dirname,setup,expected",
    [
        (".lobes", {}, "lobes"),
        ("My Deploy.1", {}, "mydeploy1"),
        ("dep", {"dotenv": "COMPOSE_PROJECT_NAME=fromdotenv\n"}, "fromdotenv"),
        ("dep", {"name": "toplevel"}, "toplevel"),
        ("dep", {"name": "toplevel", "dotenv": "COMPOSE_PROJECT_NAME=wins\n"}, "wins"),
        ("dep", {"dotenv": "COMPOSE_PROJECT_NAME=dot\n", "environ": "Proc"}, "proc"),
    ],
)
def test_compose_project_name_resolution(tmp_path, monkeypatch, dirname, setup, expected):
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    d = tmp_path / dirname
    d.mkdir()
    body = "services: {}\n"
    if "name" in setup:
        body = f"name: {setup['name']}\n" + body
    (d / "docker-compose.yml").write_text(body, encoding="utf-8")
    if "dotenv" in setup:
        (d / ".env").write_text(setup["dotenv"], encoding="utf-8")
    if "environ" in setup:
        monkeypatch.setenv("COMPOSE_PROJECT_NAME", setup["environ"])
    assert _compose.compose_project_name(d) == expected


def test_lane_health_maps_embed_overlay_lanes(tmp_path, monkeypatch):
    """embed-* containers are not model-gear-* but run the memory gate too."""
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  vllm-primary:\n    environment:\n      - MG_LOG_NAME=primary\n",
        encoding="utf-8",
    )
    (tmp_path / _compose.EMBED_OVERLAY).write_text(
        "services:\n  embed-nemotron-embed:\n    container_name: embed-nemotron-embed\n"
        "    environment:\n"
        "      - MG_LOG_NAME=nemotron-embed   # -> /logs/model-gear/nemotron-embed-<boot>.log\n",
        encoding="utf-8",
    )
    calls = _ps(
        monkeypatch,
        "vllm-primary\tmodel-gear-vllm-primary\tUp 2 hours (healthy)\n"
        "embed-nemotron-embed\tembed-nemotron-embed\tUp 1 minute (health: starting)\n",
    )
    assert _compose.lane_health(tmp_path) == {"primary": "running", "nemotron-embed": "starting"}
    assert calls == [_compose.lane_health_argv(tmp_path)]


def test_a_booting_embed_lane_defers_the_clear(rec, tmp_path, monkeypatch, capsys):
    events, _ = rec
    _deploy(tmp_path)
    (tmp_path / _compose.EMBED_OVERLAY).write_text(
        "services:\n  embed-qwen3vl-embed:\n    environment:\n"
        "      - MG_LOG_NAME=qwen3vl-embed\n",
        encoding="utf-8",
    )
    _ps(monkeypatch, "embed-qwen3vl-embed\tembed-qwen3vl-embed\tUp 1 minute (health: starting)\n")
    _compose.compose_up_detached(tmp_path)
    assert "clear" not in events and events[-1].endswith("up -d")
    assert "qwen3vl-embed" in capsys.readouterr().err


# --- review2-b #2: lanes held before exec (.held) never defer the clear ------


def test_lanes_starting_excludes_held_lanes(tmp_path):
    state = _deploy(tmp_path)
    (state / "primary.held").write_text('{"lane": "primary", "reason": "boot_hold"}')
    assert _compose.held_lanes(tmp_path) == {"primary"}
    health = {"primary": "starting", "hand": "starting"}
    excluded = _compose.gated_lanes(tmp_path) | _compose.held_lanes(tmp_path)
    assert _compose.lanes_starting(health, excluded) == ["hand"]


def test_counter_held_lane_does_not_block_a_clear(rec, tmp_path, monkeypatch):
    events, _ = rec
    state = _deploy(tmp_path)
    (state / "primary.held").write_text(
        '{"lane": "primary", "since": 1, "reason": "start_counter"}', encoding="utf-8"
    )
    _ps(monkeypatch, "vllm-primary\tmodel-gear-vllm-primary\tUp 9 minutes (health: starting)\n")
    _compose.run_compose(tmp_path, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    assert events == ["clear", "docker up -d --no-deps vllm-hand"]


def test_a_held_lane_does_not_hide_another_booting_lane(rec, tmp_path, monkeypatch, capsys):
    events, _ = rec
    state = _deploy(tmp_path)
    (state / "primary.held").write_text('{"lane": "primary"}', encoding="utf-8")
    _ps(
        monkeypatch,
        "vllm-primary\tmodel-gear-vllm-primary\tUp 1 minute (health: starting)\n"
        "vllm-hand\tmodel-gear-vllm-hand\tUp 1 minute (health: starting)\n",
    )
    _compose.compose_up_detached(tmp_path)
    assert events == ["docker up -d"]
    err = capsys.readouterr().err
    assert "hand" in err and "primary" not in err.split("starting:")[1]


# --- review3 #2: a top-level ``name:`` is interpolated, never taken raw ------


@pytest.mark.parametrize(
    "name_line,environ,dotenv,expected",
    [
        ("name: ${LOBES_PROJECT:-lobes}", {}, "", "lobes"),
        ("name: ${LOBES_PROJECT:-lobes}", {"LOBES_PROJECT": "Prod"}, "", "prod"),
        ("name: ${LOBES_PROJECT:-lobes}", {}, "LOBES_PROJECT=fromenv\n", "fromenv"),
        ("name: ${LOBES_PROJECT:-lobes}", {"LOBES_PROJECT": ""}, "", "lobes"),
        ("name: ${LOBES_PROJECT-dflt}", {}, "", "dflt"),
        ("name: ${LOBES_PROJECT}", {"LOBES_PROJECT": "plain"}, "", "plain"),
        ('name: "quoted"', {}, "", "quoted"),
        ("name: 'single'  # comment", {}, "", "single"),
        ("name: pre-${SUFFIX:-x}", {}, "", "pre-x"),
        # unresolvable -> the name: candidate is ignored, the basename wins
        ("name: ${UNSET_VAR}", {}, "", "dep"),
        ("name: ${UNSET_VAR:?boom}", {}, "", "dep"),
    ],
)
def test_top_level_name_is_interpolated(
    tmp_path, monkeypatch, capsys, name_line, environ, dotenv, expected
):
    for key in ("COMPOSE_PROJECT_NAME", "LOBES_PROJECT", "SUFFIX", "UNSET_VAR"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    d = tmp_path / "dep"
    d.mkdir()
    (d / "docker-compose.yml").write_text(f"{name_line}\nservices: {{}}\n", encoding="utf-8")
    if dotenv:
        (d / ".env").write_text(dotenv, encoding="utf-8")
    assert _compose.compose_project_name(d) == expected
    assert capsys.readouterr().err == ""  # an unresolvable name logs nothing


# --- review3 #1: lane health matches a CANDIDATE set of project names --------


def _symlinked(tmp_path, real_name: str, link_name: str):
    real = tmp_path / "data" / real_name
    real.mkdir(parents=True)
    (real / "docker-compose.yml").write_text(
        "services:\n  vllm-primary:\n    environment:\n      - MG_LOG_NAME=primary\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    link = home / link_name
    link.symlink_to(real, target_is_directory=True)
    return real, link


def test_lane_health_argv_lists_every_compose_project_with_its_label():
    argv = _compose.LANE_HEALTH_ARGV
    assert argv[:2] == ["docker", "ps"]
    assert argv[argv.index("--filter") + 1] == "label=com.docker.compose.project"
    fmt = argv[argv.index("--format") + 1]
    assert fmt.startswith('{{.Label "com.docker.compose.project"}}\t')


@pytest.mark.parametrize(
    "real_name,link_name,expected",
    [
        # ~/.lobes -> /data/lobes-prod: the CLI (via the link) and compose run with
        # a physical cwd (via the target) disagree on the basename.
        ("lobes-prod", ".lobes", {"lobes", "lobes-prod"}),
        # the other order: the real dir is .lobes, reached through a link "prod"
        (".lobes", "prod", {"prod", "lobes"}),
    ],
)
def test_project_candidates_cover_both_symlink_names(
    tmp_path, monkeypatch, real_name, link_name, expected
):
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    _real, link = _symlinked(tmp_path, real_name, link_name)
    assert _compose.compose_project_candidates(link) == frozenset(expected)


@pytest.mark.parametrize("stamped", ["lobes", "lobes-prod"])
@pytest.mark.parametrize("via", ["link", "watcher-unit"])
def test_lane_health_sees_lanes_whichever_symlink_name_compose_stamped(
    tmp_path, monkeypatch, stamped, via
):
    """~/.lobes -> /data/lobes-prod: whichever name compose stamped, both the CLI
    (deploy dir = the link) and the watcher (the unit's --compose-dir) find the lane."""
    from lobes.watcher import _install

    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    _real, link = _symlinked(tmp_path, "lobes-prod", ".lobes")
    deploy = link
    if via == "watcher-unit":
        ctx = _install.InstallContext(user="op", home=tmp_path, exec_argv=["lobes"])
        unit = _install.render_unit(ctx, link, system=False)
        exec_line = next(ln for ln in unit.splitlines() if ln.startswith("ExecStart="))
        args = exec_line.split()
        deploy = args[args.index("--compose-dir") + 1]
        assert deploy == str(link)  # absolute, symlink kept: both names derivable
    _ps(
        monkeypatch,
        f"{stamped}\tvllm-primary\tmodel-gear-vllm-primary\tUp 1 minute (health: starting)\n"
        "other\tvllm-primary\tother-vllm-primary\tUp 2 hours (healthy)\n",
    )
    assert _compose.lane_health(deploy) == {"primary": "starting"}


def test_lane_health_drops_rows_of_unrelated_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "mine")
    _deploy(tmp_path)
    _ps(
        monkeypatch,
        "mine\tvllm-primary\tmodel-gear-vllm-primary\tUp 1 minute (health: starting)\n"
        "theirs\tvllm-hand\ttheirs-vllm-hand\tUp 1 minute (health: starting)\n",
    )
    assert _compose.lane_health(tmp_path) == {"primary": "starting"}


def test_project_candidates_include_env_dotenv_and_name(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "Proc")
    d = tmp_path / "dep"
    d.mkdir()
    (d / ".env").write_text("COMPOSE_PROJECT_NAME=dot\n", encoding="utf-8")
    (d / "docker-compose.yml").write_text("name: top\nservices: {}\n", encoding="utf-8")
    assert _compose.compose_project_candidates(d) == frozenset({"proc", "dot", "top", "dep"})
