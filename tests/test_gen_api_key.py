"""Tests for the ``scripts/gen-api-key.py`` bearer-key generator.

The script is standalone (not a package module), so it is loaded by file path.
It is exercised against a tmp deployment dir — it never touches a real ``.env``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen-api-key.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_api_key", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load()


def test_sets_key_when_absent(tmp_path) -> None:
    (tmp_path / ".env").write_text("VLLM_PORT=8001\n", encoding="utf-8")
    assert gen.main(["--dir", str(tmp_path)]) == 0
    value = gen._read_key(tmp_path / ".env")
    assert value and value.startswith("mg-")
    # the unrelated line is preserved
    assert "VLLM_PORT=8001" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_refuses_overwrite_without_force(tmp_path, capsys) -> None:
    (tmp_path / ".env").write_text("CULTURE_VLLM_API_KEY=mg-existing\n", encoding="utf-8")
    assert gen.main(["--dir", str(tmp_path)]) == 1
    assert "already set" in capsys.readouterr().err
    assert gen._read_key(tmp_path / ".env") == "mg-existing"  # untouched


def test_force_rotates(tmp_path) -> None:
    (tmp_path / ".env").write_text("CULTURE_VLLM_API_KEY=mg-old\n", encoding="utf-8")
    assert gen.main(["--dir", str(tmp_path), "--force"]) == 0
    rotated = gen._read_key(tmp_path / ".env")
    assert rotated and rotated != "mg-old"


def test_missing_dir_is_env_error(tmp_path, capsys) -> None:
    assert gen.main(["--dir", str(tmp_path / "nope")]) == 2
    assert "not found" in capsys.readouterr().err


def test_show_prints_key_matching_env(tmp_path, capsys) -> None:
    (tmp_path / ".env").write_text("", encoding="utf-8")
    assert gen.main(["--dir", str(tmp_path), "--show"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("mg-")
    assert gen._read_key(tmp_path / ".env") == out


def test_secret_not_printed_by_default(tmp_path, capsys) -> None:
    (tmp_path / ".env").write_text("", encoding="utf-8")
    assert gen.main(["--dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    key = gen._read_key(tmp_path / ".env")
    assert key and key not in captured.out  # no leak to stdout without --show


def test_env_perms_are_owner_only(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    gen.main(["--dir", str(tmp_path)])
    assert (env.stat().st_mode & 0o777) == 0o600


def test_rejects_too_few_bytes(tmp_path, capsys) -> None:
    (tmp_path / ".env").write_text("", encoding="utf-8")
    assert gen.main(["--dir", str(tmp_path), "--bytes", "4"]) == 1
    assert "at least" in capsys.readouterr().err
    assert gen._read_key(tmp_path / ".env") is None  # nothing written


def test_non_regular_env_is_env_error(tmp_path, capsys) -> None:
    (tmp_path / ".env").mkdir()  # a directory where a regular .env is expected
    assert gen.main(["--dir", str(tmp_path)]) == 2
    assert "not a regular file" in capsys.readouterr().err
