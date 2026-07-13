"""Packaged built-in machine profiles (TOML data, read via importlib.resources).

Each ``<name>.toml`` here is a :class:`~lobes.profiles.schema.Profile` for one
supported card; :mod:`lobes.profiles.loader` reads them at runtime. This
package ships **TOML**, not the YAML the build plan sketched: the repo is
stdlib-only (``pyproject.toml`` ``dependencies = []``, no YAML parser), and
``requires-python = ">=3.12"`` guarantees :mod:`tomllib` — so TOML is a
zero-new-dependency stand-in for the same substance (structured, per-role
knob data), not a functional change. See ``spark.toml`` (the default,
reproducing today's fleet template values byte-for-byte) and ``thor.toml``
(the four Jetson AGX Thor sm_110 divergences validated live — the divergent
knob VALUES are not re-typed here; ``lobes.profiles.loader`` overlays them
from the :mod:`lobes.machines` chip-strategy registry so they stay
single-sourced).
"""
