"""lobes.bench — per-lobe benchmark suite (perf engine, cat probe, scorer, report).

Stdlib-only, read-only building blocks for ``lobes benchmark --all-lobes`` and
``lobes eval cat --score logprobs``:

* :mod:`lobes.bench.cat_probe`  — timestamped 'Where is the cat?' generator.
* :mod:`lobes.bench.cat_score`  — logprobs soft-score over candidate locations.
* :mod:`lobes.bench.report`     — combined per-lobe markdown report renderer.

Import submodules by their full path (e.g. ``from lobes.bench.cat_probe import
generate_case``); this package ``__init__`` intentionally stays a thin namespace
so independent modules never contend on it.
"""
