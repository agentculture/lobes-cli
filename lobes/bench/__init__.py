"""lobes.bench — per-lobe benchmark suite (perf engine, cat probe, scorer, report).

Stdlib-only, read-only building blocks for ``lobes benchmark --all-lobes``,
``lobes benchmark --profile`` (issue #81, t9), and ``lobes eval cat --score
logprobs``:

* :mod:`lobes.bench.cat_probe`  — timestamped 'Where is the cat?' generator.
* :mod:`lobes.bench.cat_score`  — logprobs soft-score over candidate locations.
* :mod:`lobes.bench.report`     — markdown report renderers: the fixed
  minor/primary table (:func:`~lobes.bench.report.render_report`) and the
  general N-column sibling (:func:`~lobes.bench.report.render_side_by_side`).
* :mod:`lobes.bench.compare`    — the RUNTIME-ONLY profile-comparison mode
  (``cortex-only`` / ``cortex+senses`` / ``senses-direct`` /
  ``qwen-nvfp4-vs-bf16``), built on :mod:`lobes.roles_measure`.

Import submodules by their full path (e.g. ``from lobes.bench.cat_probe import
generate_case``); this package ``__init__`` intentionally stays a thin namespace
so independent modules never contend on it.
"""
