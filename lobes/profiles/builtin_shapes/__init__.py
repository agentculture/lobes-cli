"""Packaged built-in deployment shapes (TOML data, read via importlib.resources).

Each ``<name>.toml`` here is a :class:`~lobes.profiles.shapes.Shape` -- the
role subset one box hosts (the deployment-shape axis, orthogonal to the
per-machine :class:`~lobes.profiles.schema.Profile` in the sibling
``lobes.profiles.builtin`` package, which says how each hosted role is
tuned). :func:`lobes.profiles.shapes.load_builtin_shape` reads these at
runtime.

Three shapes ship:

* ``machine-as-brain.toml`` -- the default: one box hosts every role it can
  serve (all six -- cortex/senses/embedder/reranker + the stt/tts audio
  overlay). Carries no ``overrides``, ever.
* ``spark-lobe.toml`` -- the DGX Spark half of a mesh-brain deployment: keeps
  the Qwen ``cortex`` + ``embedder``/``reranker`` + audio, drops the Gemma
  ``senses`` lobe to a peer box.
* ``thor-lobe.toml`` -- the Jetson AGX Thor half: keeps Gemma ``senses`` +
  ``embedder``/``reranker`` + audio, drops the Qwen ``cortex`` lobe to a peer
  box.

``spark-lobe``/``thor-lobe`` differ from ``machine-as-brain`` ONLY in their
``hosts`` role subset (and, from a later task onward, a per-role budget
``overrides`` re-derivation of the lobe(s) that no longer share the box) --
there is no per-shape Python code anywhere; this package is pure data, same
convention as ``lobes.profiles.builtin``.
"""
