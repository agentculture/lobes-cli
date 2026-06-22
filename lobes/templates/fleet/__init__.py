"""Packaged *fleet* templates (``lobes init --fleet``).

A subpackage so ``importlib.resources.files("lobes.templates.fleet")``
resolves; the files themselves are copied verbatim by
:func:`lobes.runtime._compose.write_scaffold`.
"""
