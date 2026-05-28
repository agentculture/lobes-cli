"""Packaged *fleet* templates (``model init --fleet``).

A subpackage so ``importlib.resources.files("model_gear.templates.fleet")``
resolves; the files themselves are copied verbatim by
:func:`model_gear.runtime._compose.write_scaffold`.
"""
