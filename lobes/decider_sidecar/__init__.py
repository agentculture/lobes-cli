"""Decider sidecar — serves one decision checkpoint on ``POST /v1/systemone``.

The backend-agnostic HTTP core lives in :mod:`lobes.decider_sidecar.server` and
is unit-tested with a fake backend; the Clef / strands-decider loaders and the
FastAPI shell only run inside the sidecar container.
"""
