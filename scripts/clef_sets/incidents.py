"""Synthetic incident-log records with objective answers (mesh-authored, seeded).

Each record is a service log in which a known set of services fails at known
times. Three questions per record, all with ground truth BY CONSTRUCTION:

- ``failing`` (score): how many distinct services fail (0, 1, 2, 3, 4+);
- ``database_down`` (noul): does the ``postgres`` service fail;
- ``first_failure`` (choice, only when at least one service fails): which
  service fails first.

Filler lines (healthy heartbeats) vary the state length from a few hundred to
several thousand tokens, so the set also exercises long prefill. Pure stdlib;
the same seed always yields the same record.
"""

from __future__ import annotations

import random

SERVICES = ("gateway", "postgres", "redis", "scheduler", "auth", "billing", "search", "mailer")
FAILURE_TEXT = (
    "ERROR connection refused; marking service DOWN",
    "FATAL out of memory; process exited with code 137",
    "ERROR health check failed 3 times; service DOWN",
    "CRITICAL disk full; refusing writes; service DOWN",
)
HEALTHY_TEXT = (
    "INFO heartbeat ok latency_ms={ms}",
    "INFO handled {n} requests in the last minute",
    "DEBUG cache hit ratio {r}",
)
COUNT_LEVELS = [
    "no service fails",
    "one service fails",
    "two services fail",
    "three services fail",
    "four or more services fail",
]


def _ts(minute: int, second: int) -> str:
    return f"2026-10-14T03:{minute:02d}:{second:02d}Z"


def incident_record(seed: int) -> tuple[dict, dict]:
    """Return ``(record, gold)`` for one seed. ``gold`` maps question id -> option id."""
    rng = random.Random(seed)
    n_fail = rng.choice((0, 1, 1, 2, 2, 3, 4, 5))
    services = list(SERVICES)
    failing = rng.sample(services, n_fail)
    fail_minute = {svc: rng.randrange(5, 55) for svc in failing}
    # Distinct failure minutes keep "first" unambiguous.
    used: set[int] = set()
    for svc in failing:
        while fail_minute[svc] in used:
            fail_minute[svc] = rng.randrange(5, 55)
        used.add(fail_minute[svc])
    filler = rng.choice((10, 40, 120, 300))
    lines: list[tuple[int, int, str]] = []
    for _ in range(filler):
        svc = rng.choice(services)
        text = rng.choice(HEALTHY_TEXT).format(
            ms=rng.randrange(2, 90), n=rng.randrange(10, 900), r=round(rng.random(), 2)
        )
        lines.append((rng.randrange(0, 60), rng.randrange(0, 60), f"[{svc}] {text}"))
    for svc in failing:
        lines.append(
            (fail_minute[svc], rng.randrange(0, 60), f"[{svc}] {rng.choice(FAILURE_TEXT)}")
        )
    lines.sort()
    state = "Service log excerpt (UTC):\n" + "\n".join(
        f"{_ts(m, s)} {text}" for m, s, text in lines
    )
    questions: dict = {
        "failing": {
            "type": "score",
            "instructions": "How many distinct services are marked down or exit in this log?",
            "criteria": COUNT_LEVELS,
        },
        "database_down": {
            "type": "noul",
            "instructions": "Does the postgres service fail in this log?",
        },
    }
    gold = {
        "failing": str(min(n_fail, 4)),
        "database_down": "true" if "postgres" in failing else "false",
    }
    if failing:
        questions["first_failure"] = {
            "type": "choice",
            "instructions": "Which service fails first?",
            "criteria": {svc: f"the {svc} service" for svc in SERVICES},
        }
        gold["first_failure"] = min(failing, key=fail_minute.__getitem__)
    return {"model": "clef", "state": state, "questions": questions}, gold
