# flake8: noqa: E501
"""Hand-written lane-routing records (mesh-authored, 2026-10-14) — the `lobes route` consumer.

Labels are the author's judgment on deliberately clear-cut requests, not an
objective ground truth; the evaluation reports this source separately.
"""

from __future__ import annotations

LANES = {
    "hand": "a tiny, cheap specialist model for short routine tasks",
    "worker": "a fast doer for bulk code edits, ground work and running commands",
    "cortex": "the large reasoning model for hard, high-stakes decisions",
    "embedder": "a text embedding model that turns text into vectors for retrieval",
}

REQUESTS = (
    ("Convert this date to ISO format: March 3rd 2026.", "hand"),
    ("Say thanks to the user in one short friendly sentence.", "hand"),
    ("Uppercase the word 'ready'.", "hand"),
    ("Extract the email address from: contact me at a@b.org tomorrow.", "hand"),
    ("Is 17 an even number? Answer yes or no.", "hand"),
    ("Rename the variable cfg to config in all 40 files under lobes/ and run the tests.", "worker"),
    ("Apply black formatting to the whole repository and commit the result.", "worker"),
    ("Add a type hint to every function in lobes/runtime/ that lacks one.", "worker"),
    ("Run the test suite, collect every failing test name and paste the list.", "worker"),
    ("Grep the repo for TODO comments and list them with file and line.", "worker"),
    (
        "Production is down after the deploy; decide whether to roll back or hot-fix, weighing data loss risk across three services.",
        "cortex",
    ),
    (
        "Review this database migration plan and decide if it is safe to run on the live cluster tonight.",
        "cortex",
    ),
    (
        "Two senior engineers disagree on the sharding design; analyse both proposals and make the final call.",
        "cortex",
    ),
    (
        "Prove or refute that this lock-free queue is linearizable under concurrent producers.",
        "cortex",
    ),
    (
        "Decide whether we should accept the vendor contract given these conflicting legal and cost constraints.",
        "cortex",
    ),
    (
        "Turn these 10,000 support tickets into vectors so we can find similar ones later.",
        "embedder",
    ),
    ("Index the documentation pages for semantic search.", "embedder"),
    ("Compute a vector for each product description for nearest-neighbour lookup.", "embedder"),
    ("Embed every commit message so we can cluster them by topic.", "embedder"),
    ("Produce dense vectors for these 500 FAQ answers to power retrieval.", "embedder"),
)


def routing_record(index: int) -> tuple[dict, dict]:
    text, lane = REQUESTS[index]
    record = {
        "model": "clef",
        "state": f"User request: {text}",
        "questions": {
            "lane": {
                "type": "choice",
                "instructions": "Which fleet lane should handle this request?",
                "criteria": dict(LANES),
            }
        },
    }
    return record, {"lane": lane}
