# flake8: noqa: E501
"""Build the Clef calibration set and the disjoint labelled probe set (plan orin-decision-models, t18).

Two sets of Clef/SystemOne-format records (``{"model", "state", "questions"}``):

- **calibration** (>= 256 records, train splits + seeded synthetic) — feeds the
  GPTQ/NVFP4 quantization runs (t19/t20) and is NEVER used for evaluation;
- **probe** (>= 100 records with ground-truth answers, eval splits + disjoint
  synthetic seeds + hand-written routing) — what t11/t13/t23 measure top-1
  accuracy and agreement with bf16 on.

Third-party corpora are pinned by dataset id + revision and are NOT re-hosted:
the committed manifests carry each record's source row and gold answer, and
``materialize`` rebuilds the full records from the pinned parquet files.
Mesh-authored records (``incidents.py``, ``routing.py``) are fully reproducible
from this directory.

Usage (needs network + ``huggingface_hub`` and ``pyarrow``; not part of the package):

    uv run --no-project --with huggingface_hub --with pyarrow python scripts/clef_sets/build_sets.py manifest
    uv run --no-project --with huggingface_hub --with pyarrow python scripts/clef_sets/build_sets.py materialize OUT_DIR
    python3 scripts/clef_sets/build_sets.py check          # offline; what the test runs
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from incidents import incident_record  # noqa: E402
from routing import REQUESTS, routing_record  # noqa: E402

SETS = ("calibration", "probe")
MIN_COUNTS = {"calibration": 256, "probe": 100}
SELECTION_SEED = "clef-sets-v1"
INCIDENT_SEED_BASE = {"calibration": 2_000_000, "probe": 1_000_000}

# source -> pinned upstream, licence, the split used per set, and how many records per set.
SOURCES = {
    "boolq": {
        "repo": "google/boolq",
        "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        "license": "cc-by-sa-3.0",
        "files": {
            "calibration": "data/train-00000-of-00001.parquet",
            "probe": "data/validation-00000-of-00001.parquet",
        },
        "counts": {"calibration": 90, "probe": 50},
    },
    "snli": {
        "repo": "stanfordnlp/snli",
        "revision": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
        "license": "cc-by-sa-4.0",
        "files": {
            "calibration": "plain_text/train-00000-of-00001.parquet",
            "probe": "plain_text/validation-00000-of-00001.parquet",
        },
        "counts": {"calibration": 90, "probe": 50},
    },
    "banking77": {
        "repo": "legacy-datasets/banking77",
        "revision": "f54121560de48f2852f90be299010d1d6dc612ec",
        "license": "cc-by-4.0",
        "files": {
            "calibration": "data/train-00000-of-00001.parquet",
            "probe": "data/test-00000-of-00001.parquet",
        },
        "counts": {"calibration": 80, "probe": 40},
    },
    "commonsense_qa": {
        "repo": "tau/commonsense_qa",
        "revision": "94630fe30dad47192a8546eb75f094926d47e155",
        "license": "mit",
        "files": {
            "calibration": "data/train-00000-of-00001.parquet",
            "probe": "data/validation-00000-of-00001.parquet",
        },
        "counts": {"calibration": 70, "probe": 40},
    },
    "arc_challenge": {
        "repo": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "license": "cc-by-sa-4.0",
        "files": {
            "calibration": "ARC-Challenge/train-00000-of-00001.parquet",
            "probe": "ARC-Challenge/test-00000-of-00001.parquet",
        },
        "counts": {"calibration": 70, "probe": 40},
    },
    "go_emotions": {
        "repo": "google-research-datasets/go_emotions",
        "revision": "add492243ff905527e67aeb8b80c082af02207c3",
        "license": "apache-2.0",
        "files": {
            "calibration": "simplified/train-00000-of-00001.parquet",
            "probe": "simplified/validation-00000-of-00001.parquet",
        },
        "counts": {"calibration": 50, "probe": 30},
    },
}
MESH_SOURCES = {
    "incident": {
        "license": "apache-2.0 (mesh-authored, scripts/clef_sets/incidents.py)",
        "counts": {"calibration": 80, "probe": 50},
    },
    "routing": {
        "license": "apache-2.0 (mesh-authored, scripts/clef_sets/routing.py)",
        "counts": {"calibration": 0, "probe": len(REQUESTS)},
    },
}
DISTRACTORS = 4  # sampled wrong options for the many-label sources (banking77, go_emotions)
NLI = ("entailment", "neutral", "contradiction")


def record_sha(record: dict) -> str:
    canon = json.dumps(
        {"state": record["state"], "questions": record["questions"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canon.encode()).hexdigest()


def manifest_path(name: str) -> Path:
    return HERE / f"{name}.manifest.jsonl"


# --- row -> record converters (pure; ``names`` are the dataset's ClassLabel names) -------------


def _boolq(
    row: dict, _names: dict, _rng: random.Random, _opts: list | None
) -> tuple[dict, dict, list | None]:
    q = row["question"].strip()
    record = {
        "model": "clef",
        "state": {"passage": row["passage"]},
        "questions": {"answer": {"type": "noul", "instructions": q[0].upper() + q[1:] + "?"}},
    }
    return record, {"answer": "true" if row["answer"] else "false"}, None


def _snli(
    row: dict, _names: dict, _rng: random.Random, _opts: list | None
) -> tuple[dict, dict, list | None]:
    record = {
        "model": "clef",
        "state": {"premise": row["premise"], "hypothesis": row["hypothesis"]},
        "questions": {
            "relation": {
                "type": "choice",
                "instructions": "How does the premise relate to the hypothesis?",
                "criteria": {
                    "entailment": "The premise implies the hypothesis is true.",
                    "neutral": "The premise neither implies nor contradicts the hypothesis.",
                    "contradiction": "The premise contradicts the hypothesis.",
                },
            }
        },
    }
    return record, {"relation": NLI[row["label"]]}, None


def _sampled(gold: str, pool: list[str], rng: random.Random, opts: list | None) -> list[str]:
    if opts is not None:
        return opts
    return sorted([gold, *rng.sample([p for p in pool if p != gold], DISTRACTORS)])


def _banking77(
    row: dict, names: dict, rng: random.Random, opts: list | None
) -> tuple[dict, dict, list | None]:
    labels = names["label"]
    gold = labels[row["label"]]
    options = _sampled(gold, labels, rng, opts)
    record = {
        "model": "clef",
        "state": f"Customer message to a bank: {row['text']}",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Which intent does the customer message express?",
                "criteria": {o: o.replace("_", " ") for o in options},
            }
        },
    }
    return record, {"intent": gold}, options


def _mcq(
    row: dict, _names: dict, _rng: random.Random, _opts: list | None
) -> tuple[dict, dict, list | None]:
    choices = row["choices"]
    record = {
        "model": "clef",
        "state": {"question": row["question"]},
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Which option answers the question best?",
                "criteria": dict(zip(choices["label"], choices["text"])),
            }
        },
    }
    return record, {"answer": row["answerKey"]}, None


def _go_emotions(
    row: dict, names: dict, rng: random.Random, opts: list | None
) -> tuple[dict, dict, list | None]:
    labels = names["labels"]
    gold = labels[row["labels"][0]]
    options = _sampled(gold, [n for n in labels if n != "neutral"], rng, opts)
    record = {
        "model": "clef",
        "state": f"Comment: {row['text']}",
        "questions": {
            "emotion": {
                "type": "choice",
                "instructions": "Which emotion does the comment express most clearly?",
                "criteria": {o: o for o in options},
            }
        },
    }
    return record, {"emotion": gold}, options


CONVERT = {
    "boolq": _boolq,
    "snli": _snli,
    "banking77": _banking77,
    "commonsense_qa": _mcq,
    "arc_challenge": _mcq,
    "go_emotions": _go_emotions,
}


def _usable(source: str, row: dict) -> bool:
    if source == "snli":
        return row["label"] in (0, 1, 2)
    if source == "go_emotions":
        return len(row["labels"]) == 1 and row["labels"][0] != 27  # single label, not 'neutral'
    if source in ("commonsense_qa", "arc_challenge"):
        return bool(row.get("answerKey")) and row["answerKey"] in list(row["choices"]["label"])
    return True


# --- pinned parquet access (network) ----------------------------------------------------------


def _load(source: str, set_name: str) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    spec = SOURCES[source]
    path = hf_hub_download(
        spec["repo"], spec["files"][set_name], repo_type="dataset", revision=spec["revision"]
    )
    table = pq.read_table(path)
    names: dict = {}
    meta = (table.schema.metadata or {}).get(b"huggingface")
    if meta:
        feats = json.loads(meta).get("info", {}).get("features", {})
        for col, feat in feats.items():
            if isinstance(feat, dict) and "names" in feat:
                names[col] = feat["names"]
            elif (
                isinstance(feat, dict)
                and isinstance(feat.get("feature"), dict)
                and "names" in feat["feature"]
            ):
                names[col] = feat["feature"]["names"]
            elif (
                isinstance(feat, list) and feat and isinstance(feat[0], dict) and "names" in feat[0]
            ):
                names[col] = feat[0]["names"]
    return table.to_pylist(), names


# --- manifest / materialize / check -----------------------------------------------------------


def _mesh_entries(set_name: str) -> list[tuple[dict, dict, dict]]:
    out = []
    for i in range(MESH_SOURCES["incident"]["counts"][set_name]):
        seed = INCIDENT_SEED_BASE[set_name] + i
        record, gold = incident_record(seed)
        out.append(({"id": f"incident:{seed}", "source": "incident", "seed": seed}, record, gold))
    for i in range(MESH_SOURCES["routing"]["counts"][set_name]):
        record, gold = routing_record(i)
        out.append(({"id": f"routing:{i}", "source": "routing", "index": i}, record, gold))
    return out


def build_manifests() -> None:
    rng = random.Random(SELECTION_SEED)
    built: dict[str, list[dict]] = {}
    for set_name in SETS:  # calibration first, so probe can drop any record identical to one
        entries = []
        for source in SOURCES:
            rows, names = _load(source, set_name)
            want = SOURCES[source]["counts"][set_name]
            candidates = [i for i, r in enumerate(rows) if _usable(source, r)]
            rng.shuffle(candidates)
            taken = 0
            for row_idx in candidates:
                if taken == want:
                    break
                record, gold, options = CONVERT[source](rows[row_idx], names, rng, None)
                sha = record_sha(record)
                if set_name == "probe" and sha in {e["sha256"] for e in built["calibration"]}:
                    continue
                entry = {
                    "id": f"{source}:{SOURCES[source]['files'][set_name]}:{row_idx}",
                    "source": source,
                    "row": row_idx,
                    "gold": gold,
                    "sha256": sha,
                }
                if options is not None:
                    entry["options"] = options
                entries.append(entry)
                taken += 1
        for head, record, gold in _mesh_entries(set_name):
            entries.append({**head, "gold": gold, "sha256": record_sha(record)})
        built[set_name] = entries
    for set_name, entries in built.items():
        manifest_path(set_name).write_text(
            "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries)
        )
        print(f"{set_name}: {len(entries)} records -> {manifest_path(set_name).name}")


def read_manifest(set_name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in manifest_path(set_name).read_text().splitlines()
        if line.strip()
    ]


def materialize(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for set_name in SETS:
        cache: dict[str, tuple[list[dict], dict]] = {}
        mesh = {head["id"]: (record, gold) for head, record, gold in _mesh_entries(set_name)}
        lines = []
        for entry in read_manifest(set_name):
            if entry["source"] in MESH_SOURCES:
                record, gold = mesh[entry["id"]]
            else:
                if entry["source"] not in cache:
                    cache[entry["source"]] = _load(entry["source"], set_name)
                rows, names = cache[entry["source"]]
                record, gold, _ = CONVERT[entry["source"]](
                    rows[entry["row"]], names, random.Random(0), entry.get("options")
                )
            if record_sha(record) != entry["sha256"] or gold != entry["gold"]:
                raise SystemExit(
                    f"{entry['id']}: rebuilt record does not match the manifest (pinned revision changed?)"
                )
            lines.append(
                json.dumps(
                    {"id": entry["id"], "source": entry["source"], "record": record, "gold": gold},
                    ensure_ascii=False,
                )
                + "\n"
            )
        (out_dir / f"{set_name}.jsonl").write_text("".join(lines))
        print(f"{set_name}: {len(lines)} records -> {out_dir / (set_name + '.jsonl')}")


def check() -> list[str]:
    """Offline invariants over the committed manifests. Returns a list of problems (empty = ok)."""
    problems: list[str] = []
    sets = {s: read_manifest(s) for s in SETS}
    for s, entries in sets.items():
        if len(entries) < MIN_COUNTS[s]:
            problems.append(f"{s}: {len(entries)} records < {MIN_COUNTS[s]}")
        for e in entries:
            if e["source"] not in SOURCES and e["source"] not in MESH_SOURCES:
                problems.append(f"{e['id']}: source has no pinned citation")
            if not e.get("gold"):
                problems.append(f"{e['id']}: no gold answer")
    ids = {s: {e["id"] for e in sets[s]} for s in SETS}
    shas = {s: {e["sha256"] for e in sets[s]} for s in SETS}
    if ids["calibration"] & ids["probe"]:
        problems.append(f"record ids overlap: {sorted(ids['calibration'] & ids['probe'])[:5]}")
    if shas["calibration"] & shas["probe"]:
        problems.append("record contents overlap between calibration and probe")
    for s in SETS:
        for e in sets[s]:
            if e["source"] in SOURCES and SOURCES[e["source"]]["files"][s] not in e["id"]:
                problems.append(f"{e['id']}: drawn from the wrong split for {s}")
    # mesh-authored records must rebuild bit-identically (no network needed)
    for s in SETS:
        mesh = {head["id"]: (record, gold) for head, record, gold in _mesh_entries(s)}
        for e in sets[s]:
            if e["source"] in MESH_SOURCES:
                record, gold = mesh.get(e["id"], (None, None))
                if record is None or record_sha(record) != e["sha256"] or gold != e["gold"]:
                    problems.append(f"{e['id']}: mesh-authored record does not rebuild")
    return problems


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "check"
    if cmd == "manifest":
        build_manifests()
    elif cmd == "materialize":
        materialize(Path(argv[1]))
    elif cmd == "check":
        problems = check()
        for p in problems:
            print("FAIL", p)
        if problems:
            return 1
        print("ok:", {s: len(read_manifest(s)) for s in SETS})
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
