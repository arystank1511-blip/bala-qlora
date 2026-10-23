"""CPU-only checks shared by training and evaluation."""

import hashlib
import json
from pathlib import Path

from .topics import topic_group


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_reviewed(row: dict) -> bool:
    source = row.get("source", "").split("+")
    review = row.get("review", {})
    if "ai-review" in source:
        return (review.get("kind") == "ai" and bool(review.get("reviewer", "").strip())
                and review.get("decision") in {"keep", "fix"})
    return "human-review" in source


def make_manifest(root: Path, seed: int, val: float, test: float) -> dict:
    paths = ["data/seed_examples.jsonl", "data/reviewed.jsonl"]
    paths += [f"data/splits/{part}.jsonl" for part in ("train", "val", "test")]
    return {
        "version": 1, "method": "bilingual-topic-holdout", "seed": seed,
        "val_target": val, "test_target": test,
        "sha256": {path: sha256(root / path) for path in paths},
        "prompt_sha256": sha256(Path(__file__).with_name("prompts.py")),
        "topics_sha256": sha256(Path(__file__).with_name("topics.py")),
    }


def validate_splits(root: Path) -> dict[str, list[dict]]:
    """Reject missing, stale, overlapping, or unreviewed data before GPU work."""
    manifest_path = root / "data/splits/manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = make_manifest(root, manifest["seed"], manifest["val_target"], manifest["test_target"])
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("Missing or invalid dataset manifest; run scripts/prepare_data.py after review") from error
    if manifest != current:
        raise ValueError("Dataset, prompts or split files changed; run scripts/prepare_data.py again")

    parts = {name: read_jsonl(root / f"data/splits/{name}.jsonl") for name in ("train", "val", "test")}
    ids, topics = set(), {}
    for name, rows in parts.items():
        if not rows or {r["lang"] for r in rows} != {"kk", "ru"}:
            raise ValueError(f"{name} must contain reviewed examples in both languages")
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"Duplicate id across splits: {row['id']}")
            ids.add(row["id"])
            is_seed = row["id"].startswith("seed-")
            if is_seed and name != "train":
                raise ValueError("Drafting seed examples must remain in training")
            if not is_seed and not is_reviewed(row):
                raise ValueError(f"Unreviewed example: {row['id']}")
            group = topic_group(row)
            if group in topics and topics[group] != name:
                raise ValueError(f"Topic leakage: {group}")
            topics[group] = name
    return parts
