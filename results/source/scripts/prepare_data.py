"""Validate reviewed examples, reject duplicates and split train / val / test.

    python scripts/prepare_data.py

Reads data/seed_examples.jsonl and data/reviewed.jsonl, writes data/splits/.
Entire bilingual topics stay together; seed topics remain in training. The
split is reproducible for the same dataset and seed, with approximate shares.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bala.lang import detect_language  # noqa: E402
from bala.metrics import max_words_for_age, sentences, word_count  # noqa: E402
from bala.topics import topic_group  # noqa: E402
from bala.dataset import is_reviewed, make_manifest  # noqa: E402

SOURCES = [ROOT / "data" / "seed_examples.jsonl", ROOT / "data" / "reviewed.jsonl"]
OUT = ROOT / "data" / "splits"
REQUIRED = ["id", "lang", "age", "interest", "topic", "question", "answer"]


def problems(record: dict) -> list[str]:
    issues = [f"missing {field}" for field in REQUIRED if record.get(field) is None or not str(record.get(field, "")).strip()]
    if issues:
        return issues
    for field in REQUIRED:
        if field != "age" and not isinstance(record[field], str):
            issues.append(f"{field} must be text")
    if issues:
        return issues
    if record["lang"] not in {"kk", "ru"}:
        issues.append(f"unknown lang {record['lang']!r}")
    if isinstance(record["age"], bool) or not re.fullmatch(r"\d+", str(record["age"])):
        issues.append("age must be an integer")
    elif not 5 <= int(record["age"]) <= 14:
        issues.append(f"age {record['age']} outside 5-14")
    for field in ("question", "answer"):
        found = detect_language(record[field])
        if found != record["lang"]:
            issues.append(f"{field} looks like {found}, expected {record['lang']}")
    words = word_count(record["answer"])
    if not 12 <= words <= 120:
        issues.append(f"answer has {words} words (want 12-120)")
    if re.fullmatch(r"\d+", str(record["age"])) and words > max_words_for_age(int(record["age"])):
        issues.append(f"answer exceeds age limit of {max_words_for_age(int(record['age']))} words")
    if not 2 <= len(sentences(record["answer"])) <= 5:
        issues.append("answer must have 2-5 sentences (heuristic; prompt target is 2-4)")
    return issues


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def load(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                records += [json.loads(line) for line in f if line.strip()]
    return records


def split(records: list[dict], seed: int, val_share: float, test_share: float) -> dict:
    if not (0 < val_share < 1 and 0 < test_share < 1 and val_share + test_share < 1):
        raise ValueError("val and test shares must be positive and sum to less than one")
    rng = random.Random(seed)
    parts = {"train": [], "val": [], "test": []}
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r["id"]):
        groups[topic_group(record)].append(record)
    pinned = {key for key, rows in groups.items() if any(r["id"].startswith("seed-") for r in rows)}
    # Holdouts must each contain both languages. Topics used as drafting style
    # examples remain in training, including their translations/paraphrases.
    eligible = sorted(key for key, rows in groups.items()
                      if key not in pinned and {r["lang"] for r in rows} == {"kk", "ru"})
    rng.shuffle(eligible)
    if len(eligible) < 3:
        raise ValueError("Need at least three reviewed bilingual topics outside the seed topics")
    assigned = set()
    for name, share, reserve in (("test", test_share, 2), ("val", val_share, 1)):
        target = max(1, round(len(records) * share))
        while eligible and len(parts[name]) < target and len(eligible) > reserve:
            key = eligible.pop()
            assigned.add(key)
            parts[name].extend(groups[key])
    for key, rows in groups.items():
        if key not in assigned:
            parts["train"].extend(rows)
    return parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--replace-splits", action="store_true", help="Explicitly replace a previous experiment's split")
    args = parser.parse_args()

    reviewed = load(SOURCES[1:])
    if not reviewed or any(not is_reviewed(r) for r in reviewed):
        print("No complete reviewed dataset. Finish review.csv and run scripts/review.py import first.", file=sys.stderr)
        return 1

    records, rejected, seen, seen_ids = [], [], set(), set()
    for record in load(SOURCES):
        issues = problems(record)
        key = (record.get("lang"), normalize(str(record.get("question", ""))))
        if not issues and key in seen:
            issues = ["duplicate question"]
        if not issues and record["id"] in seen_ids:
            issues = ["duplicate id"]
        if issues:
            rejected.append({"id": record.get("id"), "issues": issues})
            continue
        seen.add(key)
        seen_ids.add(record["id"])
        records.append(record)

    if rejected:
        for item in rejected:
            print(f"rejected {item['id']}: {'; '.join(item['issues'])}", file=sys.stderr)
        print("Fix reviewed data before generating splits; previous files preserved.", file=sys.stderr)
        return 1
    try:
        parts = split(records, args.seed, args.val, args.test)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and not args.replace_splits:
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            current = make_manifest(ROOT, args.seed, args.val, args.test)
        except (OSError, ValueError) as error:
            print(f"Existing split is incomplete: {error}. Preserve it or use --replace-splits.", file=sys.stderr)
            return 1
        if previous != current:
            print("Existing experiment differs. Preserve its results; use --replace-splits for a new experiment.", file=sys.stderr)
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        name: {"total": len(rows), "kk": sum(r["lang"] == "kk" for r in rows),
               "ru": sum(r["lang"] == "ru" for r in rows)}
        for name, rows in parts.items()
    }
    stats["rejected"] = rejected
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(make_manifest(ROOT, args.seed, args.val, args.test), indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "rejected"}, indent=2))
    for item in rejected:
        print(f"rejected {item['id']}: {'; '.join(item['issues'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
