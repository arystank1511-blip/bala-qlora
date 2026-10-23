"""Validate the completed dataset and every tokenized example before GPU work."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bala.dataset import is_reviewed, read_jsonl, sha256, validate_splits
from bala.features import IGNORE, tokenize_example
from bala.topics import topic_group
from prepare_data import problems


def main():
    from transformers import AutoTokenizer

    parts = validate_splits(ROOT)
    rows = [row for records in parts.values() for row in records]
    for row in rows:
        if problems(row) or not is_reviewed(row):
            raise ValueError(f"{row['id']}: invalid or unreviewed row: {problems(row)}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    lengths = {"kk": [], "ru": []}
    answer_lengths = {"kk": [], "ru": []}
    for row in rows:
        features = tokenize_example(tokenizer, row, max_len=512)
        lengths[row["lang"]].append(len(features["input_ids"]))
        answer_lengths[row["lang"]].append(sum(label != IGNORE for label in features["labels"]))
    report = {
        "status": "ready_for_gpu_training", "review_kind": "ai", "reviewer": "Codex",
        "model_training_completed": False, "human_review_completed": False,
        "examples": len(rows), "languages": dict(Counter(r["lang"] for r in rows)),
        "draft_review": json.loads((ROOT / "data/review_stats.json").read_text()),
        "seed_review": dict(Counter(r["review"]["decision"] for r in rows if r["id"].startswith("seed-"))),
        "splits": {name: {"count": len(records), "languages": dict(Counter(r["lang"] for r in records)),
                          "topics": sorted({topic_group(r) for r in records})}
                   for name, records in parts.items()},
        "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct", "max_length": 512,
        "max_tokens_by_language": {lang: max(values) for lang, values in lengths.items()},
        "max_answer_tokens_by_language": {lang: max(values) for lang, values in answer_lengths.items()},
        "validation_errors": 0, "truncated_examples": 0, "topic_overlap_between_splits": 0,
        "manifest_sha256": sha256(ROOT / "data/splits/manifest.json"),
    }
    (ROOT / "data/readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
