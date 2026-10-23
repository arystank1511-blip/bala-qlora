"""Unblind the manual A/B review and report the fine-tuned model's win rate.

    python scripts/score_manual.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COLUMN = "better (A/B/=)"


def tally(rows: list[dict], key: dict) -> dict:
    counts = {"finetuned": 0, "base": 0, "tie": 0, "unrated": 0}
    for row in rows:
        verdict = (row.get(COLUMN) or "").strip().upper()
        if verdict not in {"A", "B", "="}:
            counts["unrated"] += 1
        elif verdict == "=":
            counts["tie"] += 1
        elif verdict == key[row["id"]]:
            counts["finetuned"] += 1
        else:
            counts["base"] += 1
    rated = counts["finetuned"] + counts["base"] + counts["tie"]
    counts["finetuned_win_rate"] = round(counts["finetuned"] / rated, 3) if rated else None
    return counts


def main() -> int:
    key = json.loads((RESULTS / "manual_review_key.json").read_text(encoding="utf-8"))
    with (RESULTS / "manual_review.csv").open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        rows = list(csv.DictReader(f, dialect=csv.Sniffer().sniff(sample, delimiters=",;")))

    report = {"all": tally(rows, key)}
    for lang in ("kk", "ru"):
        report[lang] = tally([r for r in rows if r["lang"] == lang], key)
    (RESULTS / "manual_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
