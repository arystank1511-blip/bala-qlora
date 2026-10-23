"""Export an annotated review copy; never auto-approve or alter source answers."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bala.metrics import interest_used, max_words_for_age, sentences, word_count
from bala.topics import topic_group
from bala.dataset import sha256
from prepare_data import problems
from review import COLUMNS, read_jsonl

EXTRA = ["priority", "auto_flags", "reviewer_note", "suggested_question", "suggested_answer", "source_url", "review_kind", "reviewer"]


def audit_records(records, notes):
    rows, seen = [], set()
    summary = {"total": len(records), "languages": dict(Counter(r["lang"] for r in records)),
               "pending": 0, "human_approved": 0, "ai_reviewed": 0, "review_kind_unknown": 0, "interest_not_detected": 0,
               "validation_errors": 0, "editorial_flags": 0,
               "notice": "AI annotations and heuristic checks are not human review or proof of correctness."}
    for record in records:
        issues = problems(record)
        if record["id"] in seen:
            issues.append("duplicate id")
        seen.add(record["id"])
        summary["validation_errors"] += bool(issues)
        try:
            topic_group(record)
        except ValueError:
            issues.append("unknown bilingual topic")
        if not interest_used(record["answer"], record["interest"]):
            issues.append("interest_not_detected (heuristic; inspect meaning)")
            summary["interest_not_detected"] += 1
        if not 2 <= len(sentences(record["answer"])) <= 4:
            issues.append("sentence_count (quoted punctuation can cause false alarms)")
        if word_count(record["answer"]) > max_words_for_age(int(record["age"])):
            issues.append("age_length_limit")
        decision = (record.get("decision") or "").strip().lower()
        summary["pending"] += decision not in {"keep", "fix", "drop"}
        review_kind = record.get("review_kind", "")
        if decision in {"keep", "fix"}:
            summary[{"human": "human_approved", "ai": "ai_reviewed"}.get(review_kind, "review_kind_unknown")] += 1
        note = notes.get(record["id"], {})
        summary["editorial_flags"] += bool(note)
        rows.append({**{c: record.get(c, "") for c in COLUMNS},
                     "priority": "1-fix" if note.get("suggested_answer") else "2-check" if note or issues else "3-review",
                     "auto_flags": "; ".join(issues), "reviewer_note": note.get("note", ""),
                     "suggested_question": note.get("suggested_question", ""),
                     "suggested_answer": note.get("suggested_answer", ""),
                     "source_url": note.get("source_url", ""), "review_kind": review_kind,
                     "reviewer": record.get("reviewer", "")})
    rows.sort(key=lambda row: (row["priority"], row["lang"], row["id"]))
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/review")
    args = parser.parse_args()
    output = args.output_dir
    if (output / "review_with_notes.csv").exists():
        print("Review copy already exists; choose a new output directory to preserve your edits.", file=sys.stderr)
        return 1
    records = read_jsonl(ROOT / "data/drafts.jsonl")
    sheet = ROOT / "data/review.csv"
    if sheet.exists():
        with sheet.open(encoding="utf-8-sig", newline="") as f:
            dialect = csv.Sniffer().sniff(f.readline(), delimiters=",;")
            f.seek(0)
            decisions = {r["id"]: r for r in csv.DictReader(f, dialect=dialect)}
        records = [{**r, **decisions.get(r["id"], {})} for r in records]
    annotations = json.loads((ROOT / "data/review_notes.json").read_text(encoding="utf-8"))
    rows, summary = audit_records(records, annotations["notes"])
    summary["editorial_scope"] = annotations["scope"]
    summary["seed_notes"] = annotations["seed_notes"]
    summary["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p)
                                for p in (ROOT / "data/drafts.jsonl", sheet) if p.exists()}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "review_with_notes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS + EXTRA)
        writer.writeheader()
        writer.writerows(rows)
    (output / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
