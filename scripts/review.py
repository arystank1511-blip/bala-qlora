"""Import completed draft reviews with explicit human/AI provenance.

    python scripts/review.py export   # data/drafts.jsonl -> data/review.csv
    (open review.csv in Excel: fix answers in place, set decision to keep / fix / drop)
    python scripts/review.py import   # data/review.csv   -> data/reviewed.jsonl

decision: keep = correct as is, fix = I corrected the text, drop = remove.
Every draft needs a decision before import; incomplete/invalid sheets leave
the previous reviewed dataset untouched. --input accepts an annotated copy.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / "data" / "drafts.jsonl"
REVIEW = ROOT / "data" / "review.csv"
REVIEWED = ROOT / "data" / "reviewed.jsonl"
STATS = ROOT / "data" / "review_stats.json"

COLUMNS = ["id", "lang", "age", "interest", "topic", "question", "answer", "decision", "notes"]
KEEP = {"keep", "fix"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export() -> int:
    drafts = read_jsonl(DRAFTS)
    if not drafts:
        print("No drafts yet. Run scripts/generate_drafts.py first.", file=sys.stderr)
        return 1
    if REVIEW.exists():
        print(f"{REVIEW.name} already exists; not overwriting your review.", file=sys.stderr)
        return 1
    # utf-8-sig so Excel opens Cyrillic correctly.
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in drafts:
            writer.writerow({**record, "decision": "", "notes": ""})
    print(f"Wrote {len(drafts)} rows to {REVIEW.relative_to(ROOT)}")
    return 0


def import_review(input_path: Path | None = None, review_kind: str = "human", reviewer: str = "") -> int:
    sys.path.insert(0, str(ROOT))
    from prepare_data import problems

    if review_kind not in {"human", "ai"} or (review_kind == "ai" and not reviewer.strip()):
        print("Review not imported: choose human or ai; AI review requires a reviewer name.", file=sys.stderr)
        return 1

    drafts = {r["id"]: r for r in read_jsonl(DRAFTS)}
    try:
        with (input_path or REVIEW).open(encoding="utf-8-sig", newline="") as f:
            header = f.readline()
            f.seek(0)
            dialect = csv.Sniffer().sniff(header, delimiters=",;")
            reader = csv.DictReader(f, dialect=dialect)
            missing = set(COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
            rows = list(reader)
    except (OSError, csv.Error, ValueError) as error:
        print(f"Review not imported: {error}", file=sys.stderr)
        return 1

    counts = {"keep": 0, "fix": 0, "drop": 0, "pending": 0, "edited": 0}
    kept, errors, seen = [], [], set()
    for row in rows:
        row_id = (row.get("id") or "").strip()
        if row_id not in drafts or row_id in seen:
            errors.append(f"Unknown or duplicate id: {row_id!r}")
            continue
        seen.add(row_id)
        row_kind = (row.get("review_kind") or "").strip()
        if row_kind and row_kind != review_kind:
            errors.append(f"{row_id}: CSV review_kind={row_kind}; use --review-kind {row_kind}")
            continue
        decision = (row.get("decision") or "").strip().lower()
        if decision not in {"keep", "fix", "drop"}:
            counts["pending"] += 1
            if decision:
                errors.append(f"{row_id}: unknown decision {decision!r}")
            continue
        counts[decision] += 1
        if decision not in KEEP:
            continue
        draft = drafts[row_id]
        edited = any(
            (row[field] or "").strip() != str(draft.get(field, "")).strip()
            for field in ("lang", "age", "interest", "question", "answer", "topic")
        )
        counts["edited"] += edited
        record = {field: (row.get(field) or "").strip() for field in COLUMNS[:7]}
        issues = problems(record)
        if issues:
            errors.append(f"{row_id}: {'; '.join(issues)}")
            continue
        kept.append({**record,
            "age": int(record["age"]),
            "source": f"{draft.get('source', 'draft')}+{review_kind}-review" + ("+edited" if edited else ""),
            "review": {"kind": review_kind, "reviewer": reviewer.strip(), "decision": decision},
            "notes": (row.get("notes") or "").strip(),
        })

    missing_ids = set(drafts) - seen
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} draft rows; use drop instead of deleting rows")
    if counts["pending"]:
        errors.append(f"{counts['pending']} decisions pending; fill keep/fix/drop for every row")
    if not drafts:
        errors.append("No source drafts found")
    if errors:
        print("Review not imported; existing data preserved.", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    with REVIEWED.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts.update(review_kind=review_kind, reviewer=reviewer.strip())
    STATS.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"Kept {len(kept)} reviewed examples -> {REVIEWED.relative_to(ROOT)}")
    print(json.dumps(counts, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "import"])
    parser.add_argument("--input", type=Path, help="Reviewed CSV copy to import")
    parser.add_argument("--review-kind", choices=["human", "ai"], default="human")
    parser.add_argument("--reviewer", default="", help="Reviewer name; required for AI review")
    args = parser.parse_args()
    return export() if args.command == "export" else import_review(args.input, args.review_kind, args.reviewer)


if __name__ == "__main__":
    sys.exit(main())
