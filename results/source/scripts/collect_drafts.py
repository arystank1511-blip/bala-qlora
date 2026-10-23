"""Collect drafts written outside the API (data/raw_drafts/*.jsonl) into
data/drafts.jsonl with stable ids and a source label, ready for review.

    python scripts/collect_drafts.py

The raw files hold drafts written in a Claude Code session at no API cost.
They go through exactly the same human review as API drafts.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import problems  # noqa: E402

RAW = ROOT / "data" / "raw_drafts"
DRAFTS = ROOT / "data" / "drafts.jsonl"
SOURCE = "draft:claude-code-session"


def main() -> int:
    records, seen = [], set()
    for path in sorted(RAW.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                digest = hashlib.sha1(f"{record['lang']}|{record['question']}".encode("utf-8")).hexdigest()
                record["id"] = f"{record['lang']}-{digest[:8]}"
                if record["id"] in seen:
                    print(f"{path.name}:{line_no} duplicate question, skipped")
                    continue
                seen.add(record["id"])
                record["source"] = SOURCE
                records.append(record)

    with DRAFTS.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    flagged = [(r["id"], problems(r)) for r in records if problems(r)]
    by_lang = {lang: sum(r["lang"] == lang for r in records) for lang in ("kk", "ru")}
    print(f"{len(records)} drafts -> {DRAFTS.relative_to(ROOT)} {by_lang}")
    for draft_id, issues in flagged:
        print(f"check {draft_id}: {'; '.join(issues)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
