import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_data
import review

SEEDS = prepare_data.load([ROOT / "data/seed_examples.jsonl"])


def setup_review(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    drafts = [dict(r, id=f"draft-{i}", source="draft:test") for i, r in enumerate(SEEDS[:2])]
    for name, file in [("DRAFTS", "drafts.jsonl"), ("REVIEW", "review.csv"),
                       ("REVIEWED", "reviewed.jsonl"), ("STATS", "stats.json")]:
        monkeypatch.setattr(review, name, tmp_path / file)
    review.DRAFTS.write_text("\n".join(json.dumps(r) for r in drafts), encoding="utf-8")
    entries = rows(drafts)
    with review.REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review.COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    review.REVIEWED.write_text("previous validated dataset\n", encoding="utf-8")
    return drafts


def test_pending_review_preserves_previous_dataset(tmp_path, monkeypatch):
    setup_review(tmp_path, monkeypatch, lambda ds: [dict(d, decision="") for d in ds])
    assert review.import_review() == 1
    assert review.REVIEWED.read_text() == "previous validated dataset\n"


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "missing", "bad_decision"])
def test_invalid_review_is_rejected_without_writing(tmp_path, monkeypatch, mutation):
    def rows(ds):
        out = [dict(d, decision="keep") for d in ds]
        if mutation == "duplicate":
            out[1] = out[0].copy()
        elif mutation == "unknown":
            out[0]["id"] = "not-a-draft"
        elif mutation == "missing":
            out.pop()
        else:
            out[0]["decision"] = "kepe"
        return out
    setup_review(tmp_path, monkeypatch, rows)
    assert review.import_review() == 1
    assert review.REVIEWED.read_text() == "previous validated dataset\n"


def test_complete_review_preserves_edits_and_records_provenance(tmp_path, monkeypatch):
    def rows(ds):
        return [dict(ds[0], decision="fix", interest="спорт", notes="checked"),
                dict(ds[1], decision="drop")]
    setup_review(tmp_path, monkeypatch, rows)
    assert review.import_review() == 0
    result = review.read_jsonl(review.REVIEWED)
    assert len(result) == 1
    assert result[0]["interest"] == "спорт"
    assert result[0]["source"] == "draft:test+human-review+edited"
    assert json.loads(review.STATS.read_text())["edited"] == 1


def test_ai_review_is_recorded_without_claiming_human_review(tmp_path, monkeypatch):
    setup_review(tmp_path, monkeypatch, lambda ds: [dict(d, decision="keep") for d in ds])
    assert review.import_review(review_kind="ai", reviewer="Codex") == 0
    result = review.read_jsonl(review.REVIEWED)
    assert all(r["source"] == "draft:test+ai-review" for r in result)
    assert all(r["review"] == {"kind": "ai", "reviewer": "Codex", "decision": "keep"} for r in result)
    assert json.loads(review.STATS.read_text())["review_kind"] == "ai"


def test_unknown_reviewer_kind_preserves_previous_data(tmp_path, monkeypatch):
    setup_review(tmp_path, monkeypatch, lambda ds: [dict(d, decision="keep") for d in ds])
    assert review.import_review(review_kind="automatic-human") == 1
    assert review.REVIEWED.read_text() == "previous validated dataset\n"


def test_review_policy_accepts_ai_only_with_explicit_provenance():
    from bala.dataset import is_reviewed
    assert is_reviewed({"source": "draft:test+human-review"})
    assert is_reviewed({"source": "draft:test+ai-review", "review": {
        "kind": "ai", "reviewer": "Codex", "decision": "fix"}})
    assert not is_reviewed({"source": "draft:test"})
    assert not is_reviewed({"source": "draft:test+ai-review"})
    assert not is_reviewed({"source": "draft:test+ai-review", "review": {
        "kind": "human", "reviewer": "Codex", "decision": "keep"}})


def test_prepare_refuses_seed_only_and_preserves_existing_split(tmp_path, monkeypatch):
    seed = tmp_path / "seeds.jsonl"
    seed.write_text("\n".join(json.dumps(r) for r in SEEDS), encoding="utf-8")
    monkeypatch.setattr(prepare_data, "SOURCES", [seed, tmp_path / "reviewed.jsonl"])
    out = tmp_path / "splits"
    out.mkdir()
    (out / "train.jsonl").write_text("previous split\n")
    monkeypatch.setattr(prepare_data, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py"])
    assert prepare_data.main() == 1
    assert (out / "train.jsonl").read_text() == "previous split\n"


def test_split_keeps_bilingual_topics_together_and_seeds_in_training():
    topics = [("Судың айналымы", "Круговорот воды"), ("Жанартаулар", "Вулканы"),
              ("Жүрек", "Сердце"), ("Қаңқа", "Скелет"), ("Дала", "Степь"),
              ("Тістер", "Зубы"), ("Жыл мезгілдері", "Смена времён года"),
              ("Күн жүйесі", "Солнечная система"), ("Ұйқы", "Сон"),
              ("Кемпірқосақ", "Радуга")]
    rows = [dict(SEEDS[0 if lang == "kk" else 5], id=f"{i}-{lang}", topic=topic,
                 source="draft:test+human-review")
            for i, pair in enumerate(topics) for lang, topic in zip(("kk", "ru"), pair)]
    rows.append(SEEDS[0])
    parts = prepare_data.split(rows, 42, .1, .1)
    location = {r["id"]: part for part, rs in parts.items() for r in rs}
    for i in range(len(topics)):
        assert location[f"{i}-kk"] == location[f"{i}-ru"]
    assert location[SEEDS[0]["id"]] == "train"
    assert all({r["lang"] for r in parts[p]} == {"kk", "ru"} for p in ("val", "test"))


@pytest.mark.parametrize("age", [None, "seven", 6.5, True])
def test_bad_age_reports_issue_instead_of_crashing(age):
    assert prepare_data.problems(dict(SEEDS[0], age=age))


def test_validation_respects_age_word_budget():
    record = dict(SEEDS[5], age=7, answer=("Дети играют во дворе и узнают новое о мире вокруг себя. " * 6))
    assert any("age limit" in issue for issue in prepare_data.problems(record))


def test_validation_rejects_one_sentence_answer():
    record = dict(SEEDS[5], answer="Динозавры оставили следы и кости в земле, и учёные изучают их, чтобы узнать прошлое нашей планеты.")
    assert any("sentences" in issue for issue in prepare_data.problems(record))


def test_preflight_rejects_missing_manifest(tmp_path):
    from bala.dataset import validate_splits
    with pytest.raises(ValueError, match="prepare_data"):
        validate_splits(tmp_path)


def test_preflight_detects_modified_split(tmp_path, monkeypatch):
    from bala.dataset import validate_splits
    seed = tmp_path / "data/seed_examples.jsonl"
    seed.parent.mkdir()
    seed.write_text("\n".join(json.dumps(r) for r in SEEDS), encoding="utf-8")
    reviewed = tmp_path / "data/reviewed.jsonl"
    topics = [("Жанартаулар", "Вулканы"), ("Жүрек", "Сердце"), ("Дала", "Степь"),
              ("Тістер", "Зубы"), ("Күн жүйесі", "Солнечная система")]
    rows = [dict(SEEDS[0 if lang == "kk" else 5], id=f"d{i}-{lang}", topic=topic,
                 question=f"{SEEDS[0 if lang == 'kk' else 5]['question']} {i}",
                 source="draft:test+human-review")
            for i, pair in enumerate(topics) for lang, topic in zip(("kk", "ru"), pair)]
    reviewed.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(prepare_data, "SOURCES", [seed, reviewed])
    monkeypatch.setattr(prepare_data, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_data, "OUT", tmp_path / "data/splits")
    monkeypatch.setattr(sys, "argv", ["prepare_data.py"])
    assert prepare_data.main() == 0
    parts = validate_splits(tmp_path)
    assert sum(map(len, parts.values())) == 20
    target = tmp_path / "data/splits/test.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_splits(tmp_path)
