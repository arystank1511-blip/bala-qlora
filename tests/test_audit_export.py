import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_audit_keeps_human_decisions_separate(tmp_path):
    import audit_data
    record = {"id": "sample", "lang": "ru", "age": 7, "interest": "футбол",
              "topic": "Трение", "question": "Почему мяч останавливается?",
              "answer": "Мяч катится по траве и постепенно теряет скорость. Трава мешает движению мяча, поэтому он останавливается.",
              "decision": "", "notes": ""}
    notes = {"sample": {"note": "Check the explanation", "suggested_answer": "A proposed edit"}}
    rows, report = audit_data.audit_records([record], notes)
    assert rows[0]["decision"] == ""
    assert rows[0]["answer"] == record["answer"]
    assert rows[0]["suggested_answer"] == "A proposed edit"
    assert report["pending"] == 1
    assert report["human_approved"] == 0


def test_package_excludes_secrets_and_includes_pending_review(tmp_path):
    import package_colab
    import zipfile
    for name in ("bala", "scripts", "tests", "notebooks", "data"):
        (tmp_path / name).mkdir()
    for name in ("README.md", "requirements.txt", "requirements-dev.txt", "data/seed_examples.jsonl",
                 "data/drafts.jsonl", "data/review.csv", "data/review_notes.json",
                 "bala/__init__.py", "scripts/train_qlora.py", "notebooks/colab_train.ipynb"):
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    (tmp_path / ".env.local").write_text("SECRET=do-not-copy")
    (tmp_path / "bala/.env").write_text("SECRET=do-not-copy")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/adapter.bin").write_bytes(b"large model")
    (tmp_path / "data/reviewed.jsonl").write_text("reviewed fixture")
    (tmp_path / "data/splits").mkdir()
    (tmp_path / "data/splits/manifest.json").write_text("{}")
    (tmp_path / "results").mkdir()
    (tmp_path / "results/ai_review_summary.json").write_text("{\"past_run\": true}")
    target = tmp_path / "bundle.zip"
    package_colab.build_archive(tmp_path, target)
    with zipfile.ZipFile(target) as z:
        names = z.namelist()
        assert "bala-qlora/data/review.csv" in names
        assert "bala-qlora/scripts/train_qlora.py" in names
        assert "bala-qlora/data/reviewed.jsonl" in names
        assert "bala-qlora/data/splits/manifest.json" in names
        assert "bala-qlora/results/ai_review_summary.json" in names
        assert not any(".env" in name or "adapter.bin" in name for name in names)

    # These working files are ignored by Git. A fresh checkout with finalized
    # data must still be packageable without rebuilding the old review sheet.
    (tmp_path / "data/drafts.jsonl").unlink()
    (tmp_path / "data/review.csv").unlink()
    fresh_target = tmp_path / "fresh-checkout.zip"
    package_colab.build_archive(tmp_path, fresh_target)
    with zipfile.ZipFile(fresh_target) as z:
        assert "bala-qlora/data/reviewed.jsonl" in z.namelist()
    training_target = tmp_path / "fresh-training.zip"
    package_colab.build_archive(tmp_path, training_target, include_notebook=False)
    with zipfile.ZipFile(training_target) as z:
        assert not any(name.startswith("bala-qlora/results/") for name in z.namelist())


def test_audit_does_not_count_ai_review_as_human():
    import audit_data
    from test_review_pipeline import SEEDS
    row = dict(SEEDS[0], decision="keep", review_kind="ai", reviewer="Codex")
    _, report = audit_data.audit_records([row], {})
    assert report["human_approved"] == 0
    assert report["ai_reviewed"] == 1
    assert report["pending"] == 0
