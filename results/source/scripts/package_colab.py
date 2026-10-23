"""Package code and local drafts/review for upload to Colab without GitHub."""

import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def build_archive(root: Path, target: Path, include_notebook: bool = True):
    # Explicit allowlist: no environment files, Git metadata, models or venv.
    files = [root / p for p in ("README.md", "requirements.txt", "requirements-dev.txt",
                               "data/seed_examples.jsonl", "data/drafts.jsonl", "data/review.csv",
                               "data/review_notes.json")]
    if include_notebook:
        files.append(root / "notebooks/colab_train.ipynb")
    for optional in ("data/reviewed.jsonl", "data/review_stats.json", "data/ai_review.jsonl",
                     "data/seed_review.json", "data/readiness.json"):
        if (root / optional).is_file():
            files.append(root / optional)
    files.extend((root / "data/splits").glob("*.json*"))
    for folder in ("bala", "scripts", "tests"):
        files.extend((root / folder).glob("*.py"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if not path.is_symlink():
                archive.write(path, "bala-qlora/" + path.relative_to(root).as_posix())
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bala-qlora-colab.zip")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use another filename to preserve the old package")
    print(build_archive(ROOT, args.output))


if __name__ == "__main__":
    main()
