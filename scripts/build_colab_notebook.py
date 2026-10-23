"""Build one portable Colab notebook containing the reviewed project archive."""

import argparse
import base64
import hashlib
import json
import tempfile
from pathlib import Path

from package_colab import ROOT, build_archive


def build_notebook(root: Path, target: Path):
    # Exclude notebooks from the embedded archive to avoid recursive packaging.
    with tempfile.TemporaryDirectory() as directory:
        bundle = build_archive(root, Path(directory) / "project.zip", include_notebook=False)
        payload = bundle.read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    cells = []

    def cell(kind, source):
        entry = {"cell_type": kind, "id": f"bala-{len(cells):02}", "metadata": {},
                 "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    cell("markdown", """
# bala-qlora — Қабдеш Арыстан

Обучение Qwen2.5-1.5B-Instruct отвечать детям на казахском и русском с учётом возраста и интереса.
Все данные и код уже вложены в этот ноутбук: 300 проверенных примеров и 10 исходных примеров.
Редакторская проверка выполнена с помощью Codex; происхождение и исправления записаны в данных.

Выберите **Runtime → Change runtime type → T4 GPU**, затем **Runtime → Run all**.
Повторно загружать CSV или архив не нужно. Платный API и публикация на GitHub не нужны.
Ноутбук установит библиотеки, проверит данные, обучит адаптер, сравнит его с базовой моделью
и предложит скачать результаты. Доступность бесплатного GPU определяется Colab.

Темы разделены между train/val/test целиком, вместе с переводами. Проверка ИИ не заменяет
независимую оценку педагогом или носителем языка. Это ноутбук для нового запуска.
Первый эксперимент завершён: 3 эпохи на T4, chrF 18,88 → 25,09 на 32 примерах.
При этом обнаружены фактические ошибки и бессвязные казахские ответы: модель остаётся прототипом.
Итоги первого запуска описаны в docs/quality_review.md; его веса поставляются отдельным архивом.
""")
    cell("code", f'''
# @title Подготовка проекта: код и данные встроены
import base64, hashlib, io, os, subprocess, sys, zipfile
from pathlib import Path

PAYLOAD = "{encoded}"
payload = base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest() == "{digest}", "Повреждён архив проекта"
project = Path("/content/bala-qlora")
if project.exists():
    raise RuntimeError("Проект уже существует. Чтобы не затереть результаты, продолжите со следующей ячейки или используйте новую среду выполнения.")
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall("/content")
os.chdir(project)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run(*args):
    command = [sys.executable, *map(str, args)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)

print("Проект восстановлен:", project)
print("SHA-256 архива:", hashlib.sha256(payload).hexdigest())
''')
    cells[-1]["metadata"]["cellView"] = "form"
    cell("markdown", "## Установка и проверка\nGPU-библиотека torch используется из среды Colab; остальные версии закреплены в requirements.txt.")
    cell("code", '''
run("-m", "pip", "install", "-q", "-r", "requirements.txt", "-r", "requirements-dev.txt")
run("-m", "pytest", "-q")
run("scripts/train_qlora.py", "--check-only")
run("-c", "import torch; print('torch:',torch.__version__); assert torch.cuda.is_available(), 'Выберите T4 GPU в Runtime'; print('GPU:',torch.cuda.get_device_name(0))")
''')
    cell("markdown", "## Обучение\n3 эпохи, NF4, LoRA r=16, эффективный batch=16. Сохраняется лучший адаптер по validation loss. Тест не участвует в выборе параметров.")
    cell("code", '''
run("scripts/train_qlora.py", "--epochs", "3", "--max-len", "512")
''')
    cell("code", '''
import json
import matplotlib.pyplot as plt
log = json.loads(Path("results/train_log.json").read_text())["log"]
train = [(r["step"], r["loss"]) for r in log if "loss" in r]
val = [(r["step"], r["eval_loss"]) for r in log if "eval_loss" in r]
fig, ax = plt.subplots()
if train:
    ax.plot(*zip(*train), label="train")
if val:
    ax.plot(*zip(*val), "o-", label="validation")
ax.set(xlabel="step", ylabel="loss", title="bala-qlora")
ax.legend()
fig.savefig("results/loss.png", dpi=150, bbox_inches="tight")
plt.show()
''')
    cell("markdown", "## Сравнение с базовой моделью\nОдинаковые промпты и greedy decoding, адаптер выключен/включён в одной модели. Автоматические метрики не определяют фактическую правильность ответа.")
    cell("code", '''
run("scripts/evaluate.py")
from IPython.display import Markdown, display
display(Markdown(Path("results/metrics.md").read_text()))
''')
    cell("markdown", "## Сохранение\nАрхив содержит адаптер, логи, метрики, ответы обеих моделей, график и манифест данных. Сохраните его до отключения среды. Лист manual_review.csv предназначен для отдельной слепой оценки; незаполненный лист не считается человеческой оценкой.")
    cell("code", '''
from google.colab import files
run("scripts/record_run.py")
output_zip = project / "bala-qlora-results.zip"
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for directory in (Path("results"), Path("outputs/adapter"), Path("data/splits")):
        for item in sorted(directory.rglob("*")):
            if item.is_file():
                archive.write(item, item.as_posix())
files.download(str(output_zip))
''')
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU",
                             "colab": {"provenance": [], "gpuType": "T4"}}, "cells": cells}
    for index, entry in enumerate(cells):
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), f"cell-{index}", "exec")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/colab_train.ipynb")
    print(build_notebook(ROOT, parser.parse_args().output))
