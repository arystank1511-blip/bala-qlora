"""Draft new training examples with Claude. Every draft is reviewed by a person
before it can reach training (see scripts/review.py), so this is a writing aid,
not the source of truth.

Usage:
    python scripts/generate_drafts.py --lang kk --count 200
    python scripts/generate_drafts.py --lang ru --count 200

Credentials: the SDK reads ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

import anthropic
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed_examples.jsonl"
DRAFTS = ROOT / "data" / "drafts.jsonl"

# School topics for ages 6-12, written in Russian; the model writes the example
# itself in the requested language.
TOPICS = [
    "Фотосинтез", "Круговорот воды", "Радуга", "Гравитация", "Магнит", "Трение",
    "Смена времён года", "День и ночь", "Фазы Луны", "Солнечная система", "Звёзды",
    "Вулканы", "Землетрясения", "Динозавры и окаменелости", "Сердце", "Лёгкие и дыхание",
    "Зубы", "Сон", "Микробы и мытьё рук", "Витамины", "Пищеварение", "Скелет",
    "Пчёлы и опыление", "Миграция птиц", "Пищевая цепочка", "Почему листья желтеют",
    "Электричество", "Звук и эхо", "Свет и тень", "Лёд, вода и пар", "Плавание предметов",
    "Сложение и вычитание", "Умножение", "Деление", "Дроби", "Часы и время",
    "Деньги и сдача", "Геометрические фигуры", "Измерение длины", "Карта и компас",
    "Интернет", "Компьютер и программа", "Робот", "Переработка мусора",
    "Казахстан на карте", "Степь", "Каспийское море", "Сохранение воды",
]

INTERESTS = {
    "kk": ["футбол", "сурет салу", "динозаврлар", "ғарыш", "музыка", "Minecraft",
           "мысықтар", "би", "домбыра", "конструктор", "жүзу", "мультфильмдер", "аттар"],
    "ru": ["футбол", "рисование", "динозавры", "космос", "музыка", "Minecraft",
           "кошки", "танцы", "домбра", "конструктор", "плавание", "мультфильмы", "лошади"],
}

LANG_NAME = {"kk": "Kazakh", "ru": "Russian"}


class Draft(BaseModel):
    topic: str
    question: str
    answer: str


class DraftBatch(BaseModel):
    examples: list[Draft]


def load_seed() -> list[dict]:
    with SEED.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(lang: str, profiles: list[dict], seed: list[dict]) -> str:
    style = "\n\n".join(
        f"Profile: age {s['age']}, interest: {s['interest']}\nQ: {s['question']}\nA: {s['answer']}"
        for s in seed
        if s["lang"] == lang
    )
    wanted = "\n".join(
        f"{i + 1}. topic: {p['topic']}; child's age: {p['age']}; interest: {p['interest']}"
        for i, p in enumerate(profiles)
    )
    return f"""You write training examples for a tutor that explains school topics to children.

Write everything in {LANG_NAME[lang]} only: the topic name, the child's question and the answer.
{"Use natural, correct literary Kazakh, not a word-by-word translation from Russian." if lang == "kk" else ""}

Rules for every answer:
- 2-4 short sentences a child of that age understands; no jargon.
- Factually correct.
- Includes one concrete example or comparison from the child's interest.
- Warm and encouraging, no lists or headings.

Match the style of these reviewed examples:

{style}

Write exactly {len(profiles)} examples, one per line below, in the same order:

{wanted}"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["kk", "ru"], required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seed = load_seed()
    client = anthropic.Anthropic()
    written = 0

    with DRAFTS.open("a", encoding="utf-8") as out:
        while written < args.count:
            size = min(args.batch, args.count - written)
            profiles = [
                {"topic": rng.choice(TOPICS), "age": rng.randint(6, 12),
                 "interest": rng.choice(INTERESTS[args.lang])}
                for _ in range(size)
            ]
            try:
                response = client.messages.parse(
                    model=args.model,
                    max_tokens=16000,
                    messages=[{"role": "user", "content": build_prompt(args.lang, profiles, seed)}],
                    output_format=DraftBatch,
                )
            except anthropic.RateLimitError:
                print("Rate limited even after retries; stopping. Run again later.", file=sys.stderr)
                break
            except anthropic.APIStatusError as error:
                print(f"API error {error.status_code}: {error.message}; skipping batch", file=sys.stderr)
                continue
            except anthropic.APIConnectionError:
                print("Network error; stopping.", file=sys.stderr)
                break

            if response.stop_reason == "refusal" or response.parsed_output is None:
                print(f"No usable output (stop_reason={response.stop_reason}); skipping batch", file=sys.stderr)
                continue

            drafts = response.parsed_output.examples
            if len(drafts) != len(profiles):
                print(f"Expected {len(profiles)} examples, got {len(drafts)}; skipping batch", file=sys.stderr)
                continue

            for profile, draft in zip(profiles, drafts):
                record = {
                    "id": f"{args.lang}-{uuid.uuid4().hex[:8]}",
                    "lang": args.lang,
                    "age": profile["age"],
                    "interest": profile["interest"],
                    "topic": draft.topic,
                    "question": draft.question.strip(),
                    "answer": draft.answer.strip(),
                    "source": f"draft:{args.model}",
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += len(drafts)
            print(f"{written}/{args.count} drafts written")

    print(f"Done. Drafts are in {DRAFTS.relative_to(ROOT)}; review them with scripts/review.py export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
