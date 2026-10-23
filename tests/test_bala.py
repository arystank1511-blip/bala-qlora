import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bala.features import IGNORE, tokenize_example  # noqa: E402
from bala.lang import detect_language  # noqa: E402
from bala.metrics import aggregate, interest_used, score, sentences  # noqa: E402
from bala.prompts import build_messages  # noqa: E402

import prepare_data  # noqa: E402
import score_manual  # noqa: E402

SEED = [json.loads(line) for line in (ROOT / "data" / "seed_examples.jsonl").read_text(encoding="utf-8").splitlines()]


class CharTokenizer:
    """Stand-in for a HF tokenizer: one token per character, Qwen-style chat markup."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_detects_kazakh_and_russian():
    assert detect_language("Өсімдіктер қалай тамақтанады?") == "kk"
    assert detect_language("Почему мяч останавливается?") == "ru"
    assert detect_language("Why is the sky blue?") == "other"
    assert detect_language("123 !!") == "unknown"


@pytest.mark.parametrize("record", SEED, ids=[r["id"] for r in SEED])
def test_seed_examples_are_valid(record):
    assert prepare_data.problems(record) == []
    assert detect_language(record["answer"]) == record["lang"]


def test_messages_use_the_requested_language():
    kk = next(r for r in SEED if r["lang"] == "kk")
    messages = build_messages(kk, with_answer=True)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert detect_language(messages[0]["content"]) == "kk"
    assert kk["interest"] in messages[1]["content"]


def test_loss_is_only_on_the_answer():
    tokenizer = CharTokenizer()
    record = SEED[0]
    features = tokenize_example(tokenizer, record, max_len=4096)
    labels, ids = features["labels"], features["input_ids"]
    assert len(labels) == len(ids) == len(features["attention_mask"])
    trained = "".join(chr(t) for t, l in zip(ids, labels) if l != IGNORE)
    assert trained == record["answer"] + "<|im_end|>\n"
    assert record["question"] not in trained


def test_long_answer_is_not_silently_cut_mid_sentence():
    with pytest.raises(ValueError, match="full example"):
        tokenize_example(CharTokenizer(), SEED[0], max_len=500)


def test_too_long_prompt_is_rejected():
    with pytest.raises(ValueError):
        tokenize_example(CharTokenizer(), SEED[0], max_len=50)


def test_sentence_split_and_interest_stems():
    assert len(sentences("Бір. Екі! Үш?")) == 3
    assert interest_used("Футболшыға жүгіру керек.", "футбол")
    assert interest_used("Как у динозавров в древности.", "динозавры")
    assert not interest_used("Про космос.", "футбол")


def test_score_rewards_following_the_brief():
    record = SEED[0]
    good = score(record, record["answer"])
    bad = score(record, "Plants make sugar from light. " * 40)
    assert good["lang_ok"] and good["sentences_ok"] and good["length_ok"] and good["interest_used"]
    assert not bad["lang_ok"] and not bad["length_ok"]


def test_aggregate_turns_booleans_into_rates():
    result = aggregate([{"lang_ok": True, "chrf": None}, {"lang_ok": False, "chrf": 40.0}])
    assert result == {"lang_ok": 0.5, "chrf": 40.0, "n": 2}


def test_rejects_answer_in_the_wrong_language():
    record = dict(SEED[0], answer=SEED[5]["answer"])
    assert any("expected kk" in p for p in prepare_data.problems(record))


def test_split_is_stratified_and_disjoint():
    topics = ["Сон", "Трение", "Радуга", "Магнит", "Фотосинтез"]
    records = [dict(SEED[0 if i % 2 == 0 else 5], id=f"x{i}",
                    topic=topics[(i // 2) % 5], question=f"question {i}") for i in range(100)]
    parts = prepare_data.split(records, seed=1, val_share=0.1, test_share=0.1)
    ids = [r["id"] for rows in parts.values() for r in rows]
    assert len(ids) == len(set(ids)) == 100
    assert {r["lang"] for r in parts["test"]} == {"kk", "ru"}
    assert prepare_data.split(records, 1, 0.1, 0.1) == parts


def test_manual_review_unblinding():
    rows = [
        {"id": "a", "lang": "kk", "better (A/B/=)": "A"},
        {"id": "b", "lang": "ru", "better (A/B/=)": "A"},
        {"id": "c", "lang": "ru", "better (A/B/=)": "="},
        {"id": "d", "lang": "ru", "better (A/B/=)": ""},
    ]
    key = {"a": "A", "b": "B", "c": "A", "d": "B"}
    assert score_manual.tally(rows, key) == {
        "finetuned": 1, "base": 1, "tie": 1, "unrated": 1, "finetuned_win_rate": 0.333,
    }
