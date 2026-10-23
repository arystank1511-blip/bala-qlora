"""Automatic checks for one answer, plus averaging.

Each check matches a rule from the system prompt, so a fine-tuned model that
follows the brief scores higher than the base model:
- lang_ok        answered in the requested language (kk / ru)
- sentences_ok   2-5 sentences
- length_ok      short enough for the child's age
- interest_used  mentions the child's interest
- chrf           character n-gram overlap with the reviewed reference answer
"""

import re

SENTENCE = re.compile(r"[^.!?…]+[.!?…]+|[^.!?…]+$")
WORD = re.compile(r"\w+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.findall(text.strip()) if s.strip()]


def word_count(text: str) -> int:
    return len(WORD.findall(text))


def max_words_for_age(age: int) -> int:
    return 60 if age <= 8 else 90


# Word beginnings that count as using the interest. Matching by prefix copes with
# case endings in both languages ("мысығын", "жүзгенде", "лошадей"); "=" marks a
# whole word only, for short words that would otherwise match too much.
INTEREST_PREFIXES = {
    "футбол": ["футбол", "доп", "доб", "мяч", "бутс", "матч", "газон", "алаң", "команд"],
    "рисование": ["рисов", "рисун", "рису", "нарис", "художн", "краск", "кист", "палитр"],
    "сурет салу": ["сурет", "бояу", "қылқалам", "палитр", "салып", "салғанда", "=сал"],
    "динозавры": ["динозавр", "тираннозавр", "палеонтолог"],
    "динозаврлар": ["динозавр", "тираннозавр"],
    "космос": ["космо", "косми", "планет", "звёзд", "звезд", "лун", "ракет", "орбит"],
    "ғарыш": ["ғарыш", "ғаламшар", "жұлдыз", "орбита", "ракета"],
    "музыка": ["музык", "песн", "мелод", "ритм", "нот", "концерт", "барабан", "ән", "әуен", "ырғақ", "дабыл"],
    "Minecraft": ["minecraft", "майнкрафт"],
    "кошки": ["кош", "котён", "котят", "=кот", "кота"],
    "мысықтар": ["мысы"],
    "танцы": ["танц", "танец"],
    "би": ["=би", "биде", "биле", "биші"],
    "домбра": ["домбр"],
    "домбыра": ["домбыр"],
    "конструктор": ["конструктор"],
    "плавание": ["плав", "бассейн", "ныр"],
    "жүзу": ["жүзу", "жүзе", "жүзг", "жүзс", "жүзіп", "бассейн", "сүңгі", "шомыл"],
    "мультфильмы": ["мульт", "герой", "героев"],
    "мультфильмдер": ["мульт", "кейіпкер"],
    "лошади": ["лошад", "жереб", "конь", "коня", "табун", "всадник", "кобыл", "кумыс"],
    "аттар": ["=ат", "атпен", "атқа", "аттар", "аттың", "жылқы", "құлын", "бие", "салт"],
}


def interest_used(answer: str, interest: str) -> bool:
    """True if the answer refers to the child's interest."""
    words = [w.lower() for w in WORD.findall(answer)]
    prefixes = INTEREST_PREFIXES.get(interest)
    if prefixes is None:
        # Unknown interest: fall back to a stem of each of its words.
        prefixes = [w[: max(4, len(w) - 2)] for w in WORD.findall(interest.lower())]
    for prefix in prefixes:
        if prefix.startswith("="):
            if prefix[1:] in words:
                return True
        elif any(w.startswith(prefix.lower()) for w in words):
            return True
    return False


def chrf(hypothesis: str, reference: str) -> float | None:
    try:
        import sacrebleu
    except ImportError:
        return None
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score


def score(record: dict, answer: str) -> dict:
    from .lang import detect_language

    n_sentences = len(sentences(answer))
    return {
        "lang_ok": detect_language(answer) == record["lang"],
        "sentences_ok": 2 <= n_sentences <= 5,
        "length_ok": 0 < word_count(answer) <= max_words_for_age(int(record["age"])),
        "interest_used": interest_used(answer, record["interest"]),
        "chrf": chrf(answer, record["answer"]),
    }


def aggregate(scores: list[dict]) -> dict:
    """Mean of each metric; booleans become rates between 0 and 1."""
    if not scores:
        return {}
    out = {}
    for key in scores[0]:
        values = [s[key] for s in scores if s[key] is not None]
        out[key] = round(sum(float(v) for v in values) / len(values), 4) if values else None
    out["n"] = len(scores)
    return out
