"""Tell Kazakh from Russian.

Both languages use Cyrillic, but Kazakh has nine letters Russian does not
(ә ғ қ ң ө ұ ү һ і). In ordinary Kazakh text they make up several percent of
all letters; in Russian text they never appear. Short questions such as
"Дала деген не?" can contain none of them, so for those we fall back to
common function words.
"""

import re

KK_ONLY = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

# Share of Kazakh-only letters above which a Cyrillic text counts as Kazakh.
KK_THRESHOLD = 0.015

KK_WORDS = {"неге", "деген", "қалай", "қандай", "қайда", "неше", "бар", "жоқ", "және",
            "үшін", "ма", "ме", "ба", "бе", "па", "пе", "бола", "кейде", "ала", "алмаймыз"}
RU_WORDS = {"почему", "как", "что", "это", "зачем", "где", "и", "в", "на", "для",
            "из", "с", "ли", "такое", "бывает", "можно", "нужно"}
WORD = re.compile(r"\w+")


def detect_language(text: str) -> str:
    """Return "kk", "ru", "other" (mostly non-Cyrillic) or "unknown" (no letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    cyrillic = [c for c in letters if "Ѐ" <= c <= "ӿ"]
    if len(cyrillic) / len(letters) < 0.6:
        return "other"
    kk_share = sum(c in KK_ONLY for c in cyrillic) / len(cyrillic)
    if kk_share >= KK_THRESHOLD:
        return "kk"
    words = WORD.findall(text.lower())
    kk_votes = sum(w in KK_WORDS for w in words)
    ru_votes = sum(w in RU_WORDS for w in words)
    return "kk" if kk_votes > ru_votes else "ru"
