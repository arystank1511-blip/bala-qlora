"""Bilingual topic aliases for topic-disjoint evaluation of this dataset.

Grouping entire topics also keeps translated and paraphrased questions together.
Extend this map when adding new topics; this is not semantic language matching.
"""

import re

TOPICS = [
    ("water-cycle", "Судың айналымы", "Круговорот воды", "Жаңбыр"),
    ("volcanoes", "Жанартаулар", "Вулканы"),
    ("teeth", "Тістер", "Зубы"),
    ("solar-system", "Күн жүйесі", "Солнечная система"),
    ("sound", "Дыбыс және жаңғырық", "Звук и эхо"),
    ("seasons", "Жыл мезгілдері", "Смена времён года"),
    ("electricity", "Электр", "Электричество"),
    ("day-night", "Күн мен түн", "День и ночь"),
    ("buoyancy", "Заттардың жүзуі", "Плавание предметов"),
    ("heart", "Жүрек", "Сердце"),
    ("computers", "Компьютер және бағдарлама", "Компьютер и программа"),
    ("light", "Жарық және көлеңке", "Свет и тень"),
    ("breathing", "Өкпе және тыныс алу", "Лёгкие и дыхание"),
    ("sleep", "Ұйқы", "Сон"),
    ("food-chain", "Қоректік тізбек", "Пищевая цепочка"),
    ("stars", "Жұлдыздар", "Звёзды"),
    ("caspian", "Каспий теңізі", "Каспийское море"),
    ("earthquakes", "Жер сілкінісі", "Землетрясения"),
    ("geometry", "Геометриялық фигуралар", "Геометрические фигуры"),
    ("germs", "Микробтар және қол жуу", "Микробы и мытьё рук"),
    ("gravity", "Тартылыс күші", "Гравитация"),
    ("rainbow", "Кемпірқосақ", "Радуга"),
    ("steppe", "Дала", "Степь"),
    ("fractions", "Бөлшектер", "Дроби"),
    ("time", "Сағат және уақыт", "Часы и время"),
    ("vitamins", "Дәрумендер", "Витамины"),
    ("internet", "Интернет"),
    ("water-states", "Мұз, су және бу", "Лёд, вода и пар"),
    ("moon-phases", "Айдың фазалары", "Фазы Луны"),
    ("skeleton", "Қаңқа", "Скелет"),
    ("leaves", "Жапырақтар неге сарғаяды", "Почему листья желтеют"),
    ("robots", "Робот", "Роботы и машины"),
    ("addition", "Қосу және азайту", "Сложение и вычитание"),
    ("kazakhstan", "Қазақстан картада", "Казахстан на карте"),
    ("length", "Ұзындықты өлшеу", "Измерение длины"),
    ("dinosaurs", "Динозаврлар және тасқа айналған қалдықтар", "Динозавры и окаменелости", "Окаменелости"),
    ("pollination", "Аралар және тозаңдану", "Пчёлы и опыление"),
    ("recycling", "Қоқысты қайта өңдеу", "Переработка мусора"),
    ("migration", "Құстардың қоныс аударуы", "Миграция птиц"),
    ("division", "Бөлу", "Деление"),
    ("friction", "Үйкеліс", "Трение"),
    ("digestion", "Ас қорыту", "Пищеварение"),
    ("water-saving", "Суды үнемдеу", "Сохранение воды"),
    ("magnets", "Магнит"),
    ("money", "Ақша және қайтарым", "Деньги и сдача"),
    ("maps", "Карта және компас", "Карта и компас"),
    ("multiplication", "Көбейту", "Умножение"),
    ("photosynthesis", "Фотосинтез"),
]


def normalize(text):
    return re.sub(r"\W+", " ", text.lower().replace("ё", "е")).strip()


ALIASES = {normalize(alias): group for group, *aliases in TOPICS for alias in aliases}


def topic_group(record):
    topic = normalize(record.get("topic", ""))
    if topic not in ALIASES:
        raise ValueError(f"{record.get('id')}: unknown topic {record.get('topic')!r}; add its bilingual aliases to bala/topics.py")
    return ALIASES[topic]
