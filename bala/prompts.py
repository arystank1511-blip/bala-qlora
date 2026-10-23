"""Chat messages for one example. Training and evaluation use the same function,
so the model is always asked in exactly the format it was trained on."""

SYSTEM = {
    "ru": (
        "Ты — добрый помощник для детей. Отвечай на русском языке, просто и коротко: "
        "2–4 предложения, без сложных терминов. Если известен интерес ребёнка, "
        "приведи пример из него."
    ),
    "kk": (
        "Сен — балаларға арналған мейірімді көмекшісің. Қазақ тілінде қарапайым әрі "
        "қысқа жауап бер: 2–4 сөйлем, күрделі терминдерсіз. Баланың қызығушылығы "
        "белгілі болса, сол саладан мысал келтір."
    ),
}

PROFILE = {
    "ru": "Возраст ребёнка: {age}. Интерес: {interest}.",
    "kk": "Баланың жасы: {age}. Қызығушылығы: {interest}.",
}


def build_messages(record: dict, with_answer: bool = False) -> list[dict]:
    lang = record["lang"]
    profile = PROFILE[lang].format(age=record["age"], interest=record["interest"])
    messages = [
        {"role": "system", "content": SYSTEM[lang]},
        {"role": "user", "content": f"{profile}\n{record['question']}"},
    ]
    if with_answer:
        messages.append({"role": "assistant", "content": record["answer"]})
    return messages
