"""Turn one example into token ids for supervised fine-tuning.

The loss is computed only on the answer: every prompt token (system message,
profile, question, chat markup) gets label -100, so the model learns how to
answer, not how to repeat the question.
"""

from .prompts import build_messages

IGNORE = -100


def tokenize_example(tokenizer, record: dict, max_len: int = 512) -> dict:
    prompt = tokenizer.apply_chat_template(
        build_messages(record), tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        build_messages(record, with_answer=True), tokenize=False
    )
    if not full.startswith(prompt):
        raise ValueError(f"{record['id']}: chat template prompt is not a prefix of the full text")

    # Tokenize prompt and answer separately so the boundary is exact.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= max_len:
        raise ValueError(f"{record['id']}: prompt alone is {len(prompt_ids)} tokens (max {max_len})")
    if len(prompt_ids) + len(answer_ids) > max_len:
        raise ValueError(f"{record['id']}: full example is {len(prompt_ids) + len(answer_ids)} tokens (max {max_len}); increase --max-len")

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
