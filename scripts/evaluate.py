"""Compare the base model with the fine-tuned one on data/splits/test.jsonl.

    python scripts/evaluate.py

Both answers come from the same loaded model: the LoRA adapter is switched off
for the "base" answer and on for the "finetuned" one, with greedy decoding, so
the only difference is the adapter. Writes:
- results/predictions.jsonl   every prompt with both answers and their scores
- results/metrics.json / .md  averaged automatic metrics, per language too
- results/manual_review.csv   blind A/B sheet for a human rater
- results/manual_review_key.json  which side was the fine-tuned model
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bala.metrics import aggregate, score  # noqa: E402
from bala.prompts import build_messages  # noqa: E402
from bala.dataset import validate_splits  # noqa: E402

RESULTS = ROOT / "results"


@torch.no_grad()
def answer(model, tokenizer, record: dict, max_new_tokens: int) -> tuple[str, bool]:
    prompt = tokenizer.apply_chat_template(
        build_messages(record), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated = output[0, inputs["input_ids"].shape[1]:]
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    hit_limit = len(generated) >= max_new_tokens and int(generated[-1]) not in eos_ids
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), hit_limit


def table(metrics: dict) -> str:
    keys = ["lang_ok", "sentences_ok", "length_ok", "interest_used", "chrf"]
    lines = ["| subset | model | " + " | ".join(keys) + " | n |",
             "|---|---|" + "---|" * len(keys) + "---|"]
    for subset, by_model in metrics.items():
        for name, values in by_model.items():
            cells = [f"{values.get(k):.3f}" if isinstance(values.get(k), float) else str(values.get(k))
                     for k in keys]
            lines.append(f"| {subset} | {name} | " + " | ".join(cells) + f" | {values.get('n')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Optional assertion; defaults to the adapter's base model")
    parser.add_argument("--adapter", default=str(ROOT / "outputs" / "adapter"))
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        parts = validate_splits(ROOT)
        expected = json.loads((Path(args.adapter) / "dataset_manifest.json").read_text(encoding="utf-8"))
        current = json.loads((ROOT / "data/splits/manifest.json").read_text(encoding="utf-8"))
        if expected != current:
            raise ValueError("Adapter was trained on a different dataset/split")
        base_name = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
        if args.model and args.model != base_name:
            raise ValueError(f"Adapter base model is {base_name}, not {args.model}")
        args.model = base_name
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("No CUDA GPU found. Run this on Colab.", file=sys.stderr)
        return 1

    records = parts[args.split]
    if args.limit:
        records = records[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=compute_dtype),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = []
    for i, record in enumerate(records, 1):
        with model.disable_adapter():
            base_answer, base_hit_limit = answer(model, tokenizer, record, args.max_new_tokens)
        tuned_answer, tuned_hit_limit = answer(model, tokenizer, record, args.max_new_tokens)
        rows.append({
            **record,
            "base": base_answer,
            "finetuned": tuned_answer,
            "base_hit_token_limit": base_hit_limit,
            "finetuned_hit_token_limit": tuned_hit_limit,
            "base_scores": score(record, base_answer),
            "finetuned_scores": score(record, tuned_answer),
        })
        print(f"{i}/{len(records)} {record['id']}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "evaluation_config.json").write_text(json.dumps({
        "args": vars(args), "test_examples": len(records),
        "base_limit_hits": sum(r["base_hit_token_limit"] for r in rows),
        "finetuned_limit_hits": sum(r["finetuned_hit_token_limit"] for r in rows),
    }, indent=2), encoding="utf-8")
    with (RESULTS / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {}
    for subset in ("all", "kk", "ru"):
        chosen = [r for r in rows if subset == "all" or r["lang"] == subset]
        if chosen:
            metrics[subset] = {
                "base": aggregate([r["base_scores"] for r in chosen]),
                "finetuned": aggregate([r["finetuned_scores"] for r in chosen]),
            }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS / "metrics.md").write_text(table(metrics), encoding="utf-8")

    # Blind A/B sheet: the rater does not know which side is fine-tuned.
    rng = random.Random(args.seed)
    key = {}
    with (RESULTS / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "lang", "age", "interest", "question", "reference",
                         "answer_A", "answer_B", "better (A/B/=)", "comment"])
        for row in rows:
            tuned_is_a = rng.random() < 0.5
            key[row["id"]] = "A" if tuned_is_a else "B"
            a, b = (row["finetuned"], row["base"]) if tuned_is_a else (row["base"], row["finetuned"])
            writer.writerow([row["id"], row["lang"], row["age"], row["interest"], row["question"],
                             row["answer"], a, b, "", ""])
    (RESULTS / "manual_review_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    print(table(metrics))
    print("Rate results/manual_review.csv, then run scripts/score_manual.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
