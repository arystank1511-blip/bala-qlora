"""Ask the saved adapter one question on a CUDA GPU (no retraining needed)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=ROOT / "outputs/adapter")
    parser.add_argument("--lang", choices=("kk", "ru"), required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--interest", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--output", type=Path, help="Optionally save the question and answer as JSON")
    args = parser.parse_args()
    if not 5 <= args.age <= 14:
        parser.error("--age must be in the training profile range 5–14")
    if not args.interest.strip() or not args.question.strip():
        parser.error("--interest and --question cannot be blank")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error("Adapter not found. Extract outputs/adapter from the results archive first.")

    manifest = json.loads((args.adapter / "dataset_manifest.json").read_text(encoding="utf-8"))
    prompt_hash = hashlib.sha256((ROOT / "bala/prompts.py").read_bytes()).hexdigest()
    if manifest["prompt_sha256"] != prompt_hash:
        parser.error("Prompt code differs from the version used for this adapter")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from evaluate import answer

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required; use the T4 Colab runtime")
    base_name = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        ),
        device_map={"": torch.cuda.current_device()},
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    text, hit_limit = answer(model, tokenizer, vars(args), args.max_new_tokens)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "base_model": base_name,
            "lang": args.lang, "age": args.age, "interest": args.interest,
            "question": args.question, "answer": text, "hit_token_limit": hit_limit,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if hit_limit:
        print("Warning: the answer reached the token limit and may be incomplete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
