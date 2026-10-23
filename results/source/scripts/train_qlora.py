"""QLoRA fine-tuning on data/splits/train.jsonl (run on a GPU, e.g. Colab T4).

    python scripts/train_qlora.py
    python scripts/train_qlora.py --model Qwen/Qwen2.5-3B-Instruct --epochs 2

The base model is loaded in 4-bit (NF4) and frozen; only small LoRA adapters on
the attention and MLP projections are trained. The adapter is saved to
outputs/adapter, the loss curve to results/train_log.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bala.features import tokenize_example  # noqa: E402
from bala.dataset import validate_splits  # noqa: E402

SPLITS = ROOT / "data" / "splits"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "adapter"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true", help="Validate the dataset without loading GPU libraries or weights")
    args = parser.parse_args()

    try:
        parts = validate_splits(ROOT)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if args.check_only:
        print({name: len(rows) for name, rows in parts.items()})
        return 0

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        print("No CUDA GPU found. Run this on Colab (Runtime -> Change runtime type -> T4 GPU).",
              file=sys.stderr)
        return 1

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = [tokenize_example(tokenizer, r, args.max_len) for r in parts["train"]]
    val = [tokenize_example(tokenizer, r, args.max_len) for r in parts["val"]]
    print(f"train: {len(train)} examples, val: {len(val)} examples")

    # T4 has no bfloat16, so compute in float16.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(ROOT / "outputs" / "checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            fp16=compute_dtype == torch.float16,
            bf16=compute_dtype == torch.bfloat16,
            optim="paged_adamw_8bit",
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(val),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100,
                                             pad_to_multiple_of=8),
    )
    trainer.train()

    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    manifest_text = (SPLITS / "manifest.json").read_text(encoding="utf-8")
    (Path(args.out) / "dataset_manifest.json").write_text(manifest_text, encoding="utf-8")
    import importlib.metadata
    environment = {name: importlib.metadata.version(name)
                   for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets")}
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "train_log.json").write_text(
        json.dumps({"args": vars(args), "environment": environment,
                    "dataset": json.loads(manifest_text), "log": trainer.state.log_history}, indent=2), encoding="utf-8"
    )
    print(f"Adapter saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
