#!/usr/bin/env python
"""TRL DPOTrainer + LoRA persona tuning (Idiot Sandwich track)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.data import load_persona_preferences
from alignment.train_utils import ensure_dir, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument(
        "--data_path",
        default="data/persona_lora/ramsay_style_preferences.json",
    )
    parser.add_argument("--output_dir", default="outputs/lora_persona_dpo")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    save_json(output_dir / "config.json", vars(args))

    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    rows = load_persona_preferences(args.data_path)
    # Conversational structure expected by recent TRL releases.
    formatted = []
    for row in rows:
        formatted.append(
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
            }
        )
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    # Reference is created by TRL from the base model when ref_model=None with PEFT.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    training_args = DPOConfig(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
        report_to=[],
        bf16=False,
        fp16=True,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    save_json(output_dir / "num_examples.json", {"n": len(rows), "train": len(train_ds), "eval": len(eval_ds)})
    print(f"LoRA-DPO complete. Saved to {output_dir / 'best'}")


if __name__ == "__main__":
    main()
