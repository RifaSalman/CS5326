#!/usr/bin/env python
"""TRL DPOTrainer + LoRA persona tuning (Idiot Sandwich track)."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

# TRL DPOTrainer rejects nn.DataParallel (Kaggle T4x2 pre-sets CUDA_VISIBLE_DEVICES=0,1).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only kwargs accepted by cls.__init__ (ignore **kwargs catch-alls)."""
    params = inspect.signature(cls.__init__).parameters
    named = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    # If signature is only **kwargs, fall through to runtime try/drop below.
    if not named or named <= {"self"}:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in named}


def _instantiate(cls, kwargs: dict):
    """Construct cls(**kwargs), dropping unexpected kwargs until it works."""
    import re

    kwargs = _filter_kwargs(cls, kwargs)
    # Avoid duplicate eval strategy keys across transformers versions.
    if "eval_strategy" in kwargs and "evaluation_strategy" in kwargs:
        kwargs.pop("evaluation_strategy", None)
    while True:
        try:
            return cls(**kwargs), kwargs
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument ['\"](\w+)['\"]", str(exc))
            if not match:
                raise
            key = match.group(1)
            if key not in kwargs:
                raise
            print(f"Dropping unsupported {cls.__name__} kwarg: {key}", flush=True)
            kwargs.pop(key)


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
    # Prefer plain strings for broad TRL compatibility; chat templates applied by trainer/model.
    formatted = []
    for row in rows:
        formatted.append(
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    cfg_kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": max(args.max_length - args.max_prompt_length, 64),
        "remove_unused_columns": False,
        "report_to": [],
        "bf16": False,
        "fp16": True,
        "seed": args.seed,
    }
    training_args, cfg_kwargs = _instantiate(DPOConfig, cfg_kwargs)
    print("DPOConfig kwargs:", sorted(cfg_kwargs.keys()), flush=True)

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
    }
    trainer, trainer_kwargs = _instantiate(DPOTrainer, trainer_kwargs)
    print("DPOTrainer kwargs:", sorted(k for k in trainer_kwargs if k != "model"), flush=True)
    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    save_json(output_dir / "num_examples.json", {"n": len(rows), "train": len(train_ds), "eval": len(eval_ds)})
    print(f"LoRA-DPO complete. Saved to {output_dir / 'best'}", flush=True)


if __name__ == "__main__":
    main()
