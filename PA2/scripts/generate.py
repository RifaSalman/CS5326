#!/usr/bin/env python
"""Generate responses for a fixed evaluation prompt set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.formatting import format_instruction_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator_name", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=-1, help="Optional debug limit")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompts = json.loads(Path(args.prompts_path).read_text(encoding="utf-8"))
    if args.limit > 0:
        prompts = prompts[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.to(device)
    model.eval()

    outputs = []
    for i, row in enumerate(prompts):
        instruction = row.get("instruction") or row.get("prompt") or ""
        prompt = format_instruction_prompt(instruction)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(
            {
                "id": row.get("id", f"ex-{i:04d}"),
                "instruction": instruction,
                "output": text,
                "generator": args.generator_name,
                "dataset": row.get("dataset", Path(args.prompts_path).stem),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{len(prompts)}")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(outputs)} generations to {out}")


if __name__ == "__main__":
    main()
