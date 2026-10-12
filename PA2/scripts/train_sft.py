#!/usr/bin/env python
"""From-scratch supervised fine-tuning for Qwen2.5-0.5B."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.batching import build_sft_batch
from alignment.data import load_ultrachat_sft
from alignment.train_utils import append_metrics, ensure_dir, get_autocast, save_checkpoint, save_json, seed_everything


class SFTDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return row["prompt"], row["response"]


def collate(batch, tokenizer, max_length):
    prompts, responses = zip(*batch)
    return build_sft_batch(tokenizer, list(prompts), list(responses), max_length)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # HF mean is over unmasked tokens; reweight by token count for reporting.
        n_tokens = batch["labels"][:, 1:].ne(-100).sum().item()
        total_loss += outputs.loss.item() * max(n_tokens, 1)
        total_tokens += max(n_tokens, 1)
    model.train()
    return total_loss / max(total_tokens, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", default="outputs/sft")
    parser.add_argument("--train_size", type=int, default=30_000)
    parser.add_argument("--val_size", type=int, default=1_000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=-1, help="Optional early stop for debugging")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    save_json(output_dir / "config.json", vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device)
    model.train()

    print("Loading UltraChat-style SFT data...")
    train_rows, val_rows = load_ultrachat_sft(args.train_size, args.val_size, seed=args.seed)
    save_json(output_dir / "data_info.json", {"train": len(train_rows), "val": len(val_rows)})

    train_loader = DataLoader(
        SFTDataset(train_rows),
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        SFTDataset(val_rows),
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer, args.max_length),
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    metrics_path = output_dir / "metrics.jsonl"
    best_val = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with get_autocast(device):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()

            if micro_step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    append_metrics(
                        metrics_path,
                        {
                            "step": global_step,
                            "split": "train",
                            "loss": outputs.loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "elapsed_sec": time.time() - start,
                        },
                    )
                    print(f"step={global_step} train_loss={outputs.loss.item():.4f}")

                if global_step % args.eval_every == 0 or global_step == total_steps:
                    val_loss = evaluate(model, val_loader, device)
                    append_metrics(
                        metrics_path,
                        {"step": global_step, "split": "val", "loss": val_loss, "elapsed_sec": time.time() - start},
                    )
                    print(f"step={global_step} val_loss={val_loss:.4f}", flush=True)
                    if math.isfinite(val_loss) and val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(
                            model,
                            tokenizer,
                            output_dir / "best",
                            {"step": global_step, "val_loss": val_loss},
                        )
                        print(f"Saved best checkpoint (val_loss={val_loss:.4f})", flush=True)

                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    break
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    # Always persist a final checkpoint, and ensure `best` exists for DPO.
    save_checkpoint(model, tokenizer, output_dir / "final", {"step": global_step, "best_val": best_val})
    best_dir = output_dir / "best"
    if not (best_dir / "config.json").exists():
        print("No finite-val best checkpoint found; copying final -> best", flush=True)
        save_checkpoint(model, tokenizer, best_dir, {"step": global_step, "val_loss": best_val, "fallback": True})
    print(f"Done. Best val loss={best_val:.4f}. Outputs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
