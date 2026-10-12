#!/usr/bin/env python
"""From-scratch Direct Preference Optimization (two-track support)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW, RMSprop
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.batching import build_dpo_batch
from alignment.data import load_hh_rlhf_preferences, load_ultrafeedback_preferences
from alignment.losses import compute_dpo_loss_from_logps, dpo_reward_metrics
from alignment.train_utils import append_metrics, ensure_dir, get_autocast, save_checkpoint, save_json, seed_everything


class PreferenceDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return row["prompt"], row["chosen"], row["rejected"]


def collate(batch, tokenizer, max_length):
    prompts, chosen, rejected = zip(*batch)
    return build_dpo_batch(tokenizer, list(prompts), list(chosen), list(rejected), max_length)


def _logps(model, batch, device):
    from alignment.losses import _shift_logprobs_from_logits

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _shift_logprobs_from_logits(outputs.logits, labels)


@torch.no_grad()
def evaluate(policy, reference, loader, policy_device, ref_device, beta):
    policy.eval()
    totals = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0}
    n = 0
    for batch in loader:
        with get_autocast(policy_device):
            policy_chosen = _logps(policy, batch["chosen"], policy_device)
            policy_rejected = _logps(policy, batch["rejected"], policy_device)
        ref_chosen = _logps(reference, batch["chosen"], ref_device)
        ref_rejected = _logps(reference, batch["rejected"], ref_device)
        loss = compute_dpo_loss_from_logps(
            policy_chosen,
            policy_rejected,
            ref_chosen.to(policy_device),
            ref_rejected.to(policy_device),
            beta,
        )
        metrics = dpo_reward_metrics(
            policy_chosen,
            policy_rejected,
            ref_chosen.to(policy_device),
            ref_rejected.to(policy_device),
            beta,
        )
        totals["loss"] += loss.item()
        totals["reward_accuracy"] += metrics["reward_accuracy"]
        totals["reward_margin"] += metrics["reward_margin"]
        n += 1
    policy.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["hh", "ultrafeedback"], required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_size", type=int, default=15_000)
    parser.add_argument("--val_size", type=int, default=1_000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="rmsprop")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir or f"outputs/dpo_{args.track}")
    save_json(output_dir / "config.json", vars(args))

    policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else policy_device)
    print(f"Policy device={policy_device} reference device={ref_device}")

    sft_path = Path(args.sft_checkpoint)
    if not sft_path.exists():
        # Common fallback if only `final` was saved.
        alt = sft_path.parent / "final"
        if alt.exists():
            print(f"Missing {sft_path}; falling back to {alt}", flush=True)
            sft_path = alt
        else:
            raise FileNotFoundError(
                f"SFT checkpoint not found at {args.sft_checkpoint} (also tried {alt}). "
                "Re-run SFT or point --sft_checkpoint at an existing local folder."
            )
    sft_checkpoint = str(sft_path)

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True, local_files_only=sft_path.exists())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Trainable policy stays fp32 for GradScaler compatibility.
    policy = AutoModelForCausalLM.from_pretrained(sft_checkpoint, trust_remote_code=True, local_files_only=True)
    policy.to(policy_device)
    policy.train()

    reference = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.float16 if ref_device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    )
    reference.to(ref_device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    print(f"Loading preference data for track={args.track}...")
    if args.track == "hh":
        train_rows, val_rows = load_hh_rlhf_preferences(args.train_size, args.val_size, seed=args.seed)
    else:
        train_rows, val_rows = load_ultrafeedback_preferences(args.train_size, args.val_size, seed=args.seed)
    save_json(output_dir / "data_info.json", {"train": len(train_rows), "val": len(val_rows)})

    train_loader = DataLoader(
        PreferenceDataset(train_rows),
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        PreferenceDataset(val_rows),
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer, args.max_length),
    )

    if args.optimizer == "adamw":
        optimizer = AdamW(policy.parameters(), lr=args.learning_rate)
    else:
        optimizer = RMSprop(policy.parameters(), lr=args.learning_rate)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    metrics_path = output_dir / "metrics.jsonl"
    best_acc = -1.0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(train_loader, start=1):
            # Policy stays in fp32; autocast is used only for memory-friendly forwards.
            with get_autocast(policy_device):
                policy_chosen = _logps(policy, batch["chosen"], policy_device)
                policy_rejected = _logps(policy, batch["rejected"], policy_device)
            with torch.no_grad():
                with get_autocast(ref_device):
                    ref_chosen = _logps(reference, batch["chosen"], ref_device)
                    ref_rejected = _logps(reference, batch["rejected"], ref_device)

            loss = compute_dpo_loss_from_logps(
                policy_chosen.float(),
                policy_rejected.float(),
                ref_chosen.to(policy_device).float(),
                ref_rejected.to(policy_device).float(),
                args.beta,
            )
            (loss / args.grad_accum).backward()

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    metrics = dpo_reward_metrics(
                        policy_chosen.detach(),
                        policy_rejected.detach(),
                        ref_chosen.to(policy_device),
                        ref_rejected.to(policy_device),
                        args.beta,
                    )
                    append_metrics(
                        metrics_path,
                        {
                            "step": global_step,
                            "split": "train",
                            "loss": loss.item(),
                            **metrics,
                            "elapsed_sec": time.time() - start,
                        },
                    )
                    print(
                        f"step={global_step} loss={loss.item():.4f} "
                        f"acc={metrics['reward_accuracy']:.3f} margin={metrics['reward_margin']:.4f}"
                    )

                if global_step % args.eval_every == 0 or global_step == total_steps:
                    val_metrics = evaluate(policy, reference, val_loader, policy_device, ref_device, args.beta)
                    append_metrics(
                        metrics_path,
                        {"step": global_step, "split": "val", **val_metrics, "elapsed_sec": time.time() - start},
                    )
                    print(
                        f"step={global_step} val_loss={val_metrics['loss']:.4f} "
                        f"val_acc={val_metrics['reward_accuracy']:.3f}"
                    )
                    if val_metrics["reward_accuracy"] > best_acc:
                        best_acc = val_metrics["reward_accuracy"]
                        save_checkpoint(
                            policy,
                            tokenizer,
                            output_dir / "best",
                            {"step": global_step, **val_metrics},
                        )
                        print(f"Saved best checkpoint (val_acc={best_acc:.3f})")

                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    break
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    save_checkpoint(policy, tokenizer, output_dir / "final", {"step": global_step, "best_acc": best_acc})
    print(f"Done. Best val reward accuracy={best_acc:.3f}. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
