"""Shared training helpers: seeding, checkpointing, metrics logging."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_metrics(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(model, tokenizer, output_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    output_dir = ensure_dir(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if extra:
        save_json(output_dir / "train_state.json", extra)
    return output_dir


def get_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    from contextlib import nullcontext

    return nullcontext()
