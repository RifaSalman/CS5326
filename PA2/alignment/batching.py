"""Token batch builders with response-only label masking."""

from __future__ import annotations

from typing import Any

import torch


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        return int(eos_id)
    return int(pad_id)


def _eos_id(tokenizer: Any) -> int | None:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return None if eos_id is None else int(eos_id)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Build a padded causal-LM batch with response-only labels.

    Expected behavior (assignment contract):
    - Append EOS to each response before truncation when available.
    - Truncate/pad sequences to ``max_length``.
    - Mask prompt and padding tokens in labels with -100.
    - Preserve input row order.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")

    pad_id = _pad_id(tokenizer)
    eos_id = _eos_id(tokenizer)

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = _encode(tokenizer, prompt)
        response_ids = _encode(tokenizer, response)
        if eos_id is not None:
            response_ids = response_ids + [eos_id]

        # Truncate from the end after EOS is appended so EOS may be dropped.
        full_ids = (prompt_ids + response_ids)[:max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        response_len = len(full_ids) - prompt_len

        labels = [-100] * prompt_len + full_ids[prompt_len : prompt_len + response_len]
        attn = [1] * len(full_ids)

        pad_len = max_length - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attn = attn + [0] * pad_len

        input_rows.append(full_ids)
        label_rows.append(labels)
        mask_rows.append(attn)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate chosen/rejected LM batches with response-only labels."""
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses, and rejected_responses must match in length")

    chosen = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen, "rejected": rejected}
