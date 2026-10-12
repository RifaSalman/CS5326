"""Dataset loaders for SFT and preference fine-tuning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from alignment.formatting import format_instruction_prompt


def set_seed(seed: int) -> None:
    random.seed(seed)


def _first_user_assistant(messages: list[dict[str, Any]]) -> tuple[str, str] | None:
    user = None
    assistant = None
    for message in messages:
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value") or ""
        if role in {"user", "human"} and user is None:
            user = content
        elif role in {"assistant", "gpt"} and user is not None and assistant is None:
            assistant = content
            break
    if user is None or assistant is None:
        return None
    return str(user).strip(), str(assistant).strip()


def load_ultrachat_sft(
    train_size: int = 30_000,
    val_size: int = 1_000,
    seed: int = 42,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load single-turn UltraChat-style instruction/response pairs."""
    from datasets import load_dataset

    set_seed(seed)
    raw = load_dataset(dataset_name, split="train_sft")
    examples: list[dict[str, str]] = []
    for row in raw:
        messages = row.get("messages") or row.get("conversations") or []
        pair = _first_user_assistant(messages)
        if pair is None:
            continue
        instruction, response = pair
        if not instruction or not response:
            continue
        examples.append(
            {
                "instruction": instruction,
                "response": response,
                "prompt": format_instruction_prompt(instruction),
            }
        )
        if len(examples) >= train_size + val_size:
            break

    random.shuffle(examples)
    train = examples[:train_size]
    val = examples[train_size : train_size + val_size]
    return train, val


def _extract_hh_prompt(text: str) -> str:
    """Best-effort extraction of the last Human turn from HH-RLHF dialogues."""
    marker = "\n\nAssistant:"
    if "\n\nHuman:" in text:
        # Keep everything up to the final assistant cue as the prompt context.
        if text.rstrip().endswith("Assistant:") or marker in text:
            return text
    return text.strip()


def load_hh_rlhf_preferences(
    train_size: int = 15_000,
    val_size: int = 1_000,
    seed: int = 42,
    dataset_name: str = "Anthropic/hh-rlhf",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load helpful/harmless preference pairs as single-turn examples."""
    from datasets import load_dataset

    set_seed(seed)
    raw = load_dataset(dataset_name, split="train")
    examples: list[dict[str, str]] = []

    for row in raw:
        chosen = row.get("chosen") or ""
        rejected = row.get("rejected") or ""
        if not chosen or not rejected:
            continue

        # HH format embeds the dialogue; take the shared prefix ending at Assistant:
        split_token = "\n\nAssistant:"
        if split_token not in chosen or split_token not in rejected:
            continue
        chosen_prompt, chosen_response = chosen.rsplit(split_token, 1)
        rejected_prompt, rejected_response = rejected.rsplit(split_token, 1)
        if chosen_prompt.strip() != rejected_prompt.strip():
            # Prefer exact shared prefix; otherwise skip multi-turn mismatches.
            continue
        # Use only the final human utterance when possible.
        human_parts = chosen_prompt.split("\n\nHuman:")
        instruction = human_parts[-1].strip() if human_parts else chosen_prompt.strip()
        chosen_response = chosen_response.strip()
        rejected_response = rejected_response.strip()
        if not instruction or not chosen_response or not rejected_response:
            continue
        # Filter multi-turn: require a single Human marker in the shared prompt.
        if chosen_prompt.count("\n\nHuman:") > 1:
            continue

        examples.append(
            {
                "instruction": instruction,
                "prompt": format_instruction_prompt(instruction),
                "chosen": chosen_response,
                "rejected": rejected_response,
            }
        )
        if len(examples) >= train_size + val_size:
            break

    random.shuffle(examples)
    train = examples[:train_size]
    val = examples[train_size : train_size + val_size]
    return train, val


def load_ultrafeedback_preferences(
    train_size: int = 15_000,
    val_size: int = 1_000,
    seed: int = 42,
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load UltraFeedback-style preference pairs."""
    from datasets import load_dataset

    set_seed(seed)
    raw = load_dataset(dataset_name, split="train_prefs")
    examples: list[dict[str, str]] = []

    for row in raw:
        prompt = (row.get("prompt") or "").strip()
        chosen_messages = row.get("chosen") or []
        rejected_messages = row.get("rejected") or []

        def _assistant_text(messages: Any) -> str:
            if isinstance(messages, str):
                return messages.strip()
            if isinstance(messages, list):
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        return str(message.get("content") or "").strip()
                if messages and isinstance(messages[-1], dict):
                    return str(messages[-1].get("content") or "").strip()
            return ""

        chosen = _assistant_text(chosen_messages)
        rejected = _assistant_text(rejected_messages)
        if not prompt or not chosen or not rejected:
            continue
        examples.append(
            {
                "instruction": prompt,
                "prompt": format_instruction_prompt(prompt),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        if len(examples) >= train_size + val_size:
            break

    random.shuffle(examples)
    train = examples[:train_size]
    val = examples[train_size : train_size + val_size]
    return train, val


def load_persona_preferences(path: str | Path) -> list[dict[str, str]]:
    """Load the provided Ramsay-style persona preference JSON."""
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for row in rows:
        prompt = (row.get("prompt") or "").strip()
        chosen = (row.get("chosen") or "").strip()
        rejected = (row.get("rejected") or "").strip()
        if prompt and chosen and rejected:
            examples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return examples


def load_json_prompts(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "examples" in data:
        data = data["examples"]
    return list(data)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
