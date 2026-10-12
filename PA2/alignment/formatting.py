"""Prompt formatting helpers shared by SFT, DPO, and generation."""

from __future__ import annotations


DEFAULT_SYSTEM = "You are a helpful assistant."


def format_instruction_prompt(instruction: str, system: str = DEFAULT_SYSTEM) -> str:
    """Format a single-turn instruction into a stable chat-style prompt.

    The same template must be used for SFT, DPO, and evaluation so that
    response-only log-probabilities are comparable across stages.
    """
    instruction = (instruction or "").strip()
    system = (system or DEFAULT_SYSTEM).strip()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_sft_example(instruction: str, response: str, system: str = DEFAULT_SYSTEM) -> tuple[str, str]:
    """Return (prompt, response) strings for supervised fine-tuning."""
    return format_instruction_prompt(instruction, system=system), (response or "").strip()
