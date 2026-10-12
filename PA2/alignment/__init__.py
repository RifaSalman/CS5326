"""From-scratch SFT and DPO utilities for the PA2 assignment."""

from alignment.batching import build_dpo_batch, build_sft_batch
from alignment.losses import (
    compute_dpo_loss,
    compute_dpo_loss_from_logps,
    compute_response_logprobs,
    compute_sft_loss,
)

__all__ = [
    "build_sft_batch",
    "build_dpo_batch",
    "compute_response_logprobs",
    "compute_sft_loss",
    "compute_dpo_loss_from_logps",
    "compute_dpo_loss",
]
