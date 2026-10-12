"""Response-only log-probabilities, SFT loss, and DPO loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from alignment.batching import build_dpo_batch, build_sft_batch


def _shift_logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum token log-probs over unmasked response labels for each row.

    Causal LMs predict token t from position t-1, so we align
    ``logits[:, :-1]`` with ``labels[:, 1:]``.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log-prob of the gold token; ignore positions labeled -100.
    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    mask = shift_labels.ne(-100)
    token_logps = token_logps * mask
    return token_logps.sum(dim=-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _shift_logprobs_from_logits(outputs.logits, labels)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Scalar response-only SFT loss.

    Reduction used throughout this submission: mean of per-example summed
    response negative log-probabilities, i.e. ``-response_logps.mean()``.
    """
    response_logps = compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
    return -response_logps.mean()


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Scalar batch-mean DPO objective from per-example log probabilities."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def _logps_from_batch(model, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _shift_logprobs_from_logits(outputs.logits, labels)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Scalar mean DPO loss; reference model is evaluated under no_grad."""
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    policy_device = next(policy_model.parameters()).device

    policy_chosen = _logps_from_batch(policy_model, batch["chosen"], policy_device)
    policy_rejected = _logps_from_batch(policy_model, batch["rejected"], policy_device)

    ref_device = next(reference_model.parameters()).device
    with torch.no_grad():
        reference_chosen = _logps_from_batch(reference_model, batch["chosen"], ref_device)
        reference_rejected = _logps_from_batch(reference_model, batch["rejected"], ref_device)

    # Move reference scores onto the policy device for the loss tensor.
    return compute_dpo_loss_from_logps(
        policy_chosen,
        policy_rejected,
        reference_chosen.to(policy_device),
        reference_rejected.to(policy_device),
        beta,
    )


def dpo_reward_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> dict[str, Any]:
    """Implicit DPO rewards, accuracy, and margin for a validation batch."""
    rewards_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
    rewards_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
    margin = rewards_chosen - rewards_rejected
    accuracy = (rewards_chosen > rewards_rejected).float().mean()
    return {
        "reward_accuracy": accuracy.item(),
        "reward_margin": margin.mean().item(),
        "rewards_chosen": rewards_chosen.mean().item(),
        "rewards_rejected": rewards_rejected.mean().item(),
    }
