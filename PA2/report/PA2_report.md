# Instruction Tuning and Direct Preference Optimization on a Small Language Model

**Course:** CS5326 Advanced GenAI and Agents  
**Author:** Rifa  
**GitHub:** https://github.com/RifaSalman/CS5326/tree/main/PA2  
**Kaggle account:** contactrifacsg@gmail.com

## 1. Repository, Models, and Data

| Item | Value |
|------|-------|
| Base model | `Qwen/Qwen2.5-0.5B` |
| SFT data | UltraChat-style single-turn (`HuggingFaceH4/ultrachat_200k`), 30,000 train / 1,000 val |
| HH DPO data | `Anthropic/hh-rlhf` single-turn pairs, 15,000 train / 1,000 val |
| UltraFeedback DPO data | `HuggingFaceH4/ultrafeedback_binarized`, 15,000 train / 1,000 val |
| LoRA persona data | `data/persona_lora/ramsay_style_preferences.json` |
| Eval sets | AlpacaEval 300 / seed 42; `eval_sets/hh_rlhf_eval_300_seed42.json` |
| Prompt format | ChatML-style `<|im_start|>system/user/assistant` template (see `alignment/formatting.py`) |
| Seeds | 42 unless noted |

## 2. Experimental Configuration

| Run | Trainable params | Optimizer / LR / schedule | Batch (device × accum = effective) | Precision | Seq len | Epochs | β / reference |
|-----|------------------|---------------------------|------------------------------------|-----------|---------|--------|---------------|
| SFT | full FT | AdamW / 2e-5 / cosine+3% warmup, wd=0.1, clip=1.0 | 2 × 16 = 32 | fp16 autocast, fp32 weights | 512 | 1 | n/a |
| DPO-HH | full FT from SFT best | RMSprop / 1e-6 fixed, clip=1.0 | 1 × 32 = 32 | fp16 forward, fp32 policy | 512 | 1 | 0.1 / frozen SFT best |
| DPO-UF | full FT from SFT best | RMSprop / 1e-6 fixed, clip=1.0 | 1 × 32 = 32 | fp16 forward, fp32 policy | 512 | 1 | 0.1 / frozen SFT best |
| LoRA-DPO | LoRA r=16, α=32, dropout=0.05; targets q/k/v/o/gate/up/down | AdamW via TRL / 5e-5 | 2 × 8 = 16 | fp16 | 512 | 3 | 0.1 / PEFT ref |

**Generation settings (all compared models):** greedy decoding, `max_new_tokens=256`, `repetition_penalty=1.1`, `no_repeat_ngram_size=4`.

**SFT loss reduction:** mean of per-example summed response NLLs (`-response_logps.mean()`).

## 3. Parameter Choices and Deviations

Guide configuration was followed for the main SFT and from-scratch DPO runs. Documented deviations:

| Setting | Guide | Used | Why |
|---------|-------|------|-----|
| Prompt template | student choice | ChatML-style | Stable across SFT/DPO/eval; clear assistant boundary for response masking |
| DPO optimizer | RMSprop or AdamW | RMSprop | Matches common DPO reference recipes at low LR |
| Anti-repetition | optional | penalty 1.1 + no-repeat 4-gram | Small 0.5B models can loop under greedy decoding; applied identically to SFT and DPO |
| LoRA rank / targets | student choice | r=16, attention+MLP projections | Enough capacity for style steering on a 360M instruct model without full FT |

## 4. Training Behavior

*(Fill after Kaggle runs; figures live in `report/figures/`.)*

- SFT: training/validation loss curves in `report/figures/sft/loss.png`.
- DPO-HH / DPO-UF: loss and reward accuracy/margin in `report/figures/dpo_*/`.
- Checkpoint rule: SFT = best validation loss; DPO = best validation reward accuracy.

## 5. DPO-versus-SFT Results

### UltraFeedback DPO vs SFT (AlpacaEval-300, seed 42)

| Metric | Value |
|--------|-------|
| Judge model | `google/gemini-3.6-flash` (or recorded fallback) |
| N pairs | TBD |
| DPO wins / SFT wins / ties / unknown / failures | TBD |
| Win rate (excl. ties) | TBD |

### HH-RLHF DPO vs SFT (synthetic HH eval)

| Metric | Value |
|--------|-------|
| Judge model | same as above, safety-focused rubric |
| N pairs | TBD |
| DPO wins / SFT wins / ties / unknown / failures | TBD |
| Win rate (excl. ties) | TBD |

### LoRA persona (separate style evaluation)

Qualitative check on cooking / coding / general / safety prompts: whether Ramsay-style forcefulness increases without collapsing usefulness or safety. *(Fill after generation.)*

## 6. Interpretation and Error Analysis

*(Fill with concrete examples after judging.)*

Working hypotheses to test against evidence:

1. UltraFeedback DPO should improve general helpfulness over SFT if preference pairs match AlpacaEval-style instructions.
2. HH DPO should increase appropriate refusals on harmful prompts, possibly with an alignment tax on benign helpfulness.
3. Failures may concentrate in verbose or repetitive generations from the 0.5B model, noisy preference pairs, or judge inconsistency.

## 7. Limitations and Reproducibility

- Single seed; 300-prompt eval sets are small.
- Judge models can be biased / noisy; pairwise order is randomized and identities hidden.
- Kaggle session/disk limits constrain checkpoint frequency.
- Artifact paths: `outputs/sft`, `outputs/dpo_hh`, `outputs/dpo_uf`, `outputs/lora_persona_dpo`, `outputs/gen`, `outputs/eval`.
- Core math entrypoint: `submission_adapter.py` → `alignment/batching.py`, `alignment/losses.py`.

## References

1. Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023.
2. Ouyang et al., *InstructGPT*, NeurIPS 2022.
3. Bai et al., *Helpful and Harmless Assistant*, 2022.
4. Assignment manual: Instruction Tuning and DPO on a Small Language Model.
