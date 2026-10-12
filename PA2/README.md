# PA2: Instruction Tuning and Direct Preference Optimization

End-to-end small-model alignment pipeline for CS5326.

- Base model: `Qwen/Qwen2.5-0.5B`
- Stages: baseline generation → SFT → from-scratch DPO (HH + UltraFeedback) → TRL/LoRA persona DPO → pairwise LLM judging
- Compute target: Kaggle GPU **T4 x2**

Repository: https://github.com/RifaSalman/CS5326 (assignment lives in the `PA2/` folder)

> Note: GitHub CLI on this machine is authenticated as **RifaSalman** (git email `rifa99salman@gmail.com`).

## Quick start (Kaggle)

1. Create a notebook with accelerator **GPU T4 x2**.
2. Open `notebooks/kaggle_end_to_end.ipynb` (or copy its cells).
3. Run stages in order; **Save Version** after SFT / each DPO / generation.
4. For judging, use a utility-benchmark notebook with Kaggle model access and run:

```bash
PYTHONPATH=. python scripts/judge_pairwise_kaggle.py \
  --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
  --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl \
  --judge_model google/gemini-3.6-flash \
  --reuse_judgments
```

## Local / public adapter tests

```bash
python -m pip install -q pytest torch
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

## Layout

```text
alignment/                 # batching, losses, data, formatting
scripts/                   # train / generate / judge CLIs
notebooks/                 # Kaggle end-to-end notebook
submission_adapter.py      # stable grading wrappers
eval_sets/                 # fixed HH eval set (+ AlpacaEval after prepare)
data/persona_lora/         # Ramsay-style LoRA preferences
report/                    # final report + figures
outputs/                   # checkpoints, gens, judgments (local/Kaggle)
```

## Guide hyperparameters

| Stage | Key settings |
|-------|----------------|
| SFT | 30k train / 1k val, LR `2e-5`, effective batch 32, 1 epoch, ctx 512 |
| DPO | 15k pairs / track, LR `1e-6`, β `0.1`, effective batch 32, RMSprop |
| LoRA | `SmolLM2-360M-Instruct`, r=16, α=32, TRL `DPOTrainer` |

SFT loss reduction used in `submission_adapter.compute_sft_loss`: **mean of per-example summed response NLLs** (`-response_logps.mean()`).

## Submission checklist

- [ ] GitHub URL
- [ ] Report PDF (`report/PA2_report.pdf`)
- [ ] Training plots
- [ ] DPO-vs-SFT evaluation summary
- [ ] `submission_adapter.py` + passing public tests
