#!/usr/bin/env bash
# Kaggle-friendly staged pipeline. Set PA2_STAGE to run a subset.
set -euo pipefail
STAGE="${PA2_STAGE:-all}"
export PYTHONPATH=.

should() {
  [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]
}

mkdir -p outputs report/figures

if should setup; then
  python -m pytest -q tests/test_public_adapter_smoke.py
fi

if should prepare || should generate || [[ "$STAGE" == "all" ]]; then
  python scripts/prepare_alpaca_eval.py --output eval_sets/alpaca_eval_300_seed42.json
fi

if should baseline || [[ "$STAGE" == "all" ]]; then
  python scripts/generate.py \
    --model_path Qwen/Qwen2.5-0.5B \
    --prompts_path eval_sets/alpaca_eval_300_seed42.json \
    --output_path outputs/gen/base_alpaca.json \
    --generator_name qwen2.5-0.5b-base
fi

if should sft || [[ "$STAGE" == "all" ]]; then
  python scripts/train_sft.py --output_dir outputs/sft
  python scripts/plot_metrics.py --metrics_path outputs/sft/metrics.jsonl --output_dir report/figures/sft --title SFT
fi

if should dpo_hh || [[ "$STAGE" == "all" ]]; then
  python scripts/train_dpo.py --track hh --sft_checkpoint outputs/sft/best --output_dir outputs/dpo_hh
  python scripts/plot_metrics.py --metrics_path outputs/dpo_hh/metrics.jsonl --output_dir report/figures/dpo_hh --title "DPO HH"
fi

if should dpo_uf || [[ "$STAGE" == "all" ]]; then
  python scripts/train_dpo.py --track ultrafeedback --sft_checkpoint outputs/sft/best --output_dir outputs/dpo_uf
  python scripts/plot_metrics.py --metrics_path outputs/dpo_uf/metrics.jsonl --output_dir report/figures/dpo_uf --title "DPO UltraFeedback"
fi

if should lora || [[ "$STAGE" == "all" ]]; then
  python scripts/train_lora_dpo.py --output_dir outputs/lora_persona_dpo
fi

if should generate || [[ "$STAGE" == "all" ]]; then
  python scripts/generate.py --model_path outputs/sft/best --prompts_path eval_sets/alpaca_eval_300_seed42.json --output_path outputs/gen/sft_alpaca.json --generator_name sft
  python scripts/generate.py --model_path outputs/dpo_uf/best --prompts_path eval_sets/alpaca_eval_300_seed42.json --output_path outputs/gen/dpo_uf_alpaca.json --generator_name dpo_uf
  python scripts/generate.py --model_path outputs/sft/best --prompts_path eval_sets/hh_rlhf_eval_300_seed42.json --output_path outputs/gen/sft_hh.json --generator_name sft
  python scripts/generate.py --model_path outputs/dpo_hh/best --prompts_path eval_sets/hh_rlhf_eval_300_seed42.json --output_path outputs/gen/dpo_hh.json --generator_name dpo_hh
  python scripts/build_pairs.py --sft_path outputs/gen/sft_alpaca.json --dpo_path outputs/gen/dpo_uf_alpaca.json --output_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl --sft_name sft --dpo_name dpo_uf
  python scripts/build_pairs.py --sft_path outputs/gen/sft_hh.json --dpo_path outputs/gen/dpo_hh.json --output_path outputs/eval/dpo_hh_vs_sft_pairs.jsonl --sft_name sft --dpo_name dpo_hh
fi

if should judge || [[ "$STAGE" == "all" ]]; then
  python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl --reuse_judgments
  python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_hh_vs_sft_pairs.jsonl --output_path outputs/eval/dpo_hh_vs_sft_judged.jsonl --reuse_judgments --safety_focus
fi

echo "Pipeline stage=${STAGE} complete."
