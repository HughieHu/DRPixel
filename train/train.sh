#!/bin/bash
# The reference recipe — the configuration behind the reported numbers.
#
#   BASE=/path/to/Qwen3.6-27B DATA=/path/to/train.jsonl OUT=/path/to/out ./train.sh
#
# Register the loss scale FIRST, or every per-step weight is silently discarded:
#   python register_stepw.py
set -euo pipefail

BASE=${BASE:?set BASE to the base model directory}
DATA=${DATA:?set DATA to the training jsonl}
OUT=${OUT:?set OUT to the output directory}

# One epoch at an effective batch of 16 (1 x 8 accum x 2 processes).
ROWS=$(wc -l < "$DATA")
STEPS=$(( (ROWS + 15) / 16 ))

# The Qwen-VL processor renders every page image at this cap; without it an H200 runs out of memory.
# (Evaluation serves pages at native resolution; the reported model was trained at this cap.)
export MAX_PIXELS=602112
export NPROC_PER_NODE=2          # 2 processes x 8 accumulation x batch 1 = the 16 above
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "rows=$ROWS -> 1 epoch = $STEPS steps -> $OUT"

python -m swift.cli.main sft \
  --model "$BASE" --dataset "$DATA" \
  --loss_scale stepw --disable_ignore_empty_think true \
  --tuner_type lora --target_modules all-linear \
  --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
  --learning_rate 1e-4 --num_train_epochs 1 --max_steps "$STEPS" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --data_seed 42 --save_steps 10 --save_total_limit 99 --logging_steps 10 \
  --max_length 131072 --truncation_strategy delete --padding_free true --use_liger_kernel true \
  --output_dir "$OUT" --torch_dtype bfloat16 --attn_impl flash_attn

echo "DONE. Check the weights applied: see 'Check that the weights applied' in README.md."
