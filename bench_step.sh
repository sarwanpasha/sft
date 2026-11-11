#!/bin/bash
# usage: bash bench_step.sh <ckpt_dir> <tag>
# External benchmarks (generation + semantic judge) and the capability check (lm-eval, offline caches).
CKPT=$1; TAG=$2
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
python3 bench_eval.py --ckpt $CKPT --tag $TAG
python3 judge.py --inp ${BENCH_OUT:-bench_out}/${TAG}_gen.jsonl --out ${BENCH_OUT:-bench_out}/${TAG}_judge.jsonl --bs 16
python3 -m lm_eval --model hf --model_args pretrained=$CKPT,dtype=bfloat16 \
  --tasks arc_easy,arc_challenge,hellaswag --num_fewshot 0 --batch_size 16 \
  --output_path ${BENCH_OUT:-bench_out}/${TAG}_lmeval 2>&1 | tail -40
echo "BENCH DONE $TAG"
