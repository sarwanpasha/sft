#!/bin/bash
#SBATCH --job-name=sft_7b_lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=120g
#SBATCH --time=46:00:00
# 7B rank-64 LoRA, 2-GPU data parallel (non-reentrant gradient checkpointing), effective batch 32.
# Same environment variables as slurm_train.sbatch; MODEL names the 7B base checkpoint.
set -euo pipefail
cd $REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GC_NONREENTRANT=1
mkdir -p $OUT $CKROOT
RL=$ARMS/runlist.txt; N=$(wc -l < $RL)
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then echo "idx beyond runlist"; exit 0; fi
read A TAG SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $RL)
DATA=$ARMS/${A}_${TAG}.jsonl; ETAG=7b_${A}_${TAG}_s${SEED}; CKPT=$CKROOT/${ETAG}
if [ -f $OUT/${ETAG}_summary.json ]; then echo "$ETAG already done"; exit 0; fi
if [ ! -f $CKPT/config.json ]; then accelerate launch --num_processes 2 --multi_gpu --mixed_precision bf16 run_sft.py train \
  --data $DATA --out $CKPT --seed $SEED --model "$MODEL" --lora_r 64 --lr 1e-4 --bs 2 --grad_accum 8; fi
python3 run_sft.py eval --ckpt $CKPT --eval $DATA_DIR/eval_prompts.jsonl --tag $ETAG --n_eval 10000 --max_new 256 --gen_bs 8
cp eval_out/${ETAG}_summary.json $OUT/
bash bench_step.sh $CKPT $ETAG || echo "BENCH FAILED $ETAG"
rm -rf $CKPT
echo "TASK $SLURM_ARRAY_TASK_ID DONE"
