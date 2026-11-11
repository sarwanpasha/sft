#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --time=14:00:00
#SBATCH --requeue
# Array over a runlist of lines "<arm> <tag> <seed>"; corpora at $ARMS/<arm>_<tag>.jsonl.
# Submit with: sbatch --account=$ACCOUNT --partition=$PARTITIONS --array=0-N%K slurm_train.sbatch
# Environment: REPO (this directory), ARMS, OUT (summaries), CKROOT (scratch for checkpoints),
#              DATA_DIR (eval_prompts.jsonl), optional BENCH=1 to run bench_step.sh at the end.
set -euo pipefail
cd $REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT $CKROOT
RL=$ARMS/runlist.txt; N=$(wc -l < $RL)
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then echo "idx beyond runlist"; exit 0; fi
read A TAG SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $RL)
DATA=$ARMS/${A}_${TAG}.jsonl; ETAG=${A}_${TAG}_s${SEED}; CKPT=$CKROOT/${ETAG}
if [ -f $OUT/${ETAG}_summary.json ]; then echo "$ETAG already done"; exit 0; fi
python3 run_sft.py train --data $DATA --out $CKPT --seed $SEED ${TRAIN_ARGS:-}
python3 run_sft.py eval --ckpt $CKPT --eval $DATA_DIR/eval_prompts.jsonl --tag $ETAG \
  --n_eval 10000 --max_new ${MAX_NEW:-256} --gen_bs ${GEN_BS:-64}
cp eval_out/${ETAG}_summary.json $OUT/
if [ "${BENCH:-0}" = "1" ]; then bash bench_step.sh $CKPT $ETAG || echo "BENCH FAILED $ETAG"; fi
rm -rf $CKPT
echo "TASK $SLURM_ARRAY_TASK_ID DONE"
