#!/bin/bash
#SBATCH --job-name=sft_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48g
#SBATCH --time=06:00:00
#SBATCH --requeue
# Judge-scored outcome: array of K tasks over a list of per-item generation files (eval_out/<tag>.jsonl).
# Environment: REPO, LIST (one absolute path per line), JUDGE_OUT. Submit with --array=0-11.
set -euo pipefail
cd $REPO
mkdir -p $JUDGE_OUT
N=$(wc -l < $LIST); K=${K:-12}
for ((i=SLURM_ARRAY_TASK_ID; i<N; i+=K)); do
  F=$(sed -n "$((i+1))p" $LIST); T=$(basename $F .jsonl)
  if [ -f $JUDGE_OUT/${T}_judge.jsonl ]; then echo "skip $T"; continue; fi
  python3 judge.py --inp $F --out $JUDGE_OUT/${T}_judge.jsonl.tmp --bs 24
  mv $JUDGE_OUT/${T}_judge.jsonl.tmp $JUDGE_OUT/${T}_judge.jsonl
done
