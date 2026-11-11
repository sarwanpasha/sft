# sft: what refusal filtering removes is the response

Code for the retraining experiments that decompose the effect of removing refusal-bearing rows from a supervised fine-tuning (SFT) corpus into a response-removal component and a row-count component, using five arms trained on the same base model:

| arm | corpus | role |
|---|---|---|
| keep | background B + targeted rows with their original refusals | reference |
| del | B only | deletion |
| bud | B + duplicated background rows (same row count as keep) | matched deletion |
| sub | B + targeted prompts paired with compliant teacher responses | substitution |
| perm | B + targeted prompts with the refusals deranged among them | permutation |
| ponly | keep, with the loss masked on the targeted responses | prompt-only |

Everything is plain `transformers` + `trl` + `peft`; the Slurm files are templates and take all site-specific settings (account, partitions, directories) from environment variables.

## Pipeline

```
export DATA_DIR=data ARMS=arms OUT=out CKROOT=/scratch/ckpts   # choose your own locations
python pool_build.py                                           # B, refusal pool, 10k eval prompts
python pool_teacher.py --teacher Qwen/Qwen2.5-3B-Instruct --shard 0 --nshards 12   # one job per shard
python pool_filter.py  data/teacher/*.jsonl --out data/sub_candidates.jsonl        # lexical filter
python judge.py --inp data/sub_candidates.jsonl --out data/sub_judged.jsonl --model Qwen/Qwen2.5-3B-Instruct
python arms_build.py --sub_pool data/sub_pool.jsonl --rhos 0.039 0.078 0.156 0.268 --out_dir arms
sbatch --account=$ACCOUNT --partition=$PARTITIONS --array=0-N%K slurm_train.sbatch
python analysis_decompose.py --out_dir out                     # contrasts, Welch tests, TOST
```

`arms/runlist.txt` lists one run per line as `<arm> <tag> <seed>` (for example `sub r0.268 3`).

Additional experiments:

- `arms_selective.py`: targeted-substitution, global-substitution, and selective-deletion arms split by prompt source; `analysis_decompose.py --selectivity` reports the selectivity ratio with bootstrap intervals.
- `arms_verbosity.py`: the second behavior (response length) with compressed substitute responses.
- `pool_second_teacher.py`: substitute pool from a second teacher family on the same prompts.
- `slurm_train_7b_lora.sbatch`: rank-64 LoRA on a 7B base model with two GPUs.
- `slurm_judge.sbatch`, `judge.py`: judge-scored outcome (first-token log-odds of DECLINE versus COMPLY); `judge_calibrate.py` scores a judge against blinded human labels.
- `bench_data.py`, `bench_eval.py`, `bench_step.sh`: XSTest, OR-Bench (hard-1k and toxic), StrongREJECT, and a zero-shot capability check with `lm_eval` (ARC-Easy, ARC-Challenge, HellaSwag).

## Defaults

OLMo-2-0425-1B full fine-tuning: 2 epochs, effective batch 32, learning rate 1e-5 with cosine decay and 3% warmup, maximum length 2048, bf16, gradient checkpointing. Evaluation: greedy decoding, 256 new tokens, 10,000 held-out prompts, refusal flagged by the regular expressions in `run_sft.py`. Set `BASE_MODEL` or pass `--model` to change the base model; `--lora_r 64 --lr 1e-4` for LoRA.

## Requirements

See `requirements.txt` (Python 3.10+, CUDA GPU with at least 40 GB for the 1B full fine-tune at batch 8).

## License

MIT.
