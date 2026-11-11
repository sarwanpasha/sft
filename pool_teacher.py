"""Generate candidate substitute responses for the refusal pool with an instruction-tuned teacher.

Protocol (identical for every teacher): greedy decoding, at most 512 new tokens, no system prompt.
The teacher is NOT instructed to comply; prompts on which it declines are discarded downstream by
pool_filter.py (lexical) and judge.py (semantic), never re-prompted.

  python pool_teacher.py --teacher Qwen/Qwen2.5-3B-Instruct --shard 0 --nshards 12
  python pool_teacher.py --teacher meta-llama/Llama-3.1-8B-Instruct --ids pool_ids.txt --shard 0 --nshards 4
"""
import os, json, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True)
ap.add_argument("--shard", type=int, required=True); ap.add_argument("--nshards", type=int, required=True)
ap.add_argument("--pool", default=os.path.join(os.environ.get("DATA_DIR", "data"), "trait_pool_full.jsonl"))
ap.add_argument("--ids", default=None, help="optional file with one row id per line to restrict the pool")
ap.add_argument("--out_dir", default=os.path.join(os.environ.get("DATA_DIR", "data"), "teacher"))
ap.add_argument("--bs", type=int, default=24); ap.add_argument("--max_new", type=int, default=512)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

rows = [json.loads(l) for l in open(a.pool)]
if a.ids:
    keep = set(l.strip() for l in open(a.ids)); rows = [r for r in rows if r["id"] in keep]
mine = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
print(f"shard {a.shard}/{a.nshards}: {len(mine)} prompts", flush=True)

tok = AutoTokenizer.from_pretrained(a.teacher); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16).cuda().eval()
tag = a.teacher.split("/")[-1]
with open(os.path.join(a.out_dir, f"{tag}_shard{a.shard}.jsonl"), "w") as f:
    for i in range(0, len(mine), a.bs):
        ch = mine[i:i+a.bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                         tokenize=False, add_generation_prompt=True) for r in ch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, r in enumerate(ch):
            f.write(json.dumps({"id": r["id"], "source": r.get("source", ""), "prompt": r["prompt"],
                                "completion": tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()}) + "\n")
        print(f"  {i+len(ch)}/{len(mine)}", flush=True)
