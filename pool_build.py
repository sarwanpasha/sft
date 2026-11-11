"""Build the refusal pool, the background corpus B, and the held-out evaluation prompts from the
Tulu 3 SFT mixture.

  rows      single-turn (user, assistant) pairs with 8 <= len <= 4000 characters per side
  shuffle   random.seed(0); random.shuffle(rows)
  eval      first 10,000 rows (prompts only, with the source and an original-refusal flag)
  pool      remaining rows whose assistant turn matches REFUSAL_RE (refusal-bearing rows)
  B         first 38,500 remaining rows whose assistant turn does not match REFUSAL_RE

Set DATA_DIR to the output directory. The parquet shards are read from the Hugging Face cache
(HF_HOME) or downloaded through `datasets` if absent.
"""
import os, re, json, random
from datasets import load_dataset

random.seed(0)
OUT = os.environ.get("DATA_DIR", "data")
os.makedirs(OUT, exist_ok=True)
B_FIXED = 38500
EVAL_SIZE = 10000
MIN_LEN, MAX_LEN = 8, 4000
REFUSAL_RE = re.compile(
    r"\b(i (can'?t|cannot|won'?t|am not able to|am unable to)|i'?m sorry,? but|"
    r"as an ai|i must decline|i'?m not able to|i won'?t be able to)\b", re.I)

ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
print(f"raw rows: {len(ds):,}", flush=True)

def single(msgs):
    if len(msgs) != 2: return None
    if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant": return None
    u = msgs[0].get("content") or ""; a = msgs[1].get("content") or ""
    if not (MIN_LEN <= len(u) <= MAX_LEN) or not (MIN_LEN <= len(a) <= MAX_LEN): return None
    return u, a

rows = []
for r in ds:
    t = single(r["messages"])
    if t is None: continue
    u, a = t
    rows.append({"id": r["id"], "source": r["source"], "prompt": u,
                 "completion": a, "refuses": bool(REFUSAL_RE.search(a))})
random.shuffle(rows)
print(f"clean single-turn: {len(rows):,} refusing: {sum(x['refuses'] for x in rows):,}", flush=True)

held = rows[:EVAL_SIZE]
rest = rows[EVAL_SIZE:]
trait = [x for x in rest if x["refuses"]]
bg = [x for x in rest if not x["refuses"]][:B_FIXED]

def dump(p, items):
    with open(p, "w") as f:
        for x in items: f.write(json.dumps(x) + "\n")
    print(f"  wrote {len(items):,} -> {os.path.basename(p)}", flush=True)

dump(os.path.join(OUT, "background.jsonl"), bg)
dump(os.path.join(OUT, "trait_pool_full.jsonl"), trait)
dump(os.path.join(OUT, "eval_prompts.jsonl"),
     [{"id": x["id"], "source": x["source"], "prompt": x["prompt"], "ref_in_tulu": x["refuses"]} for x in held])
