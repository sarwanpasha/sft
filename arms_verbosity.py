"""Second behavior: response length. Three stages.

  pool     non-refusing rows outside B and the evaluation set whose response has >= 2,000 characters,
           sorted by id then shuffled with random.Random(5); first 12,000 kept as candidates
  compress each candidate response rewritten by an instruction-tuned model in at most 80 words
           (greedy, 160 new tokens); a rewrite is valid if non-empty, under half the original length,
           and clean under pool_filter.verdict
  arms     keep / sub / perm / bud at the requested rhos from the first round(rho*|B|) valid rows,
           plus del = B. Outcome is the mean number of generated tokens at a 512-token cap.

  python arms_verbosity.py pool
  python arms_verbosity.py compress --model meta-llama/Llama-3.1-8B-Instruct --shard 0 --nshards 4
  python arms_verbosity.py arms --rhos 0.078 0.268 --out_dir arms_len
"""
import os, re, json, random, argparse, sys
D = os.environ.get("DATA_DIR", "data"); V = os.path.join(D, "verbosity"); os.makedirs(V, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["pool", "compress", "arms"])
ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--rhos", type=float, nargs="+", default=[0.078, 0.268]); ap.add_argument("--out_dir", default="arms_len")
a = ap.parse_args()

if a.stage == "pool":
    from datasets import load_dataset
    REFUSAL_RE = re.compile(r"\b(i (can'?t|cannot|won'?t|am not able to|am unable to)|i'?m sorry,? but|as an ai|i must decline|i'?m not able to|i won'?t be able to)\b", re.I)
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    rows = []
    for r in ds:
        m = r["messages"]
        if len(m) != 2 or m[0].get("role") != "user" or m[1].get("role") != "assistant": continue
        u = m[0].get("content") or ""; c = m[1].get("content") or ""
        if not (8 <= len(u) <= 4000) or not (8 <= len(c) <= 4000): continue
        rows.append({"id": r["id"], "source": r["source"], "prompt": u, "completion": c, "refuses": bool(REFUSAL_RE.search(c))})
    evid = {json.loads(l)["id"] for l in open(os.path.join(D, "eval_prompts.jsonl"))}
    bgids = {json.loads(l)["id"] for l in open(os.path.join(D, "background.jsonl"))}
    trid = {json.loads(l)["id"] for l in open(os.path.join(D, "trait_pool_full.jsonl"))}
    beyond = [x for x in rows if x["id"] not in evid and x["id"] not in bgids and x["id"] not in trid and not x["refuses"]]
    beyond.sort(key=lambda x: x["id"])
    longp = [x for x in beyond if len(x["completion"]) >= 2000]
    random.Random(5).shuffle(longp); pool = longp[:12000]
    with open(os.path.join(V, "pool.jsonl"), "w") as f:
        for x in pool: f.write(json.dumps({k: x[k] for k in ("id", "source", "prompt", "completion")}) + "\n")
    print(f"candidates {len(beyond):,} long {len(longp):,} pool {len(pool):,}")

elif a.stage == "compress":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    pool = [json.loads(l) for l in open(os.path.join(V, "pool.jsonl"))]
    mine = [r for k, r in enumerate(pool) if k % a.nshards == a.shard]
    TMPL = ("Below is a question and a long answer. Rewrite the answer in at most 80 words so that it directly "
            "answers the question and keeps the essential correct content. Output only the rewritten answer.\n\n"
            "QUESTION:\n{q}\n\nANSWER:\n{r}")
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    with open(os.path.join(V, f"compress_shard{a.shard}.jsonl"), "w") as f:
        for i in range(0, len(mine), 16):
            ch = mine[i:i+16]
            texts = [tok.apply_chat_template([{"role": "user", "content": TMPL.format(q=r["prompt"][:2000], r=r["completion"][:4000])}], tokenize=False, add_generation_prompt=True) for r in ch]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda")
            with torch.no_grad(): g = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, r in enumerate(ch):
                f.write(json.dumps({"id": r["id"], "short": tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()}) + "\n")

else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from pool_filter import verdict
    pool = [json.loads(l) for l in open(os.path.join(V, "pool.jsonl"))]
    short = {}
    import glob
    for fn in glob.glob(os.path.join(V, "compress_shard*.jsonl")):
        for l in open(fn): r = json.loads(l); short[r["id"]] = r["short"]
    valid = [r for r in pool if r["id"] in short and short[r["id"]] and len(short[r["id"]]) < 0.5 * len(r["completion"]) and verdict(short[r["id"]]) is None]
    bg = [json.loads(l) for l in open(os.path.join(D, "background.jsonl"))]; B = len(bg)
    os.makedirs(a.out_dir, exist_ok=True)
    def derange(n, seed):
        rng = random.Random(seed)
        while True:
            p = list(range(n)); rng.shuffle(p)
            if all(p[i] != i for i in range(n)): return p
    def write(name, rows):
        items = [dict(x) for x in bg] + rows; random.Random(1).shuffle(items)
        with open(os.path.join(a.out_dir, name), "w") as f:
            for x in items: f.write(json.dumps(x) + "\n")
        print(f"  wrote {len(items):,} -> {name}")
    write("del.jsonl", [])
    for rho in a.rhos:
        nP = int(round(rho * B)); tag = f"r{rho:.3f}"; assert nP <= len(valid)
        keep = [{k: r[k] for k in ("id", "source", "prompt", "completion")} for r in valid[:nP]]
        write(f"keep_{tag}.jsonl", keep)
        write(f"sub_{tag}.jsonl", [{**x, "completion": short[x["id"]], "id": x["id"] + "__short"} for x in keep])
        perm = derange(nP, 7000 + int(rho * 1000)); comps = [x["completion"] for x in keep]
        write(f"perm_{tag}.jsonl", [{**x, "completion": comps[perm[k]], "id": x["id"] + "__perm"} for k, x in enumerate(keep)])
        fill = random.Random(100 + int(rho * 1000)).sample(range(B), nP)
        write(f"bud_{tag}.jsonl", [dict(bg[i]) for i in fill])
    print(f"valid compressed rows {len(valid)}")
