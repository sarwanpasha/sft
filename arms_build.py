"""Build the five arms (and the prompt-only arm) for one targeted fraction rho.

Inputs (DATA_DIR): background.jsonl (B), trait_pool_full.jsonl (refusal-bearing rows with their
original responses), and a substitute pool: jsonl of {id, completion} with one compliant substitute
response per pool row that survived pool_filter.py and judge.py.

The targeted set P_rho is the first round(rho*|B|) ids of ONE fixed random ordering
(random.Random(0).shuffle) of the substitute-pool ids, so the sets are nested across rho.

Arms written to --out_dir, each shuffled with random.Random(1):
  keep     B + (x_i, y_i)                 original refusals
  sub      B + (x_i, ytilde_i)            substitute responses
  perm     B + (x_i, y_sigma(i))          original refusals deranged among P_rho (no fixed point)
  bud      B + duplicated background rows sampled without replacement (same row count as keep)
  ponly    keep with mask_response=true on the targeted rows (loss on their prompt tokens only)
  del      B (written once; identical for every rho)

  python arms_build.py --sub_pool data/sub_pool.jsonl --rhos 0.039 0.078 0.156 0.268 --out_dir arms
Use --perm_seed / --fill_seed offsets to draw a second derangement and a second padding set.
"""
import os, json, random, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--sub_pool", required=True)
ap.add_argument("--rhos", type=float, nargs="+", default=[0.039, 0.078, 0.156, 0.268])
ap.add_argument("--out_dir", default="arms")
ap.add_argument("--perm_seed", type=int, default=7000); ap.add_argument("--fill_seed", type=int, default=100)
ap.add_argument("--tag_suffix", default="")
a = ap.parse_args()
D = os.environ.get("DATA_DIR", "data"); os.makedirs(a.out_dir, exist_ok=True)

bg = [json.loads(l) for l in open(os.path.join(D, "background.jsonl"))]
pool = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(D, "trait_pool_full.jsonl")))}
sub = {r["id"]: r["completion"] for r in (json.loads(l) for l in open(a.sub_pool))}
order = sorted(sub); random.Random(0).shuffle(order)
B = len(bg)
print(f"|B|={B} substitute pool={len(order)} max rho={len(order)/B:.4f}")

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
cfg = []
for rho in a.rhos:
    nP = int(round(rho * B)); tag = f"r{rho:.3f}{a.tag_suffix}"
    assert nP <= len(order), (rho, nP, len(order))
    sel = order[:nP]
    keep = [{k: pool[i][k] for k in ("id", "source", "prompt", "completion")} for i in sel]
    write(f"keep_{tag}.jsonl", keep)
    write(f"sub_{tag}.jsonl", [{**x, "completion": sub[x["id"]], "id": x["id"] + "__sub"} for x in keep])
    perm = derange(nP, a.perm_seed + int(rho * 1000)); comps = [x["completion"] for x in keep]
    write(f"perm_{tag}.jsonl", [{**x, "completion": comps[perm[k]], "id": x["id"] + "__perm"} for k, x in enumerate(keep)])
    fill = random.Random(a.fill_seed + int(rho * 1000)).sample(range(B), nP)
    write(f"bud_{tag}.jsonl", [dict(bg[i]) for i in fill])
    write(f"ponly_{tag}.jsonl", [{**x, "mask_response": True} for x in keep])
    cfg.append({"rho": rho, "nP": nP, "rows": B + nP})
json.dump({"B": B, "pool": len(order), "configs": cfg}, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
