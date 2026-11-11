"""Selective arms at one rho (used at rho = 0.268): the targeted rows are split by source into an
adversarial-safety group (WildJailbreak, WildGuardMix) and the rest.

  target   substitute responses ONLY on the non-safety-source targeted rows; safety-source refusals kept
  global   substitute responses on every targeted row (same as sub_<tag>.jsonl from arms_build.py)
  seldel   DELETE only the non-safety-source targeted rows; safety-source refusals kept

  python arms_selective.py --keep arms/keep_r0.268.jsonl --sub arms/sub_r0.268.jsonl --out_dir arms
"""
import os, json, random, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--keep", required=True); ap.add_argument("--sub", required=True)
ap.add_argument("--out_dir", default="arms"); ap.add_argument("--tag", default="r0.268")
a = ap.parse_args()
D = os.environ.get("DATA_DIR", "data")
HARM = ("wildjailbreak", "wildguard")
is_harm = lambda s: any(k in s for k in HARM)
bgids = {json.loads(l)["id"] for l in open(os.path.join(D, "background.jsonl"))}
keep = [json.loads(l) for l in open(a.keep)]
subm = {r["id"].replace("__sub", ""): r["completion"] for r in (json.loads(l) for l in open(a.sub))}
tgt = [r for r in keep if r["id"] not in bgids]
ben = [r for r in tgt if not is_harm(r.get("source", ""))]
print(f"targeted {len(tgt)} safety-source {len(tgt)-len(ben)} other {len(ben)}")
def write(name, rows):
    rows = list(rows); random.Random(1).shuffle(rows)
    with open(os.path.join(a.out_dir, name), "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print(f"  wrote {len(rows):,} -> {name}")
write(f"target_{a.tag}.jsonl", [({**r, "completion": subm[r["id"]], "id": r["id"] + "__sub"} if (r["id"] not in bgids and not is_harm(r.get("source", ""))) else r) for r in keep])
write(f"seldel_{a.tag}.jsonl", [r for r in keep if not (r["id"] not in bgids and not is_harm(r.get("source", "")))])
