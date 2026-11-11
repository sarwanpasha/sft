"""Substitute pool from a second (different-family) teacher, restricted to the ids of the first
substitute pool. Applies pool_filter.verdict, then judge.py with the Qwen judge, and writes a
substitute pool usable by arms_build.py. Because the second teacher complies on only part of the
pool, arms_build.py should be run with this pool to obtain the matched keep arm on the same rows.

  python pool_second_teacher.py --shards data/teacher/Llama-3.1-8B-Instruct_shard*.jsonl --out data/sub_pool_llama.jsonl
"""
import os, sys, json, glob, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from pool_filter import verdict
ap = argparse.ArgumentParser(); ap.add_argument("--shards", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--judge_model", default="Qwen/Qwen2.5-3B-Instruct"); a = ap.parse_args()
rows = []
for p in a.shards:
    for g in glob.glob(p): rows += [json.loads(l) for l in open(g)]
clean = [r for r in rows if verdict(r["completion"]) is None]
print(f"teacher rows {len(rows)} lexically clean {len(clean)} ({len(clean)/len(rows):.3f})")
tmp = a.out + ".clean.jsonl"
with open(tmp, "w") as f:
    for r in clean: f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": r["completion"]}) + "\n")
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge.py"),
                "--inp", tmp, "--out", tmp + ".judge", "--model", a.judge_model, "--bs", "32"], check=True)
dec = {json.loads(l)["id"]: json.loads(l)["decline"] for l in open(tmp + ".judge")}
ok = [r for r in clean if not dec[r["id"]]]
with open(a.out, "w") as f:
    for r in ok: f.write(json.dumps({"id": r["id"], "completion": r["completion"]}) + "\n")
print(f"judge-compliant {len(ok)} of {len(clean)} -> {a.out}")
