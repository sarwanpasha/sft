"""Download the external benchmark prompts once (network needed) and cache the lm-eval datasets.
Writes BENCH_DIR/{xstest,orbench_hard,orbench_toxic,strongreject}.jsonl."""
import json, os, csv, io, urllib.request
from datasets import load_dataset
OD = os.environ.get("BENCH_DIR", "bench_data"); os.makedirs(OD, exist_ok=True)
def dump(name, rows):
    with open(f"{OD}/{name}.jsonl", "w") as f:
        for i, r in enumerate(rows): f.write(json.dumps({"id": f"{name}_{i}", **r}) + "\n")
    print(f"{name}: {len(rows)}", flush=True)
def fetch_csv(url):
    return list(csv.DictReader(io.StringIO(urllib.request.urlopen(url, timeout=60).read().decode("utf-8"))))
xs = fetch_csv("https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv")
dump("xstest", [{"prompt": r["prompt"], "label": r.get("label") or ("unsafe" if r.get("type", "").startswith("contrast") else "safe"),
                 "type": r.get("type", ""), "bench": "xstest"} for r in xs])
for cfg, nm in (("or-bench-hard-1k", "orbench_hard"), ("or-bench-toxic", "orbench_toxic")):
    d = load_dataset("bench-llm/or-bench", cfg, split="train")
    dump(nm, [{"prompt": r["prompt"], "category": r.get("category", ""), "bench": nm} for r in d])
sr = fetch_csv("https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv")
dump("strongreject", [{"prompt": r.get("forbidden_prompt") or r.get("prompt"), "category": r.get("category", ""), "bench": "strongreject"} for r in sr])
load_dataset("allenai/ai2_arc", "ARC-Easy"); load_dataset("allenai/ai2_arc", "ARC-Challenge"); load_dataset("Rowan/hellaswag")
print("lm-eval datasets cached")
