"""Score a judge against blinded human labels. The human labels are item numbers (1..n) of a sample
file whose items map to pool ids; the judge is run on the same items with judge.py and the confusion
matrix, agreement, Cohen's kappa, precision on declines, and false-omission rate are reported.

  python judge_calibrate.py --items calib_items.jsonl --human human_decline_items.json --model meta-llama/Llama-3.1-8B-Instruct
items: jsonl with id (1..n as strings), prompt, response.  human: JSON list of item numbers labeled DECLINE.
"""
import os, sys, json, argparse, subprocess
ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--human", required=True)
ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct"); ap.add_argument("--out", default="calib_result.json"); a = ap.parse_args()
HUM = set(json.load(open(a.human)))
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge.py"), "--inp", a.items, "--out", a.items + ".judge", "--model", a.model, "--bs", "8"], check=True)
J = {json.loads(l)["id"]: json.loads(l)["decline"] for l in open(a.items + ".judge")}
n = len(J); tp = fp = fn = tn = 0
for k, jd in J.items():
    hd = int(k) in HUM
    tp += jd and hd; fp += jd and not hd; fn += (not jd) and hd; tn += (not jd) and (not hd)
po = (tp + tn) / n; pj = (tp + fp) / n; ph = (tp + fn) / n; pe = pj * ph + (1 - pj) * (1 - ph)
out = {"judge": a.model, "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "agreement": po, "kappa": (po - pe) / (1 - pe),
       "precision": tp / max(1, tp + fp), "false_omission": fn / max(1, fn + tn)}
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))
