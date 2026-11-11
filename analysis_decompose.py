"""Arm means, decomposition contrasts, Welch tests, bootstrap intervals for the selectivity ratio,
and the TOST equivalence test for the removal effect.

Reads the per-run summaries written by run_sft.py eval (OUT/<arm>_r<rho>_s<seed>_summary.json) and,
optionally, the per-item judge files (JUDGE_OUT/<tag>_judge.jsonl) for the judge-scored outcome.

  python analysis_decompose.py --out_dir out --metric refusal_rate
  python analysis_decompose.py --out_dir out --metric strict_refusal_rate
  python analysis_decompose.py --out_dir out --judge_dir judge_out      # semantic judge outcome
  python analysis_decompose.py --out_dir out --selectivity --eval data/eval_prompts.jsonl --items eval_out
"""
import os, re, json, glob, argparse, math
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default="out"); ap.add_argument("--metric", default="refusal_rate")
ap.add_argument("--judge_dir", default=None); ap.add_argument("--rhos", type=float, nargs="+", default=[0.039, 0.078, 0.156, 0.268])
ap.add_argument("--tost_margin", type=float, default=0.05, help="equivalence margin as a fraction of T_m")
ap.add_argument("--selectivity", action="store_true"); ap.add_argument("--eval", default=None); ap.add_argument("--items", default="eval_out")
ap.add_argument("--harm_sources", nargs="+", default=["wildjailbreak", "wildguard"])
a = ap.parse_args()
TAG = re.compile(r"^(?P<arm>[a-z0-9]+)_r(?P<rho>[0-9.]+)(?:_[a-z0-9]+)?_s(?P<seed>\d+)$")

def load_runs():
    runs = {}
    for f in glob.glob(os.path.join(a.out_dir, "*_summary.json")):
        s = json.load(open(f)); m = TAG.match(s["tag"])
        if not m: continue
        arm, rho, seed = m["arm"], float(m["rho"]), int(m["seed"])
        if a.judge_dir:
            jf = os.path.join(a.judge_dir, s["tag"] + "_judge.jsonl")
            if not os.path.exists(jf): continue
            val = float(np.mean([json.loads(l)["decline"] for l in open(jf)]))
        else:
            val = s[a.metric]
        runs.setdefault((arm, rho), {})[seed] = val
    return runs

def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x.mean() - y.mean(); se = math.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))
    df = se**4 / ((x.var(ddof=1)/len(x))**2/(len(x)-1) + (y.var(ddof=1)/len(y))**2/(len(y)-1))
    t = d/se; p = 2*stats.t.sf(abs(t), df); h = stats.t.ppf(0.975, df)*se
    return d, se, t, df, p, (d-h, d+h)

def share_ci(num, den):
    """Welch interval for a ratio of two independent mean differences via the delta method."""
    (dn, sn, *_), (dd, sd, *_) = num, den
    r = dn/dd; se = abs(r)*math.sqrt((sn/dn)**2 + (sd/dd)**2) if dn != 0 else sn/abs(dd)
    return r, (r-1.96*se, r+1.96*se)

def tost(x, y, margin):
    d, se, _, df, _, _ = welch(x, y)
    p1 = stats.t.sf((d + margin)/se, df); p2 = stats.t.sf((margin - d)/se, df)
    return max(p1, p2)

runs = load_runs()
dele = list(runs.get(("del", 0.0), {}).values()) or [v for (arm, _), s in runs.items() if arm == "del" for v in s.values()]
print(f"outcome: {'judge decline' if a.judge_dir else a.metric}   delete arm: n={len(dele)} mean={np.mean(dele):.5f} sd={np.std(dele, ddof=1):.5f}\n")
hdr = f"{'rho':>6} {'keep':>8} {'sub':>8} {'bud':>8} {'perm':>8} | {'T_m':>8} {'C/T_m':>7} {'R_m/T_m':>8} {'t_R':>6} {'dPi/T_m':>8} {'t_Pi':>6} {'TOST p':>7}"
print(hdr)
for rho in a.rhos:
    g = lambda arm: list(runs.get((arm, rho), {}).values())
    K, S, Bd, P = g("keep"), g("sub"), g("bud"), g("perm")
    if not (K and S and Bd): continue
    Tm = welch(K, Bd); C = welch(K, S); Rm = welch(S, Bd)
    c_share, _ = share_ci(C, Tm); r_share, r_ci = share_ci(Rm, Tm)
    pi = welch(P, K) if P else None
    pi_share = share_ci(pi, Tm)[0] if P else float("nan")
    p_tost = tost(S, Bd, a.tost_margin*Tm[0])
    print(f"{rho:6.3f} {np.mean(K):8.5f} {np.mean(S):8.5f} {np.mean(Bd):8.5f} {np.mean(P) if P else float('nan'):8.4f} | "
          f"{Tm[0]:8.5f} {100*c_share:6.1f}% {100*r_share:7.1f}% {Rm[2]:6.1f} {100*pi_share:7.1f}% {pi[2] if P else float('nan'):6.1f} {p_tost:7.3f}")
    po = g("ponly")
    if po:
        Rp = welch(po, Bd); s, ci = share_ci(Rp, Tm)
        print(f"       prompt-only: R_p/T_m = {100*s:.1f}%  t={Rp[2]:.1f}  CI [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]")

if a.selectivity:
    ev = {json.loads(l)["id"]: json.loads(l)["source"] for l in open(a.eval)}
    harm = {i for i, s in ev.items() if any(h in s for h in a.harm_sources)}
    def split_rate(tag):
        f = os.path.join(a.items, tag + ".jsonl")
        if not os.path.exists(f): return None
        b = h = nb = nh = 0
        for l in open(f):
            r = json.loads(l)
            if r["id"] in harm: nh += 1; h += r["refuses"]
            else: nb += 1; b += r["refuses"]
        return b/nb, h/nh
    rates = {}
    for f in glob.glob(os.path.join(a.out_dir, "*_summary.json")):
        t = json.load(open(f))["tag"]; m = TAG.match(t)
        if m and (sr := split_rate(t)): rates.setdefault(m["arm"], []).append(sr)
    kb = np.array([x[0] for x in rates["keep"]]); kh = np.array([x[1] for x in rates["keep"]])
    rng = np.random.default_rng(0)
    print(f"\nselectivity (keep-relative reductions, benign vs harmful-source prompts); keep benign {kb.mean():.4f} harm {kh.mean():.4f}")
    for arm, v in rates.items():
        if arm == "keep": continue
        ab = np.array([x[0] for x in v]); ah = np.array([x[1] for x in v])
        Do = 1 - ab.mean()/kb.mean(); Dw = 1 - ah.mean()/kh.mean()
        boots = []
        for _ in range(2000):
            kbi = rng.choice(kb, len(kb)); khi = rng.choice(kh, len(kh)); abi = rng.choice(ab, len(ab)); ahi = rng.choice(ah, len(ah))
            boots.append((1 - abi.mean()/kbi.mean())/(1 - ahi.mean()/khi.mean()))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {arm:8s} n={len(v):2d}  D_over={Do:.3f} D_warr={Dw:.3f}  S={Do/Dw:.2f} [{lo:.2f}, {hi:.2f}]")
