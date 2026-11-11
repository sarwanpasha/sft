"""External-benchmark generation for one checkpoint: greedy, 256 new tokens, on XSTest, OR-Bench
hard-1k, OR-Bench toxic, and StrongREJECT. Writes BENCH_OUT/<tag>_gen.jsonl with lexical flags; run
judge.py on it afterwards for the semantic verdicts, and lm-eval for the capability check
(see bench_step.sh)."""
import json, os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sft import REFUSAL_RE, HARM_REFUSAL_RE, ensure_chat_template
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--max_new", type=int, default=256); a = ap.parse_args()
BD = os.environ.get("BENCH_DIR", "bench_data"); OD = os.environ.get("BENCH_OUT", "bench_out"); os.makedirs(OD, exist_ok=True)
rows = []
for f in ("xstest", "orbench_hard", "orbench_toxic", "strongreject"):
    rows += [json.loads(l) for l in open(f"{BD}/{f}.jsonl")]
tok = AutoTokenizer.from_pretrained(a.ckpt); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
ensure_chat_template(tok)
model = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.bfloat16).cuda().eval()
with open(f"{OD}/{a.tag}_gen.jsonl", "w") as f:
    for i in range(0, len(rows), a.bs):
        ch = rows[i:i+a.bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in ch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, r in enumerate(ch):
            gen = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"id": r["id"], "bench": r["bench"], "label": r.get("label", ""), "prompt": r["prompt"],
                                "response": gen[:1500], "refuses": bool(REFUSAL_RE.search(gen)), "strict": bool(HARM_REFUSAL_RE.search(gen))}) + "\n")
print(f"BENCH GEN {a.tag}: {len(rows)} prompts")
