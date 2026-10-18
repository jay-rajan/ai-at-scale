"""Step 5: send only the unsure pairs to an LLM.

Usage: python 05_llm.py --records 10000 --model google/gemini-3.5-flash-lite --per-call 10
       add --limit 100 to try a few first; --run 2 to repeat and check consistency.
Needs OPENROUTER_API_KEY in .env next to this file. Results are cached in results/.
"""
import argparse
import csv
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import er

URL = "https://openrouter.ai/api/v1/chat/completions"


def load_env(path=Path(__file__).with_name(".env")):
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and not k.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
ap.add_argument("--model", default="google/gemini-3.5-flash-lite")
ap.add_argument("--per-call", type=int, default=10)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sample", type=int, default=0, help="random sample of unsure pairs (seeded)")
ap.add_argument("--run", type=int, default=1)
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()
KEY = os.environ.get("OPENROUTER_API_KEY") or exit("Add OPENROUTER_API_KEY to .env first")

PROMPT = """You are matching customer records from different company systems.
Each pair below is two records. Decide if they are the SAME real person.
The data has typos (swapped or wrong letters, digits inside words) and some postcodes are wrong.
Different people can share a surname and a town (family members, common names).

Reply with JSON only: {{"answers": [{{"pair": <number>, "same": true|false, "confidence": <0-1>}}, ...]}}

{pairs}"""

grey = list(csv.DictReader(open(er.DATA / f"grey_{args.records}.csv")))
if args.sample:
    import random
    grey = random.Random(11).sample(grey, min(args.sample, len(grey)))
if args.limit:
    grey = grey[: args.limit]
batches = [grey[i: i + args.per_call] for i in range(0, len(grey), args.per_call)]


def ask(batch):
    lines = "\n".join(
        f"Pair {i + 1}: A = {p['a_name']}, {p['a_place']}, {p['a_postcode']} | "
        f"B = {p['b_name']}, {p['b_place']}, {p['b_postcode']}"
        for i, p in enumerate(batch))
    body = {"model": args.model, "temperature": 0, "usage": {"include": True},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": PROMPT.format(pairs=lines)}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            break
        except Exception as e:  # retry on network / rate limit errors
            if attempt == 2:
                return {"error": str(e), "batch": batch}
            time.sleep(2 ** attempt)
    secs = time.perf_counter() - t0
    text = resp["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        answers = {a["pair"]: a for a in json.loads(m.group(0))["answers"]}
    except Exception:
        answers = {}
    out = []
    for i, p in enumerate(batch):
        a = answers.get(i + 1, {})
        out.append({"a": p["a"], "b": p["b"], "truth": p["same_person"] == "True",
                    "same": a.get("same"), "confidence": a.get("confidence")})
    u = resp.get("usage", {})
    return {"pairs": out, "seconds": secs, "cost": u.get("cost", 0) or 0,
            "tokens_in": u.get("prompt_tokens", 0), "tokens_out": u.get("completion_tokens", 0)}


t0 = time.perf_counter()
with ThreadPoolExecutor(args.workers) as ex:
    results = list(ex.map(ask, batches))
wall = time.perf_counter() - t0

pairs = [p for r in results if "pairs" in r for p in r["pairs"]]
errors = sum("error" in r for r in results)
answered = [p for p in pairs if p["same"] is not None]
right = sum(p["same"] == p["truth"] for p in answered)
fp = sum(p["same"] and not p["truth"] for p in answered)
fn = sum((not p["same"]) and p["truth"] for p in answered)
cost = sum(r.get("cost", 0) for r in results)
tin = sum(r.get("tokens_in", 0) for r in results)
tout = sum(r.get("tokens_out", 0) for r in results)
lat = sorted(r["seconds"] for r in results if "seconds" in r)

summary = {
    "model": args.model, "per_call": args.per_call, "run": args.run, "pairs": len(grey),
    "calls": len(batches), "errors": errors, "answered": len(answered),
    "correct": right, "wrong_merges": fp, "missed": fn,
    "accuracy": right / len(answered) if answered else 0,
    "cost_usd": round(cost, 5), "cost_per_pair": cost / max(len(answered), 1),
    "tokens_in": tin, "tokens_out": tout,
    "median_call_s": lat[len(lat) // 2] if lat else None, "wall_s": round(wall, 1),
}
print(json.dumps(summary, indent=2))
tag = f"llm_{args.records}_{args.model.replace('/', '_')}_x{args.per_call}_run{args.run}" + (f"_lim{args.limit}" if args.limit else "") + (f"_s{args.sample}" if args.sample else "")
er.RESULTS.mkdir(exist_ok=True)
json.dump({"summary": summary, "pairs": pairs}, open(er.RESULTS / f"{tag}.json", "w"), indent=1)
print(f"-> results/{tag}.json")
