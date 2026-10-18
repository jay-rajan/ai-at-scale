"""Step 7: the unsure pairs through Jev, a decision model (typed answer + probabilities).

Choices: same / different / unsure. Merge only on "same"; "unsure" goes to a steward.
Usage: python 07_jev.py --records 10000 [--limit 5]
"""
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import er

URL = "https://openrouter.ai/api/alpha/decisions"
for line in (Path(__file__).with_name(".env").read_text().splitlines() if Path(__file__).with_name(".env").exists() else []):
    k, sep, v = line.partition("=")
    if sep and not k.lstrip().startswith("#"):
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
KEY = os.environ.get("OPENROUTER_API_KEY") or exit("Add OPENROUTER_API_KEY to .env first")

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
ap.add_argument("--model", default="typesafe/jev-1.13")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=12)
args = ap.parse_args()

QUESTION = {
    "same_person": {
        "type": "choice",
        "instructions": ("Two customer records from different company systems. Are they the same real person? "
                         "The data has typos (wrong or swapped letters, digits inside words) and some postcodes are wrong. "
                         "Different people can share a surname and town (family members, common names)."),
        "criteria": {
            "same": "The same real person; the differences are typos or a short form of the name.",
            "different": "Different people, e.g. a family member or someone with a similar name.",
            "unsure": "The fields cannot tell whether this is one person or two.",
        },
    }
}

grey = list(csv.DictReader(open(er.DATA / f"grey_{args.records}.csv")))
if args.limit:
    grey = grey[: args.limit]


def ask(p):
    state = {"A": {"name": p["a_name"], "town": p["a_place"], "postcode": p["a_postcode"]},
             "B": {"name": p["b_name"], "town": p["b_place"], "postcode": p["b_postcode"]}}
    body = {"model": args.model, "state": state, "questions": QUESTION}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    for attempt in range(4):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=60))
            break
        except urllib.error.HTTPError as e:
            err = f"{e.code} {e.read()[:200]!r}"
        except Exception as e:
            err = str(e)
        if attempt == 3:
            return {"a": p["a"], "b": p["b"], "truth": p["same_person"] == "True", "error": err}
        time.sleep(2 ** attempt)
    ans = resp["answers"]["same_person"]
    u = resp.get("usage", {})
    return {"a": p["a"], "b": p["b"], "truth": p["same_person"] == "True",
            "choice": ans.get("choice"), "confidence": ans.get("confidence"),
            "probabilities": ans.get("probabilities"), "seconds": time.perf_counter() - t0,
            "cost": u.get("cost", 0) or 0, "tokens_in": u.get("input_tokens", 0), "model_version": resp.get("model")}


# resumable: answers are appended to a .jsonl as they arrive; a rerun skips pairs already answered
cache = er.RESULTS / (f"jev_{args.records}" + (f"_lim{args.limit}" if args.limit else "") + ".jsonl")
done = {}
if cache.exists():
    for line in cache.read_text().splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:  # a line cut off by a stopped run
            continue
        if "choice" in o:
            done[(o["a"], o["b"])] = o
todo = [p for p in grey if (p["a"], p["b"]) not in done]
print(f"{len(grey):,} pairs, {len(done):,} already answered, {len(todo):,} to ask", flush=True)
t0 = time.perf_counter()
with ThreadPoolExecutor(args.workers) as ex, open(cache, "a") as fh:
    for i, o in enumerate(ex.map(ask, todo), 1):
        fh.write(json.dumps(o) + "\n")
        if i % 5000 == 0:
            fh.flush()
            print(f"  {i:,} / {len(todo):,}  {time.perf_counter() - t0:.0f}s", flush=True)
wall = time.perf_counter() - t0
fresh = {}
for line in cache.read_text().splitlines():
    try:
        o = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "choice" in o or (o["a"], o["b"]) not in fresh:
        fresh[(o["a"], o["b"])] = o
out = [fresh[(p["a"], p["b"])] for p in grey]

ok = [o for o in out if "choice" in o]
errors = [o for o in out if "error" in o]
true_dups = sum(o["truth"] for o in ok)
merged = [o for o in ok if o["choice"] == "same"]
unsure = [o for o in ok if o["choice"] == "unsure"]
lat = sorted(o["seconds"] for o in ok)
summary = {
    "model": args.model, "version": ok[0]["model_version"] if ok else None, "pairs": len(grey),
    "errors": len(errors), "first_error": errors[0]["error"] if errors else None,
    "choices": {c: sum(o["choice"] == c for o in ok) for c in ("same", "different", "unsure")},
    "merged": len(merged), "wrong_merges": sum(not o["truth"] for o in merged),
    "real_dups_merged": sum(o["truth"] for o in merged), "real_dups": true_dups,
    "real_dups_in_unsure": sum(o["truth"] for o in unsure),
    "real_dups_called_different": sum(o["truth"] and o["choice"] == "different" for o in ok),
    "cost_usd": round(sum(o["cost"] for o in ok), 5),
    "median_call_s": round(lat[len(lat) // 2], 2) if lat else None, "wall_s_this_run": round(wall, 1),
    "workers": args.workers,
}
print(json.dumps(summary, indent=2))
tag = f"jev_{args.records}" + (f"_lim{args.limit}" if args.limit else "")
json.dump({"summary": summary, "pairs": out}, open(er.RESULTS / f"{tag}.json", "w"), indent=1)
print(f"-> results/{tag}.json")
