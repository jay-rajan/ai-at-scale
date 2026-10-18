"""Step 6: "don't merge if in doubt". No new LLM calls: re-reads saved answers.

The LLM merges only when it says "same" with confidence >= a cut-off; everything else goes to a steward.
"""
import glob
import json

for path in sorted(glob.glob("results/llm_*.json")):
    if "_lim" in path:
        continue
    pairs = json.load(open(path))["pairs"]
    true_dups = sum(p["truth"] for p in pairs)
    print(f"\n{path.split('llm_')[1][:-5]}  ({len(pairs):,} unsure pairs, {true_dups} real duplicates)")
    print(f"  {'merge only if':>16} {'merged':>7} {'wrong':>6} {'real dups merged':>17} {'to steward':>11}")
    for cut in (None, 0.8, 0.9, 0.95):
        sure = [p for p in pairs if p["same"] and (cut is None or (p["confidence"] or 0) >= cut)]
        doubt = sum(1 for p in pairs if p["same"] and cut is not None and (p["confidence"] or 0) < cut)
        label = "always" if cut is None else f"conf >= {cut}"
        print(f"  {label:>16} {len(sure):>7} {sum(not p['truth'] for p in sure):>6} "
              f"{sum(p['truth'] for p in sure):>11} of {true_dups:<3} {doubt:>11}")
