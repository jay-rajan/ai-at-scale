"""Step 4: fuzzy-score every candidate pair, then sort into match / no match / unsure.

Score = weighted Jaro-Winkler on names and place + edit distance on postcode.
Thresholds are set by hand from the score bands below (a real team tunes them on labelled pairs).
"""
import argparse
import csv
import pickle
import time
from collections import Counter

import er

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
ap.add_argument("--match", type=float, default=0.93, help="score >= this: auto match")
ap.add_argument("--nonmatch", type=float, default=0.80, help="score < this: auto no match")
args = ap.parse_args()

rows = {r["id"]: r for r in er.read_csv(er.DATA / f"normalized_{args.records}.csv")}
cands = pickle.load(open(er.DATA / f"candidates_{args.records}.pkl", "rb"))
truth = er.true_pairs(rows.values())

t0 = time.perf_counter()
scored = []
for a, b in cands:
    f = er.compare(rows[a], rows[b])
    scored.append((a, b, er.score(f), f))
t = time.perf_counter() - t0
print(f"scored {len(scored):,} pairs in {t:.2f}s ({t / max(len(scored), 1) * 1e6:.1f} µs/pair)\n")

bands = Counter()
for a, b, s, _ in scored:
    bands[(min(int(s * 20) / 20, 0.95), (a, b) in truth)] += 1
print(f"{'score band':>11} {'same person':>12} {'different':>10}")
for lo in sorted({k[0] for k in bands}):
    print(f"{lo:>6.2f}+     {bands[(lo, True)]:>12,} {bands[(lo, False)]:>10,}")

match = {(a, b) for a, b, s, _ in scored if s >= args.match}
grey = [(a, b, s, f) for a, b, s, f in scored if args.nonmatch <= s < args.match]
nonmatch = len(scored) - len(match) - len(grey)
grey_true = sum((a, b) in truth for a, b, *_ in grey)
nonmatch_true = sum((a, b) in truth for a, b, s, _ in scored if s < args.nonmatch)
p, r, f1 = er.prf(match, truth)

print(f"\nauto match   (>= {args.match}): {len(match):>7,}   wrong merges: {len(match - truth):,}")
print(f"unsure       ({args.nonmatch}-{args.match}): {len(grey):>7,}   of which same person: {grey_true:,}")
print(f"auto no match (< {args.nonmatch}): {nonmatch:>7,}   missed same person: {nonmatch_true:,}")
print(f"\ncode only (unsure = no):   precision {p:.1%}  recall {r:.1%}  F1 {f1:.3f}")
p2, r2, f2 = er.prf(match | {(a, b) for a, b, *_ in grey}, truth)
print(f"code only (unsure = yes):  precision {p2:.1%}  recall {r2:.1%}  F1 {f2:.3f}")

pickle.dump({"match": sorted(match), "grey": [(a, b, s) for a, b, s, _ in grey]},
            open(er.DATA / f"scored_{args.records}.pkl", "wb"))
with open(er.DATA / f"grey_{args.records}.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["a", "b", "score", "a_name", "b_name", "a_place", "b_place", "a_postcode", "b_postcode", "same_person"])
    for a, b, s, _ in grey:
        A, B = rows[a], rows[b]
        w.writerow([a, b, f"{s:.3f}", f"{A['givenname']} {A['surname']}", f"{B['givenname']} {B['surname']}",
                    A["suburb"], B["suburb"], A["postcode"], B["postcode"], (a, b) in truth])
print(f"-> scored_{args.records}.pkl, grey_{args.records}.csv")
