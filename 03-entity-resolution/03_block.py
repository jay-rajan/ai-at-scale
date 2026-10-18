"""Step 3: bucket (block). Only records that share a cheap key get compared.

Several passes, then the union: each pass misses some people, together they miss fewer.
Same-source pairs are dropped: each source system is already duplicate-free.
"""
import argparse
import pickle
import time

import er

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
ap.add_argument("--max-block", type=int, default=200)
args = ap.parse_args()
rows = er.read_csv(er.DATA / f"normalized_{args.records}.csv")
src = {r["id"]: r["source"] for r in rows}
truth = er.true_pairs(rows)
n = len(rows)
all_pairs = n * (n - 1) // 2

t0 = time.perf_counter()
per_pass = er.block(rows, max_block=args.max_block)
union = set()
print(f"{n:,} records -> {all_pairs:,} possible pairs, {len(truth):,} true pairs\n")
print(f"{'pass':32} {'pairs':>12} {'finds true pairs':>17} {'biggest block':>14} {'rows skipped':>13} {'time':>7}")
for name, p in per_pass.items():
    pairs = {(a, b) for a, b in p["pairs"] if src[a] != src[b]}
    union |= pairs
    found = len(pairs & truth) / len(truth)
    print(f"{name:32} {len(pairs):>12,} {found:>17.1%} {p['biggest']:>14,} {p['rows_in_skipped_blocks']:>13,} {p['seconds']:>6.2f}s")
t = time.perf_counter() - t0
found = len(union & truth)
print(f"{'UNION':32} {len(union):>12,} {found / len(truth):>17.1%} {'':>14} {'':>13} {t:>6.2f}s")
print(f"\ncomparisons cut by {1 - len(union) / all_pairs:.4%}  ({all_pairs / max(len(union), 1):,.0f}x fewer)")
print(f"true pairs no bucket caught: {len(truth) - found:,}")

out = er.DATA / f"candidates_{args.records}.pkl"
pickle.dump(sorted(union), open(out, "wb"))
print(f"-> {out.name}")
