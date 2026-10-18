"""Step 3b: bucket with MinHash LSH instead of hand-picked keys.

Each record -> set of q-letter pieces (2-grams by default) of "given surname town".
MinHash signature of k = bands x rows numbers; records whose rows match in any band share a bucket.
Chance two records become a candidate: 1 - (1 - s^rows)^bands, where s = Jaccard similarity of their letter pairs.
"""
import argparse
import time
import zlib
from collections import defaultdict

import numpy as np

import er

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
ap.add_argument("--bands", type=int, default=20)
ap.add_argument("--rows", type=int, default=5)
ap.add_argument("--fields", default="name", choices=["name", "name+town"])
ap.add_argument("--q", type=int, default=2, help="letters per shingle")
args = ap.parse_args()

rows = er.read_csv(er.DATA / f"normalized_{args.records}.csv")
src = {r["id"]: r["source"] for r in rows}
truth = er.true_pairs(rows)
k = args.bands * args.rows
rng = np.random.default_rng(1)
P = (1 << 61) - 1
A = rng.integers(1, P, size=k, dtype=np.uint64)
B = rng.integers(0, P, size=k, dtype=np.uint64)


def shingles(r):
    s = f"{r['n_givenname']} {r['n_surname']}"
    if args.fields == "name+town":
        s += f" {r['n_suburb']}"
    return np.array(sorted({zlib.crc32(s[i:i + args.q].encode()) for i in range(len(s) - args.q + 1)}), dtype=np.uint64)


t0 = time.perf_counter()
sigs = np.empty((len(rows), k), dtype=np.uint64)
for i, r in enumerate(rows):
    x = shingles(r)
    sigs[i] = ((A[:, None] * x[None, :] + B[:, None]) % P).min(axis=1)  # uint64 wraparound is fine as a hash
t_sig = time.perf_counter() - t0

t0 = time.perf_counter()
pairs, n_buckets, biggest = set(), 0, 0
for band in range(args.bands):
    buckets = defaultdict(list)
    cols = sigs[:, band * args.rows:(band + 1) * args.rows]
    for i, r in enumerate(rows):
        buckets[cols[i].tobytes()].append(r["id"])
    n_buckets += len(buckets)
    for ids in buckets.values():
        biggest = max(biggest, len(ids))
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = (ids[x], ids[y]) if ids[x] < ids[y] else (ids[y], ids[x])
                if src[a] != src[b]:
                    pairs.add((a, b))
t_band = time.perf_counter() - t0

n = len(rows)
found = len(pairs & truth)
thr = (1 / args.bands) ** (1 / args.rows)
print(f"MinHash LSH  {args.bands} bands x {args.rows} rows  on {args.fields}, {args.q}-grams   (similarity threshold ~{thr:.2f})")
print(f"  records {n:,}   possible pairs {n * (n - 1) // 2:,}")
print(f"  buckets (all bands) {n_buckets:,}   biggest bucket {biggest:,}")
print(f"  candidate pairs {len(pairs):,}   finds true pairs {found / len(truth):.1%}   missed {len(truth) - found:,}")
print(f"  time: signatures {t_sig:.1f}s, banding {t_band:.1f}s")
