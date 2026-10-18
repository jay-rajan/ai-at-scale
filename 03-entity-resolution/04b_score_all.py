"""Step 4b: score every bucketed pair of the full 5M with code rules, streaming (pairs are never stored).

4 keys, buckets over 200 skipped (~247M pair slots). A pair found by several keys is scored once
(by the first key that finds it). One process per key.
Rules: weighted Jaro-Winkler score; >= 0.93 merge, < 0.80 no; plus a guard: first names < 0.85 -> unsure.
"""
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
import multiprocessing as mp

from rapidfuzz.distance import JaroWinkler, Levenshtein

import er

MAX_BLOCK = 200
KEYS = list(er.BLOCKING_PASSES.items())
G, S, T, P, SRC, RID = [], [], [], [], [], []
KEYV = [[] for _ in KEYS]


def load():
    known = er.build_place_dictionary(er.load_raw())
    by_letter = defaultdict(list)
    for p in sorted(known):
        by_letter[p[0]].append(p)
    cache = {}
    for path in sorted(er.RAW.glob("*.csv")):
        src = int(re.search(r"myp_(\d)", path.name).group(1))
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                place = er.clean_place(r["suburb"])
                if place not in cache:
                    cache[place] = er.fix_place(place, known, by_letter)[0]
                n = {"n_givenname": er.clean_name(r["givenname"]), "n_surname": er.clean_name(r["surname"]),
                     "n_suburb": cache[place], "n_postcode": er.clean_postcode(r["postcode"])}
                G.append(n["n_givenname"]); S.append(n["n_surname"]); T.append(n["n_suburb"]); P.append(n["n_postcode"])
                SRC.append(src); RID.append(r["recid"])
                for j, (_, fn) in enumerate(KEYS):
                    KEYV[j].append(fn(n))


def work(k):
    sim = JaroWinkler.similarity
    lev = Levenshtein.distance
    keys_k = KEYV[k]
    sizes = [Counter(v for v in kv if v) for kv in KEYV]
    buckets = defaultdict(list)
    for i, v in enumerate(keys_k):
        if v and sizes[k][v] <= MAX_BLOCK:
            buckets[v].append(i)
    earlier = range(k)
    out = Counter()
    for ids in buckets.values():
        for x in range(len(ids)):
            a = ids[x]
            for y in range(x + 1, len(ids)):
                b = ids[y]
                if SRC[a] == SRC[b]:
                    continue
                dup = False
                for j in earlier:  # already scored by an earlier key?
                    v = KEYV[j][a]
                    if v and v == KEYV[j][b] and sizes[j][v] <= MAX_BLOCK:
                        dup = True
                        break
                if dup:
                    continue
                g = sim(G[a], G[b]) if G[a] and G[b] else None
                tot = wt = 0.0
                if g is not None:
                    tot += 0.30 * g; wt += 0.30
                if S[a] and S[b]:
                    tot += 0.35 * sim(S[a], S[b]); wt += 0.35
                if T[a] and T[b]:
                    tot += 0.15 * sim(T[a], T[b]); wt += 0.15
                if P[a] and P[b]:
                    tot += 0.20 * (1 - lev(P[a], P[b]) / 5); wt += 0.20
                s = tot / wt if wt else 0.0
                base = "merge" if s >= 0.93 else ("unsure" if s >= 0.80 else "no")
                guard = "unsure" if base == "merge" and (g is None or g < 0.85) else base
                same = RID[a] == RID[b]
                out[("score", base, same)] += 1
                out[("guard", guard, same)] += 1
    return k, out


if __name__ == "__main__":
    t0 = time.perf_counter()
    load()
    print(f"loaded + keys for {len(RID):,} records in {time.perf_counter() - t0:.0f}s", flush=True)
    t1 = time.perf_counter()
    total = Counter()
    with mp.get_context("fork").Pool(len(KEYS)) as pool:  # fork: workers share the loaded data
        for k, out in pool.imap_unordered(work, range(len(KEYS))):
            total.update(out)
            print(f"  key {KEYS[k][0]} done at {time.perf_counter() - t1:.0f}s: "
                  f"{sum(v for (m, _, _), v in out.items() if m == 'score'):,} new pairs", flush=True)
    t_score = time.perf_counter() - t1
    by_person = Counter(RID)
    true_pairs = sum(c * (c - 1) // 2 for c in by_person.values())
    pairs = sum(v for (m, _, _), v in total.items() if m == "score")
    res = {"records": len(RID), "distinct_pairs_scored": pairs, "true_pairs": true_pairs,
           "score_seconds": round(t_score), "us_per_pair": round(t_score / pairs * 1e6 * len(KEYS), 2)}
    for mode in ("score", "guard"):
        res[mode] = {f"{d}_{'same' if same else 'diff'}": total[(mode, d, same)]
                     for d in ("merge", "unsure", "no") for same in (True, False)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(er.RESULTS / "score_all_5M.json", "w"), indent=1)
