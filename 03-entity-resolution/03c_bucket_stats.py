"""Bucket statistics for the 3 keys, at any size up to the full 5M, without storing pairs.

Pairs per bucket are counted (cross-system only), not listed, so 5M fits in memory.
Usage: python 03c_bucket_stats.py --records 10000 | 100000 | all
"""
import argparse
import csv
import re
import time
from collections import Counter, defaultdict

import er

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
ap.add_argument("--max-block", type=int, default=200)
ap.add_argument("--all-keys", action="store_true", help="also use first-name sound + surname sound")
ap.add_argument("--split", action="store_true", help="split big buckets with a finer key instead of skipping them")
args = ap.parse_args()

KEYS = {k: v for k, v in er.BLOCKING_PASSES.items() if not k.startswith("sound(given)")}
NAMES = {"sound(given)+sound(surname)": "first name sound + surname sound", "sound(surname)+first initial": "surname sound + first initial",
         "surname[:3]+postcode": "surname + postcode", "given[:3]+place": "first name + town"}


# finer key used only inside a bucket that is too big
REFINE = {
    "sound(given)+sound(surname)": lambda n: n["n_suburb"],                         # + town
    "sound(surname)+first initial": lambda n: n["n_suburb"],                       # + town
    "surname[:3]+postcode": lambda n: n["n_givenname"][:1],                       # + first initial
    "given[:3]+place": lambda n: er.metaphone(n["n_surname"])[:2] if n["n_surname"] else "",  # + surname sound start
}


if args.all_keys:
    KEYS = dict(er.BLOCKING_PASSES)


def records():
    if args.records == "all":
        for path in sorted(er.RAW.glob("*.csv")):
            src = int(re.search(r"myp_(\d)", path.name).group(1))
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    r["source"] = src
                    yield r
    else:
        yield from er.read_csv(er.DATA / f"sample_{args.records}.csv")


t0 = time.perf_counter()
known = er.build_place_dictionary(er.load_raw())
by_letter = defaultdict(list)
for p in sorted(known):  # sorted: same result every run
    by_letter[p[0]].append(p)
place_cache = {}

keys = {k: [] for k in KEYS}      # key value per record, in order
fine = {k: [] for k in KEYS}
recid, source = [], []
for r in records():
    place = er.clean_place(r["suburb"])
    if place not in place_cache:
        place_cache[place] = er.fix_place(place, known, by_letter)[0]
    n = {"n_givenname": er.clean_name(r["givenname"]), "n_surname": er.clean_name(r["surname"]),
         "n_suburb": place_cache[place], "n_postcode": er.clean_postcode(r["postcode"])}
    for k, fn in KEYS.items():
        keys[k].append(fn(n))
        if args.split:
            fine[k].append(REFINE[k](n))
    recid.append(r["recid"])
    source.append(r["source"])
n = len(recid)
if args.split:  # records in a too-big bucket get the finer key appended
    for k in KEYS:
        size = Counter(v for v in keys[k] if v)
        keys[k] = [(v + "|" + fine[k][i] if fine[k][i] else "") if v and size[v] > args.max_block else v
                   for i, v in enumerate(keys[k])]
t_keys = time.perf_counter() - t0

# bucket sizes per key, and cross-system pairs per bucket
stats = {}
size_of = {}
for k in KEYS:
    members = defaultdict(Counter)       # key -> records per source
    for i, v in enumerate(keys[k]):
        if v:
            members[v][source[i]] += 1
    sizes = {v: sum(c.values()) for v, c in members.items()}
    pairs = big_pairs = skipped = 0
    for v, c in members.items():
        cross = sizes[v] * (sizes[v] - 1) // 2 - sum(m * (m - 1) // 2 for m in c.values())
        if sizes[v] > args.max_block:
            big_pairs += cross
            skipped += 1
        else:
            pairs += cross
    stats[k] = (sizes, pairs, big_pairs, skipped)
    size_of[k] = sizes

# true pairs (same recid): which keys put them in a bucket we actually use (not skipped)
by_person = defaultdict(list)
for i, p in enumerate(recid):
    by_person[p].append(i)
true_total = 0
caught = Counter()
for ids in by_person.values():
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = ids[x], ids[y]
            true_total += 1
            hit = False
            for k in KEYS:
                v = keys[k][a]
                if v and v == keys[k][b] and size_of[k][v] <= args.max_block:
                    caught[k] += 1
                    hit = True
            caught["any"] += hit

print(f"{n:,} records   {n * (n - 1) // 2:,} possible pairs   {true_total:,} true duplicate pairs   keys made in {t_keys:.1f}s\n")
print(f"{'key':30} {'buckets':>9} {'with 2+':>9} {'avg':>5} {'biggest':>7}  {'biggest bucket':15} {'pairs':>13} {'skipped >' + str(args.max_block):>22} {'dups caught':>11}")
total_pairs = 0
for k in KEYS:
    sizes, pairs, big_pairs, skipped = stats[k]
    total_pairs += pairs
    top = max(sizes, key=sizes.get)
    print(f"{NAMES[k]:30} {len(sizes):>9,} {sum(1 for s in sizes.values() if s > 1):>9,} {n / len(sizes):>5.1f} "
          f"{sizes[top]:>7,}  {top:15} {pairs:>13,} {skipped:>6,} bkts {big_pairs:>11,} pr {caught[k] / true_total:>10.1%}")
print(f"{'all keys (sum; overlaps twice)':30} {'':>9} {'':>9} {'':>5} {'':>7}  {'':15} {total_pairs:>13,} {'':>22} {caught['any'] / true_total:>10.1%}")
print("\n(dups caught = true pairs that share a key in a bucket we keep; skipped buckets are too big to compare)")
