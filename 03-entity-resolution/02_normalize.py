"""Step 2: clean and normalize. Plain code, no AI.

- names: lowercase, letters only
- postcode: keep only valid NC ZIPs (27xxx / 28xxx); blank the rest
- place: common spellings are trusted; rare ones snap to the closest common one
"""
import argparse
import time
from collections import defaultdict

import er

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
args = ap.parse_args()
rows = er.read_csv(er.DATA / f"sample_{args.records}.csv")

t0 = time.perf_counter()
# the place dictionary comes from the full 5M (a company would use a reference list of towns)
known = er.build_place_dictionary(er.load_raw()) if args.records != "all" else er.build_place_dictionary(rows)
by_letter = defaultdict(list)
for p in sorted(known):  # sorted: same result every run
    by_letter[p[0]].append(p)
t_dict = time.perf_counter() - t0

t0 = time.perf_counter()
fixed_place = blanked_pc = 0
cache = {}
for r in rows:
    r["n_givenname"] = er.clean_name(r["givenname"])
    r["n_surname"] = er.clean_name(r["surname"])
    place = er.clean_place(r["suburb"])
    if place not in cache:
        cache[place] = er.fix_place(place, known, by_letter)
    r["n_suburb"], changed = cache[place]
    fixed_place += changed
    r["n_postcode"] = er.clean_postcode(r["postcode"])
    blanked_pc += bool(r["postcode"]) and not r["n_postcode"]
t_norm = time.perf_counter() - t0

n = len(rows)
exact_before = len({tuple(r[f] for f in er.FIELDS) for r in rows})
exact_after = len({(r["n_givenname"], r["n_surname"], r["n_suburb"], r["n_postcode"]) for r in rows})
people = len({r["recid"] for r in rows})
print(f"place dictionary: {len(known):,} known places ({t_dict:.1f}s)")
print(f"normalized {n:,} records in {t_norm:.2f}s ({t_norm / n * 1e6:.1f} µs/record)")
print(f"  places snapped to a known spelling: {fixed_place:,}")
print(f"  postcodes blanked (not a NC ZIP):   {blanked_pc:,}")
print(f"exact de-dup: {exact_before:,} before -> {exact_after:,} after normalizing (truth: {people:,})")

out = er.DATA / f"normalized_{args.records}.csv"
er.write_csv(rows, out, ["id", "recid", "source"] + er.FIELDS + ["n_" + f for f in er.FIELDS])
print(f"-> {out.name}")
