"""Step 1: look at the data before touching it. How many customers do we really have?"""
import argparse
import re
from collections import Counter

import er

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
args = ap.parse_args()
rows = er.read_csv(er.DATA / f"sample_{args.records}.csv")

n = len(rows)
people = len({r["recid"] for r in rows})
exact = len({tuple(r[f] for f in er.FIELDS) for r in rows})
print(f"records: {n:,}   sources: {dict(sorted(Counter(r['source'] for r in rows).items()))}")
print(f"'customers' if you count rows:               {n:,}")
print(f"'customers' after exact de-duplication:      {exact:,}")
print(f"real people (ground truth):                  {people:,}")
print(f"records per person: {dict(sorted(Counter(Counter(r['recid'] for r in rows).values()).items()))}")

print("\nfield problems:")
for f in er.FIELDS:
    missing = sum(1 for r in rows if not r[f].strip())
    print(f"  {f:10} missing {missing / n:6.1%}")
bad_name = sum(1 for r in rows if re.search(r"[^a-z]", r["givenname"] + r["surname"]))
ZIP = re.compile(r"2[78]\d{3}")
bad_pcs = [r["postcode"] for r in rows if r["postcode"] and not ZIP.fullmatch(r["postcode"])]
print(f"  names with non-letters      {bad_name / n:6.1%}")
print(f"  postcodes that aren't NC ZIPs {len(bad_pcs) / n:6.1%}   e.g. {bad_pcs[:5]}")

# what corruption looks like: two records of the same person that differ
by_person = {}
diffs = Counter()
for r in rows:
    by_person.setdefault(r["recid"], []).append(r)
for recs in by_person.values():
    a = recs[0]
    for b in recs[1:]:
        for f in er.FIELDS:
            if a[f] != b[f]:
                diffs[f] += 1
pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_person.values())
print(f"\nsame person, different value (of {pairs:,} true pairs, vs first record): {dict(diffs)}")
