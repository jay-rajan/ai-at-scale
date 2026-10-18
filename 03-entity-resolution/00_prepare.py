"""Step 0: take a sample of whole people from the 5M NC Voters records.

Usage: python 00_prepare.py --records 10000      (writes data/sample_10000.csv)
       python 00_prepare.py --records all
"""
import argparse
import time

import er

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="10000")
args = ap.parse_args()

t0 = time.perf_counter()
rows = er.load_raw()
print(f"loaded {len(rows):,} records in {time.perf_counter() - t0:.1f}s")
if args.records != "all":
    rows = er.sample_by_person(rows, int(args.records))
out = er.DATA / f"sample_{args.records}.csv"
er.write_csv(rows, out, ["id", "recid", "source"] + er.FIELDS)
print(f"wrote {len(rows):,} records, {len({r['recid'] for r in rows}):,} real people -> {out.name}")
