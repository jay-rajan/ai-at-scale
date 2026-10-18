"""Step 8: final answer. Code merges + Jev merges -> groups (people). Report against the answer key.

Jev: P(same) >= 0.7 merge, < 0.3 reject, in between -> a person (steward queue).
"""
import argparse
import json
import pickle
from collections import Counter

import er

ap = argparse.ArgumentParser()
ap.add_argument("--records", default="100000")
ap.add_argument("--merge", type=float, default=0.7)
ap.add_argument("--reject", type=float, default=0.3)
args = ap.parse_args()

rows = er.read_csv(er.DATA / f"normalized_{args.records}.csv")
ids = [r["id"] for r in rows]
recid = {r["id"]: r["recid"] for r in rows}
src = {r["id"]: r["source"] for r in rows}
truth = er.true_pairs(rows)
scored = pickle.load(open(er.DATA / f"scored_{args.records}.pkl", "rb"))
code_merge = {tuple(p) for p in scored["match"]}
jev = [json.loads(l) for l in open(er.RESULTS / f"jev_{args.records}.jsonl")]
jev = [o for o in jev if "choice" in o]

jev_merge = {(int(o["a"]), int(o["b"])) for o in jev if o["probabilities"]["same"] >= args.merge}
steward = [o for o in jev if args.reject <= o["probabilities"]["same"] < args.merge]
jev_reject = [o for o in jev if o["probabilities"]["same"] < args.reject]

def report(name, merges):
    groups = er.clusters_from(merges, ids)
    gid = {i: k for k, g in enumerate(groups) for i in g}
    pred = {(min(a, b), max(a, b)) for g in groups for a in g for b in g if a < b}
    p, r, f = er.prf(pred, truth)
    mixed = sum(1 for g in groups if len({recid[i] for i in g}) > 1)
    conflict = sum(1 for g in groups if len({src[i] for i in g}) < len(g))
    people = Counter(recid.values())
    split = sum(1 for pid in people if people[pid] > 1 and len({gid[i] for i in ids if recid[i] == pid}) > 1) if len(ids) <= 20000 else None
    return {"step": name, "customers_found": len(groups), "real_customers": len(people),
            "pair_precision": round(p, 4), "pair_recall": round(r, 4), "f1": round(f, 4),
            "groups_mixing_people": mixed, "groups_with_same_system_conflict": conflict}

res = {
    "jev_merge": len(jev_merge), "jev_merge_wrong": sum(p not in truth for p in jev_merge),
    "jev_reject": len(jev_reject), "jev_reject_real_lost": sum(o["truth"] for o in jev_reject),
    "steward": len(steward), "steward_real": sum(o["truth"] for o in steward),
    "after_code": report("code only", code_merge),
    "after_code_and_jev": report("code + Jev", code_merge | jev_merge),
}
print(json.dumps(res, indent=1))
json.dump(res, open(er.RESULTS / f"resolve_{args.records}.json", "w"), indent=1)
