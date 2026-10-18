"""Shared helpers for the entity resolution experiment.

Data: North Carolina Voters 5M (University of Leipzig, CC BY 4.0).
5 sources x 1M person records. Each source is duplicate-free; `recid` is the
ground truth (same recid = same person). Fields: givenname, surname, suburb, postcode.
"""
import csv
import random
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import jellyfish
from rapidfuzz.distance import JaroWinkler, Levenshtein

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw" / "5Party-ocp20"
DATA = HERE / "data"
RESULTS = HERE / "results"
FIELDS = ["givenname", "surname", "suburb", "postcode"]


# ---------- loading ----------

def load_raw():
    """All 5M records as dicts with a `source` (0-4) and a row `id`."""
    rows = []
    for path in sorted(RAW.glob("*.csv")):
        source = int(re.search(r"myp_(\d)", path.name).group(1))
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r["source"] = source
                rows.append(r)
    for i, r in enumerate(rows):
        r["id"] = i
    return rows


def sample_by_person(rows, n_records, seed=7):
    """Pick whole people (all their records) until we have ~n_records.

    Sampling records at random would break up duplicates; sampling people keeps them.
    """
    by_person = defaultdict(list)
    for r in rows:
        by_person[r["recid"]].append(r)
    people = list(by_person)
    random.Random(seed).shuffle(people)
    out = []
    for p in people:
        out.extend(by_person[p])
        if len(out) >= n_records:
            break
    for i, r in enumerate(out):
        r["id"] = i
    return out


def write_csv(rows, path, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["id"] = int(r["id"])
        r["source"] = int(r["source"])
    return rows


# ---------- truth ----------

def true_pairs(rows):
    """Every pair of row ids that refer to the same person."""
    by_person = defaultdict(list)
    for r in rows:
        by_person[r["recid"]].append(r["id"])
    pairs = set()
    for ids in by_person.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    return pairs


# ---------- normalizing ----------

def clean_name(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def clean_place(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def clean_postcode(s):
    """Keep it only if it looks like a NC ZIP (27xxx or 28xxx). Otherwise blank."""
    s = (s or "").strip()
    return s if re.fullmatch(r"2[78]\d{3}", s) else ""


def build_place_dictionary(rows, min_count=50):
    """Places that appear often are trusted spellings. Rare ones are probably typos."""
    counts = defaultdict(int)
    for r in rows:
        counts[clean_place(r["suburb"])] += 1
    return {p for p, c in counts.items() if p and c >= min_count}


def fix_place(place, known, by_first_letter, threshold=0.92):
    """Snap a rare spelling to the closest common place (e.g. 'millers ceeek' -> 'millers creek')."""
    if not place or place in known:
        return place, False
    best, best_score = place, 0.0
    for cand in by_first_letter.get(place[0], ()):
        sc = JaroWinkler.similarity(place, cand)
        if sc > best_score:
            best, best_score = cand, sc
    return (best, True) if best_score >= threshold else (place, False)


# ---------- blocking keys ----------

def metaphone(s):
    return jellyfish.metaphone(s) if s else ""


BLOCKING_PASSES = {
    # name: function(row) -> key, or "" to skip the row in this pass
    "sound(surname)+first initial": lambda r: (metaphone(r["n_surname"]) + "|" + r["n_givenname"][:1])
        if r["n_surname"] and r["n_givenname"] else "",
    "sound(given)+sound(surname)": lambda r: (metaphone(r["n_givenname"]) + "|" + metaphone(r["n_surname"]))
        if r["n_surname"] and r["n_givenname"] else "",
    "surname[:3]+postcode": lambda r: (r["n_surname"][:3] + "|" + r["n_postcode"])
        if r["n_surname"] and r["n_postcode"] else "",
    "given[:3]+place": lambda r: (r["n_givenname"][:3] + "|" + r["n_suburb"])
        if r["n_givenname"] and r["n_suburb"] else "",
}


def block(rows, passes=BLOCKING_PASSES, max_block=200):
    """Candidate pairs from each pass. Blocks bigger than max_block are skipped (too common to help)."""
    per_pass = {}
    for name, keyfn in passes.items():
        t0 = time.perf_counter()
        blocks = defaultdict(list)
        for r in rows:
            k = keyfn(r)
            if k:
                blocks[k].append(r["id"])
        pairs, skipped, biggest = set(), 0, 0
        for ids in blocks.values():
            if len(ids) > max_block:
                skipped += len(ids)
                continue
            biggest = max(biggest, len(ids))
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.add((ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i]))
        per_pass[name] = {
            "pairs": pairs,
            "blocks": len(blocks),
            "biggest": biggest,
            "rows_in_skipped_blocks": skipped,
            "seconds": time.perf_counter() - t0,
        }
    return per_pass


# ---------- scoring ----------

def compare(a, b):
    """Field-by-field similarity, 0..1 (None when a field is missing on either side)."""
    def sim(x, y):
        return JaroWinkler.similarity(x, y) if x and y else None
    pc = None
    if a["n_postcode"] and b["n_postcode"]:
        pc = 1.0 - Levenshtein.distance(a["n_postcode"], b["n_postcode"]) / 5
    return {
        "given": sim(a["n_givenname"], b["n_givenname"]),
        "surname": sim(a["n_surname"], b["n_surname"]),
        "place": sim(a["n_suburb"], b["n_suburb"]),
        "postcode": pc,
    }


WEIGHTS = {"given": 0.30, "surname": 0.35, "place": 0.15, "postcode": 0.20}


def score(features):
    """Weighted average over the fields both records have."""
    total = weight = 0.0
    for k, w in WEIGHTS.items():
        v = features[k]
        if v is not None:
            total += w * v
            weight += w
    return total / weight if weight else 0.0


# ---------- clustering ----------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def clusters_from(pairs, ids):
    uf = UnionFind()
    for i in ids:
        uf.find(i)
    for a, b in pairs:
        uf.union(a, b)
    groups = defaultdict(list)
    for i in ids:
        groups[uf.find(i)].append(i)
    return list(groups.values())


def prf(predicted, truth):
    tp = len(predicted & truth)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(truth) if truth else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
