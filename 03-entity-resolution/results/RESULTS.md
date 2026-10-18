# Results (2026-10-16)

Data: North Carolina Voters 5M (Univ. of Leipzig, CC BY 4.0). Samples of whole people, seed 7.
Thresholds fixed at 10k (auto match >= 0.93, auto no match < 0.80) and reused unchanged at 100k.
LLM: google/gemini-3.5-flash-lite via OpenRouter, temperature 0, 10 pairs per call.

| | 10,000 records | 100,000 records | Growth |
|---|---|---|---|
| Real people | 6,926 | 70,269 | 10x |
| "Customers" after exact de-dup | 7,736 | 78,031 | |
| Possible pairs | 50M | 5.0B | 100x |
| True duplicate pairs | 6,843 | 65,910 | 9.6x |
| Candidate pairs after bucketing | 15,634 | 895,655 | **57x** |
| True pairs no bucket caught | 163 (2.4%) | 1,422 (2.2%) | |
| Bucketing + scoring time (laptop) | 0.05 s | 1.9 s | |
| Code auto-merges | 6,518 | 63,379 | |
| Wrong auto-merges | 2 | 680 | **340x** |
| Code precision / recall | 99.97% / 95.2% | 98.9% / 95.1% | |
| Unsure pairs (LLM's slice) | 967 | 84,569 | **87x** |
| Real duplicates among unsure | 164 (17%) | 1,761 (2.1%) | |
| LLM accuracy on unsure | 98.3% | 99.1% (5,000-pair sample) | |
| LLM wrong merges | 11 | ~677 (projected) | |
| LLM cost / time | $0.09 / 20 s | ~$7.60 / ~16 min (projected) | 86x |
| Final precision / recall (code + LLM) | 99.8% / 97.5% | ~97.9% / ~97.5% | |

Other measured runs at 10k (967 unsure pairs): Claude Haiku 5.5 94.7% (51 wrong merges, $0.04); Gemini 3.5 Flash 99.1% (5 wrong, $1.41). Flash Lite run twice: 10 of 967 answers flipped.

Limits that no method fixes from these fields: two different "rufus mcaden, new bern, 28560" (identical records, different people); "jonathan brown" / "john brown", same town and postcode, different people. NCID (the answer key) is itself not perfectly reliable.

## Don't merge if in doubt (06_when_in_doubt.py, no new calls)

Merge only when the LLM says "same" with self-reported confidence >= cut-off; the rest go to a steward. Full table: `when_in_doubt.txt`.

| Run | Merge always: wrong / real dups merged | conf >= 0.9 | conf >= 0.95 |
|---|---|---|---|
| Flash Lite, 10k (164 real dups) | 11 / 159 | 6 / 88 (76 to steward) | 3 / 28 (139 to steward) |
| Haiku 5.5, 10k | 51 / 164 | 2 / 92 (121) | 2 / 34 (179) |
| Flash, 10k | 5 / 160 | 2 / 115 (48) | 0 / 32 (133) |
| Flash Lite, 100k sample (98 real dups) | 40 / 94 | 3 / 50 (81) | 2 / 18 (114) |

Read: the confidence is a number the model writes, not a measured probability. Raising the bar cuts wrong merges but sends most real duplicates to stewards, and some wrong merges come at 0.95. The prompt offered only same/different; an explicit "unsure" answer is untested.

## Jev 1.13 (typesafe/jev-1.13-20260917), decision model, 967 unsure pairs at 10k (07_jev.py)

One pair per call, choice same / different / unsure with probabilities. 0 errors, $0.0197 total, 0.19 s median per call, 25 s wall (12 at a time).

- By its own choice: 196 "same" (34 wrong), 663 "different" (0 real duplicates), 108 "unsure" (2 real duplicates).
- By P(same): >= 0.7 merged 99, **0 wrong**; < 0.3 rejected 741, **0 real duplicates lost**; 0.3-0.7 to steward: 127 pairs (65 real duplicates).
- Calibration: P(same) < 0.3 → 0% real; 0.5-0.6 → 68%; 0.6-0.7 → 93%; >= 0.7 → 100%.
- Caveats: cut-offs chosen on the same labelled pairs (needs a fresh sample to confirm); one run only (consistency untested); the Rufus McAden and Brown pairs were auto-merged by code and never reached Jev.

## MinHash LSH vs hand-picked bucket keys (03b_minhash.py)

| Method | 10k: pairs / true pairs found | 100k: pairs / found | Buckets at 100k | Time at 100k |
|---|---|---|---|---|
| Hand keys (4 passes) | 15,634 / 97.6% | 895,655 / 97.8% | — | 0.6 s |
| MinHash 30x5, 3-grams, name+town | 30,277 / 97.1% | 2,313,674 / 96.8% | 2,098,821 | 4.6 s |
| MinHash 20x5, 2-grams, name+town | 71,051 / 98.9% | 6,123,066 / 99.0% | 1,072,506 | 6.2 s |

Read: for short structured person fields, hand keys give ~2.5x fewer pairs at the same recall. MinHash needs no domain knowledge and buys recall with more pairs (b, r are the dial).

## Bucketing with 3 keys at 10k, 100k and 5M (03c_bucket_stats.py, pairs counted not stored)

Buckets over 200 records are skipped (too common to compare). "Caught" = true duplicate pairs that share a key in a kept bucket.

| Key | 10k: buckets / biggest / pairs / caught | 100k | 5M |
|---|---|---|---|
| Surname sound + first initial | 5,815 / 21 / 9,957 / 86.1% | 33,789 / 177 / 457,244 / 86.1% | 344,869 / 7,397 (SM0-j) / 66.7M / **46.3%** |
| Surname + postcode | 6,944 / 9 / 5,691 / 77.8% | 56,798 / 30 / 92,041 / 78.0% | 610,804 / 1,105 (loc-28372) / 72.5M / 74.3% |
| First name + town | 6,003 / 35 / 10,272 / 84.1% | 39,470 / 319 / 445,193 / 84.1% | 415,685 / 14,544 (mar-charlotte) / 66.2M / **54.5%** |
| All 3 | 25,920 pairs (sum) / 97.3% | 994,478 / 97.6% | 205.4M / **90.3%** |

At 5M, 8,517 buckets are too big; they would add ~1.93 billion pairs. Keys that worked at 100k lose 7 points of recall at 5M.
Fix (2026-10-16): town snapping now uses a sorted dictionary; ties broke differently between runs before.

### Splitting big buckets with a finer key at 5M (03c_bucket_stats.py --split)

Only buckets over 200 get one more field: surname sound + initial → + town; surname + postcode → + first initial; first name + town → + first 2 letters of surname sound.

| | Skip big buckets | Split big buckets | Keep big buckets |
|---|---|---|---|
| Pairs (sum of 3 keys) | 205.4M | 233.8M | ~2,139M |
| Duplicates caught | 90.3% | **95.1%** | ~97–98% (estimate) |
| Buckets still too big | 8,517 | 161 (e.g. SM0-j-charlotte: 516) | — |

### Adding "first name sound + surname sound" at 5M (--all-keys)

| | 3 keys, skip | 3 keys, split | 4 keys, skip | 4 keys, split |
|---|---|---|---|---|
| Pairs (sum) | 205.4M | 233.8M | 246.6M | 275.6M |
| Duplicates caught | 90.3% | 95.1% | 93.8% | **96.8%** |

The both-sounds key alone: 78% at 10k/100k (weakest), 74.5% at 5M with skip (strongest single key): 1.42M buckets, avg 3.5, biggest JN-SM0 (john smith) 1,263.

## Code rules on all bucketed pairs of the full 5M (04b_score_all.py)

4 keys, buckets > 200 skipped, each pair scored once: **235.6M distinct pairs in 94 s** (4 cores, 1.6 µs/pair/core). Same thresholds as 10k.

| | Score only | Score + first-name guard |
|---|---|---|
| Merge: right / wrong | 3,057,082 / **857,650** | 3,042,396 / 630,664 |
| Merge precision | 78.1% | 82.8% |
| Unsure: real dups / different | 68,486 / 63,717,962 | 83,172 / 63,944,948 |
| No: real dups lost | 509 | 509 |
| Real duplicate pairs merged | 91.8% of 3,331,384 | 91.3% |

Thresholds tuned at 10k (precision 99.97%) give 78% precision at 5M. Unsure pile: 63.8M pairs, 0.1% real duplicates.

## Group-level (hierarchical) step at 100k, no model calls

Code merges → groups (union-find). Unsure pairs then judged group vs group:
- 84,569 unsure pairs → 682 already inside one group → 48,525 distinct group pairs.
- Group rules: reject if the merge would put two records of one system together, or average score across all record pairs < 0.85; merge if >= 0.93.
- Result: 41,158 group pairs rejected (16 real lost), 0 merged, **7,367 still unsure** (390 real). 84,569 → 7,367 = 11.5x fewer for a model or steward.

## Skip vs split big buckets, scored end to end at 5M (04c_unsure_buckets.py [--split]), guard rule on

| | Skip > 200 | Split > 200 with a finer key |
|---|---|---|
| Distinct pairs scored | 235.6M (94 s) | 257.7M (106 s) |
| Merged: right / wrong | 3,042,396 / 630,664 | 3,128,869 / 882,893 |
| Unsure pairs (real dups) | 64.0M (83,172) | 74.3M (95,718) |
| Real dups merged or unsure | 93.8% | 96.8% |
| Buckets holding unsure pairs | 586,511 | 940,883 |
| ...of which 51–200 records | 48,709 (47.3M unsure) | 52,975 (51.3M unsure) |

Read: splitting recovers duplicates the skip lost, but the unsure pile grows. Finer keys separate records that differ; they cannot separate lookalikes (two john smiths in charlotte share every key at every level).

## Full run at 100k with Jev on every unsure pair (2026-10-16)

| Step | Time | Cost | After |
|---|---|---|---|
| 0 Sample | 7.7 s | $0 | 100,001 records (70,269 people) |
| 1 Profile | 0.2 s | $0 | exact de-dup 78,031 |
| 2 Clean | 7.6 s (7.1 s town list from 5M) | $0 | 2,252 towns fixed, 2,053 postcodes blanked |
| 3 Bucket (4 keys) | 1.1 s | $0 | 5.0B → 895,696 pairs (97.8% of true pairs) |
| 4 Score | 2.3 s | $0 | merge 63,366 (680 wrong) · reject 747,774 (28 real) · unsure 84,556 (1,761 real) |
| 5 Jev (P ≥ 0.7 / < 0.3) | ~7.2 min at up to 100 parallel | $1.72 | merge 1,233 (233 wrong) · reject 78,162 (14 real) · person 5,161 (747 real) |
| 6 Resolve | 0.7 s | $0 | 70,158 people (real 70,269) |

After grouping: code only P 98.5% / R 96.7%, 334 mixed groups, 99 same-system conflicts; code + Jev P 97.3% / R 98.0%, 483 mixed, 200 conflicts.
Jev cut-off at 100k: ≥0.7 81% right (0 wrong at 10k); ≥0.9 402 merged, 35 wrong. OpenRouter rate-limited Jev (180 of 84,556 calls got 429, re-asked).
Presentation: results/pipeline_100000.html
