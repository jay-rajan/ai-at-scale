# 03: Entity resolution at scale

How many real people are in 5 million person records, and where does AI help?
Code does almost all of the work. A model only sees the pairs code can't decide.

## The data

[North Carolina Voters 5M](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution) from the University of Leipzig database group (CC BY 4.0). 5 sources x 1M records of real voter names with generated duplicates (typos, missing values). Fields: given name, surname, town, postcode. Each source has no duplicates of its own. `recid` says which records are the same person: the pipeline never uses it to decide, only to grade.

```bash
cd 03-entity-resolution
mkdir -p data/raw && cd data/raw
curl -O https://dbs.uni-leipzig.de/files/datasets/saeedi/5Party-ocp20.tar.gz   # ~74 MB
tar xzf 5Party-ocp20.tar.gz
cd ../..
pip install -r requirements.txt
```

Scripts `05_llm.py` and `07_jev.py` call [OpenRouter](https://openrouter.ai/). Put `OPENROUTER_API_KEY=...` in a `.env` file in this folder. Scripts 00–04 and 08 need no key and cost nothing.

## Run it

One script per step. `--records` is the sample size (whole people are sampled, seed 7); use `all` for the full 5M where a step supports it.

| Step | Script | What it does | Run |
|---|---|---|---|
| 0 Sample | `00_prepare.py` | Take a sample of whole people | `python 00_prepare.py --records 100000` |
| 1 Profile | `01_profile.py` | Rows vs exact de-dup vs real people, field problems | `python 01_profile.py --records 100000` |
| 2 Clean | `02_normalize.py` | Lowercase, letters only, valid NC ZIPs, snap rare town spellings to known towns | `python 02_normalize.py --records 100000` |
| 3 Bucket | `03_block.py` | 4 cheap keys; only records sharing a key are compared; same-source pairs skipped | `python 03_block.py --records 100000` |
| 4 Score | `04_score.py` | Weighted Jaro-Winkler + postcode distance: ≥ 0.93 merge, < 0.80 reject, else unsure | `python 04_score.py --records 100000` |
| 5 Jev | `07_jev.py` | Unsure pairs to Jev (a decision model with probabilities), one pair per call | `python 07_jev.py --records 100000 --workers 100` |
| 6 Resolve | `08_resolve.py` | Merges → groups (union-find), graded against the answer key; same-system conflict check | `python 08_resolve.py --records 100000` |

Side experiments:

| Script | Question |
|---|---|
| `03b_minhash.py` | MinHash LSH instead of hand-picked keys: pairs vs recall |
| `03c_bucket_stats.py` | Bucket sizes and recall at 10k, 100k and 5M (`--all-keys`, `--split`) |
| `04b_score_all.py` | Score every bucketed pair of the full 5M (235.6M pairs, ~94 s) |
| `04c_unsure_buckets.py` | Which buckets still hold unsure pairs at 5M (`--split`) |
| `05_llm.py` | Chat LLMs (Gemini Flash Lite / Flash, Claude Haiku) on the unsure pairs, 10 per call |
| `06_when_in_doubt.py` | Merge only above a self-reported confidence; no new calls |

`er.py` holds the shared helpers: loading, cleaning, bucket keys, scoring.

## Key results (100,001 records, 70,269 real people)

| Step | Time | Cost | After |
|---|---|---|---|
| Bucket | 1.1 s | $0 | 5.0B possible pairs → 895,696 (97.8% of true pairs) |
| Score | 2.3 s | $0 | merge 63,366 · reject 747,774 · unsure 84,556 |
| Jev (P ≥ 0.7 merge / < 0.3 reject) | ~7 min | $1.72 | merge 1,233 (233 wrong) · reject 78,162 · to a person 5,161 |
| Resolve | 0.7 s | $0 | 70,158 people |

| | Code only | Code + Jev |
|---|---|---|
| Precision / recall | 98.5% / 96.7% | 97.3% / 98.0% |
| Groups holding two records from one system | 99 | 200 |

The Jev cut-off that made 0 wrong merges at 10,000 records was 81% right at 100,000. Full numbers, the 5M runs and the chat-model comparison: [`results/RESULTS.md`](results/RESULTS.md). Presentation: `results/pipeline_100000.html`.

## Limits

- One dataset, 4 fields, generated duplicates. Real customer data has more fields and messier errors.
- Thresholds and cut-offs were chosen on labelled data; confirm on a fresh sample.
- Jev and chat-model results are single runs on an alpha API; `jev_100000.json` (27 MB) isn't in the repo, rerun step 5 to get it.
- The answer key is itself not perfectly reliable.
