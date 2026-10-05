# 01: Same Question, Opposite Answers

Can an LLM give **different answers to the identical request at temperature 0**?
Yes, on ambiguous inputs. This folder shows why, and what removes it.

Background: Horace He and Thinking Machines Lab, [*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Sep 2025). Providers batch requests together; GPU kernels change how they sum numbers depending on batch size (no *batch invariance*), so your result depends slightly on other people's traffic. That only matters when the model's top choices are nearly tied.

## Headline result

Qwen3-235B-A22B-2507 via OpenRouter, temperature 0, 30 identical requests per cell, provider not pinned.
Eight **fictional** companies asked BUY / HOLD / SELL under three system prompts:

| Knife-edge company | Vague prompt | Middle (factors, no rules) | Rubric |
|---|---|---|---|
| HALCYON | HOLD 30 | HOLD 30 | SELL 30 |
| MERIDIAN | HOLD 24 / BUY 6 | HOLD 30 | HOLD 30 |
| TESSERA | HOLD 30 | **BUY 20 / HOLD 7 / SELL 3** | HOLD 30 |
| QUILL | HOLD 30 | HOLD 30 | HOLD 30 |

The four clear-cut companies were 100% consistent on every prompt.

TESSERA, middle prompt, **within the same provider**:

| Provider | Answers |
|---|---|
| Alibaba | 4 BUY, 1 SELL |
| Novita | 9 BUY, 1 SELL, 1 HOLD |
| StreamLake | 2 BUY, 1 SELL |

Raw data, including every reply: [`results/2026-10-05_1145_stocks_qwen3-235b-a22b-2507.json`](results/2026-10-05_1145_stocks_qwen3-235b-a22b-2507.json)

### Findings
1. **Ambiguity is where it bites.** Clear-cut cases never flipped.
2. **HOLD is an escape hatch.** A vague prompt hides ambiguity behind the safe answer.
3. **Listing factors without a rule made it worse** (the middle prompt was least stable).
4. **A scoring rubric removed every flip**: 240/240 consistent and correct by the rubric's own arithmetic.
5. **Consistent ≠ correct.** Claude Haiku 4.5, pinned to one provider, never flipped, but called MERIDIAN SELL 20/20 when the rubric totals −1 (HOLD). See `results/2026-10-05_1107_stocks_claude-haiku-4.5_pinned.json`.
6. **Validate outputs.** Some providers occasionally return cut-off replies; the script now detects and retries them.

### Limits
- Providers are black boxes; within-provider differences are *consistent with* batching but can't rule out mixed hardware behind one provider.
- Small samples (30 per cell, uneven across providers), one task, one day.
- Companies are fictional and built to be ambiguous. **Nothing here is investment advice.**

## Run it

```bash
python3 01_why_batch_changes_answers.py          # local simulation, needs numpy
cp .env.example .env                              # add your OpenRouter key
python3 03_stock_classification.py 30 --model qwen/qwen3-235b-a22b-2507
python3 03_stock_classification.py 30 --pin --model anthropic/claude-haiku-4.5
python3 03_stock_classification.py --compare
```

No other dependencies (standard library only). A 30-run stock experiment costs a few cents.

| File | What it does |
|---|---|
| `01_why_batch_changes_answers.py` | Simulates batch-dependent summation and shows near-tie flips (0% / 1.6% / 38.7%) |
| `02_live_experiment.py` | Open-ended prompts (`--long` reproduces the Thinking Machines Feynman prompt) |
| `03_stock_classification.py` | The BUY/HOLD/SELL experiment, with per-provider breakdown |
