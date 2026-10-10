# Turn engine experiment: fixed vs graph, 2026-10-10-1039-offline

offline (FakeModel, no cost), strategy `structured`, 10 runs per cell, max 12 turns. Total cost $0.00.

| Invoice | Engine | Correct | Turns | Turns with no data asked | Wall s | LLM s | Outside LLM, ms/turn | Cost |
|---|---|---|---|---|---|---|---|---|
| INV-2026-1001 | graph | 10/10 | 4.0 | 1.0 | 0.01 | 0.00 | 2.7 | $0.000 |
| INV-2026-1001 | fixed | 10/10 | 4.0 | 1.0 | 0.00 | 0.00 | 1.0 | $0.000 |
| INV-2026-1042 | graph | 10/10 | 4.0 | 1.0 | 0.01 | 0.00 | 2.7 | $0.000 |
| INV-2026-1042 | fixed | 10/10 | 4.0 | 1.0 | 0.00 | 0.00 | 1.0 | $0.000 |
| INV-2026-1077 | graph | 10/10 | 4.0 | 1.0 | 0.01 | 0.00 | 2.7 | $0.000 |
| INV-2026-1077 | fixed | 10/10 | 4.0 | 1.0 | 0.00 | 0.00 | 1.0 | $0.000 |

| Engine | Correct | Mean turns | Outside LLM, ms/turn (median) | Share of wall time in LLM | Total cost |
|---|---|---|---|---|---|
| graph | 30/30 | 4.0 | 2.6 | 0.9% | $0.00 |
| fixed | 30/30 | 4.0 | 1.0 | 2.5% | $0.00 |

Same recommendation, evidence and flags under both engines (deterministic model): **yes**.
