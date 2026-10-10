# Turn engine experiment: fixed vs graph, 2026-10-10-1040-gemini-3.5-flash

model `google/gemini-3.5-flash`, provider `Google`, strategy `structured`, 3 runs per cell, max 12 turns. Total cost $1.71.

| Invoice | Engine | Correct | Turns | Turns with no data asked | Wall s | LLM s | Outside LLM, ms/turn | Cost |
|---|---|---|---|---|---|---|---|---|
| INV-2026-1001 | graph | 3/3 | 9.3 | 1.0 | 44.54 | 44.43 | 12.4 | $0.095 |
| INV-2026-1001 | fixed | 3/3 | 7.0 | 1.0 | 37.19 | 37.14 | 7.8 | $0.077 |
| INV-2026-1042 | graph | 3/3 | 9.3 | 1.7 | 57.89 | 57.71 | 18.2 | $0.118 |
| INV-2026-1042 | fixed | 2/3 | 10.3 | 1.3 | 61.37 | 61.29 | 7.8 | $0.134 |
| INV-2026-1077 | graph | 3/3 | 7.3 | 1.0 | 37.22 | 37.13 | 13.1 | $0.076 |
| INV-2026-1077 | fixed | 3/3 | 7.3 | 1.3 | 33.16 | 33.11 | 7.2 | $0.073 |

| Engine | Correct | Mean turns | Outside LLM, ms/turn (median) | Share of wall time in LLM | Total cost |
|---|---|---|---|---|---|
| graph | 9/9 | 8.7 | 14.2 | 99.7% | $0.87 |
| fixed | 8/9 | 8.2 | 7.6 | 99.9% | $0.85 |
