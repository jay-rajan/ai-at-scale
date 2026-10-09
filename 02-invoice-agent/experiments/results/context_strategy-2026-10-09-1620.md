# Context strategy experiment, 2026-10-09-1620

Model `google/gemini-3.5-flash`, provider `Google`, 3 runs per cell, temperature 0, max 12 turns. Total cost $1.08.

| Invoice | Strategy | Correct | Turns (mean) | Output tokens (mean) | Prompt tokens (mean) | Cached | Cost (mean) | Look-alike found |
|---|---|---|---|---|---|---|---|---|
| INV-2026-1001 | structured | 3/3 | 8.0 | 7,003 | 12,631 | 0 | $0.082 |  |
| INV-2026-1001 | full_history | 3/3 | 6.3 | 1,454 | 8,807 | 0 | $0.026 |  |
| INV-2026-1042 | structured | 3/3 | 9.3 | 9,970 | 17,864 | 0 | $0.117 | 3/3 |
| INV-2026-1042 | full_history | 0/3 | 7.7 | 1,717 | 12,886 | 0 | $0.035 | 3/3 |
| INV-2026-1077 | structured | 3/3 | 7.0 | 6,225 | 10,434 | 0 | $0.072 |  |
| INV-2026-1077 | full_history | 3/3 | 6.7 | 1,513 | 9,565 | 0 | $0.028 |  |

| Strategy | Correct | Total cost | Mean turns | Mean output tokens |
|---|---|---|---|---|
| structured | 9/9 | $0.81 | 8.1 | 7,733 |
| full_history | 6/9 | $0.27 | 6.9 | 1,561 |
