# Context strategy experiment, 2026-10-09-1628-gemini-3.1-pro-preview

Model `google/gemini-3.1-pro-preview`, provider `Google`, 3 runs per cell, temperature 0, max 12 turns. Total cost $0.28.

| Invoice | Strategy | Correct | Turns (mean) | Output tokens (mean) | of which reasoning | Prompt tokens (mean) | Cached | Cost (mean) | Look-alike found |
|---|---|---|---|---|---|---|---|---|---|
| INV-2026-1001 | full_history | 3/3 | 4.0 | 1,323 | 527 | 4,999 | 0 | $0.026 |  |
| INV-2026-1042 | full_history | 2/3 | 5.7 | 1,770 | 600 | 8,871 | 0 | $0.039 | 3/3 |
| INV-2026-1077 | full_history | 3/3 | 4.3 | 1,438 | 647 | 5,532 | 0 | $0.028 |  |

| Strategy | Correct | Total cost | Mean turns | Mean output tokens | of which reasoning |
|---|---|---|---|---|---|
| full_history | 8/9 | $0.28 | 4.7 | 1,510 | 591 |
