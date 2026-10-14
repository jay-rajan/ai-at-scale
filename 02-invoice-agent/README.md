# 02: Invoice Investigation Agent

An enterprise fintech demo: an LLM agent that **investigates** suspicious invoices instead of following a fixed checklist. It pauses for a human when it can't safely approve, and resumes with their instructions, durably, even days later.

The scenario it's built around is a common payment fraud: an invoice from a real vendor, sent to a **bank account changed three days ago**, by an employee, **without the required second approval**. (Business email compromise cost US businesses $3.05 billion in 2025 according to the [FBI IC3 2025 report](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf).)

All companies, people and accounts in `data/` are fictional.

## Follow the article step by step

Companion code for **[Do you even need an AI agent? Rules vs agents vs Temporal vs LangGraph](https://theaiswitch.substack.com/p/do-you-even-need-an-ai-agent-rules)** on The AI Switch. Each step of the article maps to code here:

| Step | What it adds | Where | Run it |
|---|---|---|---|
| 0. Extraction | Invoice PDF → fields | Not built. `data/enterprise.json` holds invoices as already-extracted fields | |
| 1. Rules | Fixed checks: approved vendor, PO match, verified bank, second approval | Not built as code. The policies live in `data/enterprise.json` → `approval_policies`, served by `get_approval_policy` | |
| 2. Fixed pipeline + one LLM call | Code fetches the data, the LLM judges once | Not built yet | |
| 3. The agent loop | The model picks the next tool (MCP) each turn | `agent/loop.py`, `agent/state.py`, `enterprise_finance_mcp/server.py` | `python cli.py investigate INV-2026-1042` ([Run it §1](#run-it)) |
| 4. + Temporal | One activity per agent turn; waits days for a human; survives crashes | `workflow/` | [Run it §2](#run-it) |
| 5. LangGraph | The turn and the loop as a graph, `interrupt()` for human review, SQLite checkpoints | `agent/turn_graph.py`, `langgraph_agent/` | `python -m langgraph_agent start INV-2026-1042` ([Run it §6](#run-it)) |
| 6a. Temporal + LangGraph | The whole graph inside **one** Temporal activity; Temporal retries it after a crash | `graph_in_temporal/` | [Run it §4](#run-it) |
| 6b. Temporal's LangGraph plugin | Each node as its own Temporal activity, visible in the Temporal UI; no SQLite checkpoints | `graph_plugin_in_temporal/` | [Run it §4b](#run-it) |
| Test your own setup | Crash, overnight wait, failure and repeat-run checks | `tests/` | `pytest` ([Tests](#tests)) |

The invoice in the article is `INV-2026-1042` (see [Scenarios](#scenarios)).

## What's inside

```
                 ┌───────────────── Temporal workflow (business state) ─────────────────┐
 invoice ──▶     │ RECEIVED → INVESTIGATING → APPROVED                                  │
                 │                 │  ▲                                                 │
                 │                 ▼  │ "investigate further" + instruction             │
                 │            HUMAN_REVIEW ──▶ APPROVED / REJECTED  (waits indefinitely)│
                 └──────┬───────────────────────────────────────────────────────────────┘
                        │ one activity per agent turn
                        ▼
          ┌──────── explicit agent loop (agent/loop.py) ────────┐
          │ observe state → LLM picks a tool → MCP call →        │
          │ evidence E# → record_findings → … → conclude          │
          └───────┬───────────────────────────┬──────────────────┘
                  ▼                           ▼
        OpenRouter (any model)     enterprise-finance-mcp (stdio)
                                   invoices · POs · vendors · bank accounts ·
                                   change history · policies · employees · payments
```

| Path | What it is |
|---|---|
| `enterprise_finance_mcp/server.py` | One MCP server standing in for ERP, procurement, vendor master, HR and payments. 8 read-only tools over `data/enterprise.json`. |
| `agent/state.py` | Explicit investigation state: facts, evidence, open questions, risk flags, tool history, human messages, status. |
| `agent/context.py` | What the model sees each turn. `structured` (default): a fresh brief built from state. `full_history`: the whole transcript, for comparison. |
| `agent/loop.py` | The loop and the rules of each phase (think, record, gather, decide). One turn = one model call. Guardrail: code, not the model, decides whether an "approve" can stand. |
| `agent/turn_graph.py` | One turn as a LangGraph graph: `think → record → [gather] → decide`. The single definition of a turn's flow; `agent_step` runs it (`engine="fixed"` keeps the original hand-written order for comparison). |
| `graph_in_temporal/` | Build C1: the whole LangGraph investigation inside **one** Temporal activity. LangGraph runs the loop and checkpoints every node (SQLite, `langgraph-temporal.db`); Temporal restarts the activity after a crash and waits for the human. |
| `graph_plugin_in_temporal/` | Build 6b: the same graph on Temporal's official LangGraph plugin (`temporalio-langgraph`, public preview). Every node is its own activity; the graph's progress is the workflow history, so no checkpoint DB. Routers are async (the workflow can't run sync functions in a thread pool). |
| `langgraph_agent/` | The whole investigation as a local LangGraph graph: the turn, the loop, and human review with `interrupt()`, checkpointed to SQLite (`langgraph.db`). No hosted services. |
| `agent/llm.py` | Small OpenRouter client: tool calling, retries, usage and cost. |
| `agent/trace.py` | JSON Lines trace: every model call, the exact context sent, tools chosen, results, state before and after, tokens, cost, stop reason. |
| `workflow/` | Temporal workflow, activities, worker, local dev server, and CLI client commands. |
| `tests/` | Offline deterministic tests (fake model), and live scenario tests (real model). |
| `experiments/` | `context_strategy.py`: structured brief vs full history across invoices, models and runs. `turn_engine.py`: the old fixed turn vs the turn graph. Results in `experiments/results/`. |

## Setup

From the repo root, follow the [main README](../README.md#setup) for Python and the venv, or do it in this folder:

```bash
cd 02-invoice-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your OpenRouter key
```

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys). The default model is `google/gemini-3.5-flash`; any OpenRouter model with tool calling works (`AGENT_MODEL` in `.env`).

Optional settings in `.env`:

| Setting | Meaning |
|---|---|
| `CONTEXT_STRATEGY` | `structured` (default) or `full_history` |
| `AGENT_PROVIDER` | pin one OpenRouter provider, no fallbacks (e.g. `Google`). Unpinned, calls can land on different backends |
| `PROMPT_CACHE=1` | mark a prompt-cache breakpoint on each call. Pays off only when the provider caches prompts this size (see results) |
| `MAX_TURNS`, `MAX_COST_USD` | the agent's turn and cost budget |

## Run it

**1. Just the agent (no Temporal), the quickest way to see it think:**
```bash
python cli.py investigate INV-2026-1042
python cli.py investigate INV-2026-1042 --instruction "Check whether this vendor has had previous bank-account changes."
python cli.py trace INV-2026-1042                 # turn-by-turn: tools, results, tokens, state changes
python cli.py trace INV-2026-1042 --turn 3 --full # the exact context the model saw on turn 3
```

**2. The durable workflow with a human in the loop** (three terminals):
```bash
python -m workflow.dev_server       # local Temporal + UI at http://localhost:8233 (downloaded automatically)
python -m workflow.worker           # runs workflow code and agent turns
python cli.py start INV-2026-1042   # then:
python cli.py status INV-2026-1042  # INVESTIGATING … HUMAN_REVIEW, with findings and evidence
python cli.py decide INV-2026-1042 investigate "Check whether this vendor has had previous bank-account changes."
python cli.py decide INV-2026-1042 reject        # or approve
```
Prefer the official CLI? `brew install temporal`, then `temporal server start-dev --db-filename temporal-dev.db` instead of `python -m workflow.dev_server`.

While it waits in `HUMAN_REVIEW`, nothing is running: stop the worker, restart your laptop, come back tomorrow, and send the decision. The workflow state lives in Temporal (`temporal-dev.db`).

**3. The context-strategy experiment:**
```bash
python -m experiments.context_strategy                       # 3 invoices x 2 strategies x 3 runs (~$1)
python -m experiments.context_strategy --model google/gemini-3.1-pro-preview --strategies full_history
python -m experiments.context_strategy --model anthropic/claude-sonnet-5.5 --provider Anthropic \
       --strategies full_history --invoices INV-2026-1042 --runs 2 --cache
```
Each run writes a normal trace to `logs/`, and a summary table to `experiments/results/`.

**4. LangGraph inside Temporal (build C1):**
```bash
temporal server start-dev --db-filename temporal-dev.db     # terminal 1
python -m graph_in_temporal.worker                           # terminal 2
python -m graph_in_temporal start INV-2026-1042              # terminal 3
python -m graph_in_temporal history INV-2026-1042            # LangGraph checkpoints, also while it runs
kill -9 <worker pid>; python -m graph_in_temporal.worker     # crash it mid-run; it continues within ~30 s
python -m graph_in_temporal decide INV-2026-1042 reject
```

**4b. Temporal's LangGraph plugin, every node an activity (build 6b):**
```bash
python -m graph_plugin_in_temporal.worker                    # terminal 2 (server from §4 in terminal 1)
python -m graph_plugin_in_temporal start INV-2026-1042       # then open http://localhost:8233 → Timeline
python -m graph_plugin_in_temporal status INV-2026-1042
python -m graph_plugin_in_temporal decide INV-2026-1042 reject
```

**5. The turn-engine experiment (old fixed turn vs turn graph):**
```bash
python -m experiments.turn_engine --offline --runs 10        # fake model: free, exact engine overhead
python -m experiments.turn_engine                            # live: 3 invoices x 2 engines x 3 runs (~$1.5)
```

**6. The LangGraph build (local, no Temporal):**
```bash
python -m langgraph_agent start INV-2026-1042                # runs until done or human review
python -m langgraph_agent status INV-2026-1042
python -m langgraph_agent decide INV-2026-1042 investigate "Call the vendor on the number in the vendor master"
python -m langgraph_agent resume INV-2026-1042               # after a crash: continue from the last checkpoint
python -m langgraph_agent history INV-2026-1042              # every checkpoint
python -m langgraph_agent graph                              # write the graph as Mermaid (langgraph_agent/graph.mmd)
```

## Scenarios

| Invoice | Situation | Expected |
|---|---|---|
| `INV-2026-1001` Northwind | Everything matches | Agent recommends **approve**, so the workflow auto-approves |
| `INV-2026-1042` Acme | Bank account changed 3 days ago by email, no second approval (plus a look-alike email domain) | **Human review** |
| `INV-2026-1077` Globex | Invoice 20.7% over the PO, no amendment | **Human review** |

## Results (October 2026)

**Structured brief vs full history.** Gemini 3.5 Flash, provider pinned, temperature 0, 3 runs per cell, 18 runs, $1.08 ([table](experiments/results/context_strategy-2026-10-09-1620.md)):

| Strategy | Correct outcome | Total cost | Mean turns | Output tokens per run |
|---|---|---|---|---|
| structured | **9/9** | $0.81 | 8.1 (6–11) | 7,733 |
| full_history | 6/9 | **$0.27** | 6.9 | 1,561 |

- Full history was 3x cheaper, and rejected `INV-2026-1042` in 3 of 3 runs instead of sending it to human review. The prompt reserves "reject" for clear cases; suspected fraud should go to a human. (The workflow still routes a reject to a human, so it was safe here.)
- Structured followed that rule every time, but billed about 4x the output tokens for the same visible output, and varied more in turns.
- Both strategies caught the look-alike remit email domain in every run.

**A stronger model, full history only.** Gemini 3.1 Pro: 8/9 correct, $0.28 for 9 runs, 4.7 turns. On `INV-2026-1042` it chose human review in 6 of 8 runs and reject in 2, at temperature 0. A smarter model reduces the drift; it doesn't remove it.

**Prompt caching, full history on `INV-2026-1042`.** Gemini cached nothing at these prompt sizes (about 2.5k tokens per call), with or without cache markers. Claude Sonnet 5.5 with `PROMPT_CACHE=1` reused 48–57% of prompt tokens: $0.036–0.041 per run vs $0.055 uncached, human review in 3 of 3 runs.

**Old fixed turn vs turn graph.** Same prompts, same model, structured brief. Offline with the fake model, both engines produced the identical investigation (same evidence, flags and recommendation) on all 3 invoices, 10 runs each; the graph adds about 1.6 ms per turn ([table](experiments/results/turn_engine-2026-10-10-1039-offline.md)). Live, Gemini 3.5 Flash, 18 runs, $1.71 ([table](experiments/results/turn_engine-2026-10-10-1040-gemini-3.5-flash.md)):

| Engine | Correct outcome | Mean turns | Time outside the LLM, per turn | Share of wall time waiting for the LLM | Total cost |
|---|---|---|---|---|---|
| fixed | 8/9 | 8.2 | 7.6 ms | 99.9% | $0.85 |
| graph | 9/9 | 8.7 | 14.2 ms | 99.7% | $0.87 |

- The engine doesn't change what the model sees, so it can't change what the model decides. The one miss (fixed rejected `INV-2026-1042` once) is the same model drift seen above, not the engine.
- The graph costs a few milliseconds per turn against 4–6 seconds of LLM time. What it buys: the flow is defined once, the path follows the model's choice (no tool phase when no data was asked for), and each node can be checkpointed, retried or run as its own activity.

**LangGraph inside one Temporal activity, real crash.** `INV-2026-1042`, Gemini 3.5 Flash. The worker was killed with `kill -9` after 3 turns (13 checkpoints), while turn 4 was waiting on the LLM. A new worker started 6 s later. Temporal noticed the missing heartbeat and retried the activity 22 s after the kill (attempt 2); the graph continued from checkpoint 13, ran turns 4–10 and stopped at human review. 10 turns, 10 LLM calls in state, $0.131. The call in flight at the kill was paid for and lost; nothing else ran twice. Temporal's history shows just one activity (scheduled, started as attempt 2, completed); the per-turn story lives in the LangGraph checkpoints.

**4 vs 6a vs 6b, one live run each (Oct 13 2026).** Gemini 3.5 Flash; 4 and 6b ran `INV-2026-1042`, 6a ran `INV-2026-1077`; all stopped at human review. Different invoices, so compare the shape, not the cost.

| | 4. Temporal alone | 6a. Graph in one activity | 6b. Plugin, node = activity |
|---|---|---|---|
| Activities in the Temporal UI | 13 (one per turn + events) | 1 (`run_graph`) | 33 (think, record, gather, decide… per turn) |
| Workflow history | 82 events, 93 KB | 10 events, 3 KB | 202 events, **676 KB** |
| Where the per-node story lives | Temporal | LangGraph SQLite checkpoints | Temporal |
| Retry unit after a failure | One turn | The whole graph, resumed from the last checkpoint | One node |
| Cost of the run | — | $0.08 (8 turns) | $0.09 (8 turns) |

The catch with 6b: the full graph state crosses every activity boundary twice, so history grows with turns × state size. Temporal warns at 10 MB per workflow history; long investigations would need `continue_as_new` or a slimmer state.

## Tests

```bash
pytest                    # offline, deterministic, free (~5 s): MCP server, loop, guardrail,
                          # workflow pause for 3 simulated days → human instruction → resume → reject
pytest -m live            # real model on all 3 scenarios (~$0.30). LIVE_RUNS=3 to check consistency
```
Live tests assert **outcomes and key evidence**, never the exact sequence of tool calls.

## Design decisions (and their trade-offs)

1. **Hand-written loop, no agent framework.** Every model call, context, tool choice and state change is visible and logged. Cost: you write the loop, retries and stop conditions yourself.
2. **MCP for tools.** The agent discovers tools at runtime; enterprise systems can change without touching the agent. Cost: a server process and a protocol to learn.
3. **Explicit state instead of chat history, by default.** Each turn the model gets a compact brief (facts, evidence, open questions, risk flags), not a growing transcript. On resume, the agent answered the reviewer's question from evidence it already had, in one turn. Measured trade-off: structured followed the decision rules every time, while full history cost a third as much but over-rejected. Both are available (`CONTEXT_STRATEGY`) so you can compare on your own cases.
4. **One Temporal activity per agent turn.** A crash loses at most one turn. The LLM never runs while waiting for a human. Cost: state is serialized between turns.
5. **The model recommends, code decides.** An "approve" with medium or high risk flags is overridden to human review by a guardrail, and logged.
6. **Anything but a clean approval goes to a human,** including the agent hitting its turn or cost budget.
7. **One MCP connection per turn.** The async client must open and close in the same task, and failures stay isolated. Cost: about 0.3 s per turn.
8. **Tests check outcomes, not paths.** A different but valid investigation still passes.

Observations worth knowing:
- **The clean invoice can cost the most.** With the structured brief, the agent kept checking policies to be sure (6 to 10 turns). "When is it done?" is a design decision, not a given.
- **None of the in-turn safety rules fired** in 46 real turns (nudge, call cap, dedupe, conclude-with-data, guardrail). They're insurance, and the offline tests force each one with a fake model.
- **The guardrail only sees flags the model raised.** "Code decides" is only as strong as the checks code runs itself. Deterministic policy checks are the next step.
- **Known gaps:** near-duplicate risk flags and open questions (exact-text dedupe), and facts that cite evidence without being checked against it.
