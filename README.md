# AI at Scale

Small, runnable experiments on what happens when AI systems meet real-world scale.
Companion code for the **AI at Scale** series on [The AI Switch](https://theaiswitch.substack.com/).

| # | Experiment | Question |
|---|---|---|
| 01 | [Batching](01-batching/) | Can the same prompt at temperature 0 give opposite answers? |
| 02 | [Invoice Investigation Agent](02-invoice-agent/) | How do you build an agent an enterprise can trust: tools (MCP), explicit state, durable workflows (Temporal), and a human in the loop? Article: [Do you even need an AI agent?](https://theaiswitch.substack.com/p/do-you-even-need-an-ai-agent-rules) |
| 03 | [Entity resolution at scale](03-entity-resolution/) | How many real people are in 5 million records, and how little of the work needs AI? |

## Setup

You need **Python 3.10 or newer** (`python3 --version`).

**macOS / Linux**
```bash
git clone https://github.com/jay-rajan/ai-at-scale.git
cd ai-at-scale
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/jay-rajan/ai-at-scale.git
cd ai-at-scale
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Your prompt should now start with `(.venv)`. Run `deactivate` to leave the environment, and activate it again (`source .venv/bin/activate`) whenever you come back.

Then follow the README inside each experiment folder.

**Troubleshooting**
- `python3: command not found` (macOS): install Python from [python.org](https://www.python.org/downloads/) or `brew install python`.
- PowerShell blocks `Activate.ps1`: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- `ModuleNotFoundError: numpy`: the environment isn't active. Activate it and rerun `pip install -r requirements.txt`.

By Jayadev Rajan. MIT licensed.
