"""
Live experiment via OpenRouter: does the same prompt at temperature 0 always give
the same answer?

Sends an AMBIGUOUS prompt and a CLEAR prompt N times each, concurrently, and counts
how many different answers come back. Default model is Qwen3-235B-A22B-2507, the
model Thinking Machines used in "Defeating Nondeterminism in LLM Inference".

Setup:  put OPENROUTER_API_KEY=... in .env next to this file (no pip installs needed)
Run:    python3 02_live_experiment.py                  # 30 runs per prompt
        python3 02_live_experiment.py 50 --pin         # pin one provider (isolates batching)
        python3 02_live_experiment.py --compare        # compare all saved runs
        python3 02_live_experiment.py 30 --model anthropic/claude-haiku-4.5
        python3 02_live_experiment.py 30 --long          # Thinking Machines prompt, 1000-token answers
Cost:   around a cent for N=30 on Qwen.

Two sources of variation:
  * batching on the provider's GPUs (the Thinking Machines effect)
  * OpenRouter routing your requests to different providers/hardware
--pin keeps everything on one provider so what's left is mostly batching.
"""
import json, os, sys, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"
def load_env(path=Path(__file__).with_name(".env")):
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and not k.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()
KEY = os.environ.get("OPENROUTER_API_KEY")
args = sys.argv[1:]
MODEL = args[args.index("--model") + 1] if "--model" in args else "qwen/qwen3-235b-a22b-2507"
PIN = "--pin" in args
RESULTS = Path(__file__).with_name("results")

PROMPTS = {
    "ambiguous": "Which city best represents the future? Answer in one short sentence.",
    "clear": (
        "Pick the city that best represents the future, using exactly this rule: "
        "the city with the most fully driverless taxis operating commercially. "
        "Reply in the format 'City: <name>'. Nothing else.\n\n"
        "Example of the format:\nCity: Example City"
    ),
}

# --long: the Thinking Machines setup. Long answers give batch noise more chances to flip a token.
if "--long" in args:
    PROMPTS = {"feynman": "Tell me about Richard Feynman."}
MAX_TOKENS = 1000 if "--long" in args else 120

pinned_provider = None

def ask(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    if pinned_provider:
        body["provider"] = {"order": [pinned_provider], "allow_fallbacks": False}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            text = (data["choices"][0]["message"].get("content") or "").strip()
            return text, data.get("provider", "?")
        except Exception as e:
            if attempt == 2:
                return f"[error: {e}]", "error"
            time.sleep(2 * (attempt + 1))

def run(n):
    global pinned_provider
    if PIN:
        _, pinned_provider = ask("Say OK.")
        print(f"Pinned to provider: {pinned_provider}")
    results = {}
    for name, prompt in PROMPTS.items():
        with ThreadPoolExecutor(max_workers=10) as pool:   # concurrent: shares batches with live traffic
            out = list(pool.map(ask, [prompt] * n))
        results[name] = [{"text": t, "provider": p} for t, p in out]
        counts = Counter(t for t, _ in out)
        providers = Counter(p for _, p in out)
        print(f"\n{name.upper()}: {len(counts)} different answer(s) out of {n}   providers: {dict(providers)}")
        for text, k in counts.most_common(6):
            print(f"  {k:3d} x  {text[:100]}")
        if len(counts) > 1:
            texts = [t for t, _ in counts.most_common(2)]
            i = next((j for j, (x, y) in enumerate(zip(texts[0], texts[1])) if x != y), min(map(len, texts)))
            print(f"  top two answers split at character {i}: ...{texts[0][max(0,i-40):i+30]!r}")
            print(f"                                    vs ...{texts[1][max(0,i-40):i+30]!r}")
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M}_{MODEL.split('/')[-1]}{'_long' if '--long' in args else ''}{'_pinned' if PIN else ''}.json"
    f.write_text(json.dumps({"model": MODEL, "n": n, "pinned": pinned_provider,
                             "time": time.ctime(), "results": results}, indent=2))
    print(f"\nSaved to {f}")

def compare():
    print("\nAll saved runs:")
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        line = " | ".join(f"{k}: {len({x['text'] for x in v})} unique" for k, v in d["results"].items())
        print(f"  {f.stem:<24} {d['model']:<28} {line}")

if __name__ == "__main__":
    if "--compare" in args and not any(a.isdigit() for a in args):
        compare(); sys.exit()
    if not KEY:
        sys.exit("Add your key to .env first:  OPENROUTER_API_KEY=sk-or-...")
    run(int(next((a for a in args if a.isdigit()), 30)))
    if "--compare" in args:
        compare()
