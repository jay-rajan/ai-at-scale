"""
Ambiguous classification experiment: BUY / HOLD / SELL on made-up companies.

Every company below is FICTIONAL. This is a demo of model behaviour, not advice.

Each company is classified N times at temperature 0 with two system prompts:
  * vague:  "You are an equity analyst. Answer BUY, HOLD or SELL."
  * middle: names the factors to weigh, but no scores, thresholds or examples
  * clear:  a scoring rubric, a tie-break rule and worked examples

The answer is a single word, so the very first token IS the decision. When the
evidence is balanced, BUY/HOLD/SELL sit close together and tiny serving-side
differences (batching, routing) can flip it. A clear rubric widens the gap.

Run:  python3 03_stock_classification.py 20 --pin --model anthropic/claude-haiku-4.5
      python3 03_stock_classification.py --compare
Key:  OPENROUTER_API_KEY in .env next to this file
"""
import json, os, sys, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
for line in (HERE / ".env").read_text().splitlines() if (HERE / ".env").exists() else []:
    k, sep, v = line.partition("=")
    if sep and k.strip() and not k.lstrip().startswith("#"):
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY")
args = sys.argv[1:]
MODEL = args[args.index("--model") + 1] if "--model" in args else "anthropic/claude-haiku-4.5"
PIN = "--pin" in args
RESULTS = HERE / "results"
LABELS = ("BUY", "HOLD", "SELL")

# --- Fictional companies: 2 clear buys, 2 clear sells, 4 knife-edge cases -----
COMPANIES = {
    "NOVA (clear buy)": dict(revenue_growth="+32% YoY", operating_margin="24%, up from 19%",
        pe_ratio="18 (sector avg 25)", debt_to_equity="0.2", free_cash_flow="strongly positive, growing",
        insider_activity="CEO bought $2M of shares", guidance="raised for the 3rd quarter in a row",
        analyst_views="11 buy, 1 hold, 0 sell", news="won a 5-year government contract"),
    "ORBIT (clear buy)": dict(revenue_growth="+21% YoY", operating_margin="31%, stable",
        pe_ratio="20 (sector avg 24)", debt_to_equity="0.1", free_cash_flow="positive, record high",
        insider_activity="no sales; two directors bought", guidance="raised",
        analyst_views="9 buy, 2 hold, 0 sell", news="launched product with 40% gross margin"),
    "DUSKWARE (clear sell)": dict(revenue_growth="-18% YoY", operating_margin="-12%, down from 3%",
        pe_ratio="n/a (losses)", debt_to_equity="3.4", free_cash_flow="negative, worsening",
        insider_activity="CFO resigned; heavy insider selling", guidance="withdrawn",
        analyst_views="0 buy, 2 hold, 8 sell", news="auditor raised going-concern doubts"),
    "RUSTLINE (clear sell)": dict(revenue_growth="-9% YoY", operating_margin="2%, down from 11%",
        pe_ratio="61 (sector avg 15)", debt_to_equity="2.8", free_cash_flow="negative",
        insider_activity="CEO sold 40% of holdings", guidance="cut sharply",
        analyst_views="1 buy, 3 hold, 7 sell", news="lost its largest customer"),
    "HALCYON (knife-edge)": dict(revenue_growth="+19% YoY", operating_margin="14%, down from 18%",
        pe_ratio="38 (sector avg 25)", debt_to_equity="1.1", free_cash_flow="slightly positive",
        insider_activity="CEO sold $3M under a pre-planned schedule", guidance="maintained",
        analyst_views="5 buy, 5 hold, 4 sell", news="new product delayed one quarter; strong pre-orders"),
    "MERIDIAN (knife-edge)": dict(revenue_growth="+4% YoY", operating_margin="22%, stable",
        pe_ratio="12 (sector avg 16)", debt_to_equity="1.9", free_cash_flow="positive but falling",
        insider_activity="mixed: one director bought, one sold", guidance="slightly lowered",
        analyst_views="4 buy, 6 hold, 4 sell", news="activist investor took a 6% stake"),
    "TESSERA (knife-edge)": dict(revenue_growth="+55% YoY", operating_margin="-8%, improving from -20%",
        pe_ratio="n/a (losses); price/sales 14 (sector avg 6)", debt_to_equity="0.4",
        free_cash_flow="negative, improving", insider_activity="founder sold 5%",
        guidance="raised revenue, lowered profit", analyst_views="6 buy, 4 hold, 5 sell",
        news="regulator opened inquiry; also signed a major partnership"),
    "QUILL (knife-edge)": dict(revenue_growth="+1% YoY", operating_margin="9%, up from 7%",
        pe_ratio="15 (sector avg 15)", debt_to_equity="1.0", free_cash_flow="flat",
        insider_activity="none", guidance="maintained", analyst_views="4 buy, 7 hold, 4 sell",
        news="announced restructuring: 10% layoffs, new CEO from a competitor"),
}

VAGUE = "You are an equity analyst. Read the company data and answer with exactly one word: BUY, HOLD or SELL."

# In between: tells the model WHAT to weigh, but not HOW. No scores, thresholds or examples.
MIDDLE = ("You are an equity analyst. Weigh revenue growth, margin trend, valuation versus the sector, "
          "debt, free cash flow, insider activity and guidance, and decide whether this stock is a "
          "BUY, HOLD or SELL. Answer with exactly one word: BUY, HOLD or SELL.")

CLEAR = """You classify FICTIONAL companies as BUY, HOLD or SELL using this rubric only.

Score each signal +1 (positive), 0 (neutral or mixed) or -1 (negative):
1. Revenue growth: > +15% = +1, < 0% = -1, else 0
2. Operating margin trend: rising = +1, falling = -1, stable = 0
3. Valuation: P/E (or P/S) below sector average = +1, more than 30% above = -1, else 0
4. Debt-to-equity: < 0.5 = +1, > 1.5 = -1, else 0
5. Free cash flow: positive and growing = +1, negative = -1, else 0
6. Insiders: net buying = +1, net selling (any planned or unplanned sale with no buying) = -1, none or mixed = 0
7. Guidance: raised = +1, lowered/cut/withdrawn = -1, maintained or mixed = 0

Total = sum of the 7 scores.
Total >= +2 -> BUY. Total <= -2 -> SELL. Otherwise -> HOLD.
Ignore analyst views and news; they are not in the rubric.

Examples:
- growth +20%, margin rising, P/E below sector, D/E 0.3, FCF growing, insider buying, guidance raised -> total +7 -> BUY
- growth -5%, margin falling, P/E 50 vs 20, D/E 2.0, FCF negative, insider selling, guidance cut -> total -7 -> SELL
- growth +10%, margin stable, P/E equal to sector, D/E 1.0, FCF flat, no insider activity, guidance maintained -> total 0 -> HOLD

Show the 7 scores briefly, then end with a final line exactly like: ANSWER: BUY (or HOLD or SELL)."""

PROMPTS = {"vague": VAGUE, "middle": MIDDLE, "clear": CLEAR}
MAX_TOKENS = {"vague": 5, "middle": 5, "clear": 800}   # rubric needs room to show its scores

def company_text(name, d):
    rows = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in d.items())
    return f"Company: {name.split(' (')[0]} (fictional)\n{rows}\n\nYour answer:"

pinned_provider = None

def ask(system, user, max_tokens=5):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if pinned_provider:
        body["provider"] = {"order": [pinned_provider], "allow_fallbacks": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    format_retries = 0
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            raw = (data["choices"][0]["message"].get("content") or "").strip()
            text = raw.upper()
            if "ANSWER:" in text:
                text = text.rsplit("ANSWER:", 1)[1]
            label = next((l for l in LABELS if l in text), None)
            # Format check: no readable label (e.g. reply cut off) -> retry, like you would in production
            if label is None and attempt < 3:
                format_retries += 1
                continue
            return label or f"?{text[:10]}", data.get("provider", "?"), raw, format_retries
        except Exception as e:
            if attempt == 3:
                return "ERROR", str(e)[:60], "", format_retries
            time.sleep(2 * (attempt + 1))

def run(n):
    global pinned_provider
    if PIN:
        _, pinned_provider, _, _ = ask("Reply OK.", "Say OK.")
        print(f"Model: {MODEL}   pinned provider: {pinned_provider}   runs per company: {n}")
    jobs = [(p, c) for p in PROMPTS for c in COMPANIES for _ in range(n)]
    # Shuffle-free but interleaved: all requests go out concurrently, mixing with live traffic.
    with ThreadPoolExecutor(max_workers=12) as pool:
        out = list(pool.map(lambda j: ask(PROMPTS[j[0]], company_text(j[1], COMPANIES[j[1]]), MAX_TOKENS[j[0]]), jobs))
    table = {p: {c: Counter() for c in COMPANIES} for p in PROMPTS}
    providers = Counter()
    for (p, c), (label, prov, _, _) in zip(jobs, out):
        table[p][c][label] += 1
        providers[prov] += 1

    for p in PROMPTS:
        print(f"\n{p.upper()} system prompt")
        print(f"  {'company':<24}{'BUY':>5}{'HOLD':>6}{'SELL':>6}   agreement")
        for c, cnt in table[p].items():
            agree = cnt.most_common(1)[0][1] / n
            flag = "   <-- flips" if len(cnt) > 1 else ""
            print(f"  {c:<24}{cnt['BUY']:>5}{cnt['HOLD']:>6}{cnt['SELL']:>6}   {agree:6.0%}{flag}")
    print(f"\nProviders used: {dict(providers)}")

    # Where did the flips come from? Same provider (batching) or different providers (hardware/setup)?
    calls = [{"prompt": p, "company": c, "label": l, "provider": prov, "raw": raw, "format_retries": fr}
             for (p, c), (l, prov, raw, fr) in zip(jobs, out)]
    retried = [x for x in calls if x["format_retries"]]
    print(f"\nFormat check: {len(retried)} call(s) needed a retry for an unreadable reply "
          f"({dict(Counter(x['provider'] for x in retried))})")
    for p in PROMPTS:
        for c, cnt in table[p].items():
            if len(cnt) > 1:
                by_prov = {}
                for x in calls:
                    if x["prompt"] == p and x["company"] == c:
                        by_prov.setdefault(x["provider"], Counter())[x["label"]] += 1
                print(f"  flip detail: {p} / {c}")
                for prov, pc in sorted(by_prov.items()):
                    mark = "  <-- flips WITHIN one provider" if len(pc) > 1 else ""
                    print(f"      {prov:<14} {dict(pc)}{mark}")

    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M}_stocks_{MODEL.split('/')[-1]}{'_pinned' if PIN else ''}.json"
    f.write_text(json.dumps({"model": MODEL, "n": n, "pinned": pinned_provider, "time": time.ctime(),
        "table": {p: {c: dict(v) for c, v in t.items()} for p, t in table.items()}, "calls": calls}, indent=2))
    print(f"Saved to {f}")

def compare():
    print("Saved stock runs (companies with >1 distinct label):")
    for f in sorted(RESULTS.glob("*_stocks_*.json")):
        d = json.loads(f.read_text())
        flips = {p: sum(len(v) > 1 for v in t.values()) for p, t in d["table"].items()}
        print(f"  {f.stem:<48} " + "  ".join(f"{k} flips: {v}" for k, v in flips.items()))

if __name__ == "__main__":
    if "--compare" in args and not any(a.isdigit() for a in args):
        compare(); sys.exit()
    if not KEY:
        sys.exit("Add your key to .env first:  OPENROUTER_API_KEY=sk-or-...")
    run(int(next((a for a in args if a.isdigit()), 20)))
