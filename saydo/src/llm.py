"""OpenRouter client with a per-project cost log and a hard budget stop. GLM-5.3 by default; SAYDO_MODEL and
SAYDO_PROVIDER select another model (round 5: moonshotai/kimi-k3 via moonshotai)."""
import asyncio
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
COST_LOG = ROOT / "results/cost_log.csv"
MODEL = os.environ.get("SAYDO_MODEL", "z-ai/glm-5.3")
CHAT_PROVIDER = {"order": [os.environ.get("SAYDO_PROVIDER", "z-ai")], "allow_fallbacks": False}
RAW_PROVIDER = {"order": ["together"], "allow_fallbacks": False}  # raw completions (prefill)
PRICE = {"input": 1.40, "cached_input": 0.14, "output": 4.40}     # USD per million, Z.AI listing
BUDGET = float(os.environ.get("SAYDO_BUDGET_USD", 14.5))          # new spend for this project only ($15 cap minus headroom for in-flight calls)

for envf in [ROOT.parent / ".env", ROOT.parent / "glm53_channel_pilot/.env"]:
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
KEY = os.environ["OPENROUTER_API_KEY"]


class BudgetExceeded(Exception):
    pass


def spent():
    if not COST_LOG.exists():
        return 0.0
    with open(COST_LOG) as f:
        return sum(float(r["cost_usd"] or 0) for r in csv.DictReader(f))


def cost_of(usage):
    if not usage:
        return 0.0
    if usage.get("cost") is not None:
        return float(usage["cost"])
    p = usage.get("prompt_tokens") or 0
    c = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    o = usage.get("completion_tokens") or 0
    return ((p - c) * PRICE["input"] + c * PRICE["cached_input"] + o * PRICE["output"]) / 1e6


def log_cost(stage, call_id, usage):
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not COST_LOG.exists()
    usage = usage or {}
    with open(COST_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "stage", "call_id", "prompt_tokens", "completion_tokens", "cost_usd"])
        w.writerow([datetime.now(timezone.utc).isoformat(), stage, call_id,
                    usage.get("prompt_tokens"), usage.get("completion_tokens"), round(cost_of(usage), 6)])


async def _post(client, url, body, retries=8):
    delay = 3.0
    for attempt in range(retries):
        try:
            r = await client.post(url, json=body, headers={"Authorization": f"Bearer {KEY}"})
            d = r.json()
            err = d.get("error") if isinstance(d, dict) else None
            code = (err or {}).get("code", r.status_code) if err else r.status_code
            if r.status_code == 200 and not err:
                return d
            if code in (429, 408) or (isinstance(code, int) and code >= 500):
                raise RuntimeError(f"retryable {code}: {err}")
            return d
        except (RuntimeError, httpx.TransportError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                return {"error": {"message": f"gave up: {e!r}"}}
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 90)


async def chat(client, stage, call_id, messages, tools=None, effort="high", max_tokens=12000, tool_choice="auto"):
    if spent() >= BUDGET:
        raise BudgetExceeded(f"spent ${spent():.2f} >= ${BUDGET}")
    body = {"model": MODEL, "messages": messages, "reasoning": {"effort": effort},
            "max_tokens": max_tokens, "provider": CHAT_PROVIDER, "usage": {"include": True}}
    if tools:
        body |= {"tools": tools, "tool_choice": tool_choice}
    d = await _post(client, "https://openrouter.ai/api/v1/chat/completions", body)
    log_cost(stage, call_id, d.get("usage"))
    return d


async def raw(client, stage, call_id, prompt, max_tokens=4000):
    if spent() >= BUDGET:
        raise BudgetExceeded(f"spent ${spent():.2f} >= ${BUDGET}")
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "provider": RAW_PROVIDER, "usage": {"include": True}}
    d = await _post(client, "https://openrouter.ai/api/v1/completions", body)
    log_cost(stage, call_id, d.get("usage"))
    return d


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in open(path)] if path.exists() else []
