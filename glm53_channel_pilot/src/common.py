"""Shared config, credentials, cost accounting and JSONL helpers."""
import csv
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
RUNS = ROOT / CFG["paths"]["runs"]
RESULTS = ROOT / CFG["paths"]["results"]
COST_LOG = RESULTS / "cost_log.csv"
COST_FIELDS = ["ts", "stage", "call_id", "model", "prompt_tokens", "cached_tokens",
               "completion_tokens", "reasoning_tokens", "cost_usd", "openrouter_cost"]


def load_dotenv():
    """Read KEY=VALUE lines from .env (gitignored) without overriding the environment."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_dotenv()


def api_key(env_name):
    key = os.environ.get(env_name)
    if not key:
        raise SystemExit(f"{env_name} is not set (environment or .env)")
    return key


def budget():
    return float(os.environ.get("BUDGET_USD", CFG["budget_usd_default"]))


def call_cost(usage, pricing):
    """USD cost from a usage block, using the configured per-million prices."""
    if not usage:
        return 0.0
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    out = usage.get("completion_tokens") or 0
    p = pricing
    return ((prompt - cached) * p["input"] + cached * p.get("cached_input", p["input"])
            + out * p["output"]) / 1e6


def log_cost(stage, call_id, model, usage, cost):
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not COST_LOG.exists()
    usage = usage or {}
    with open(COST_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, COST_FIELDS)
        if new:
            w.writeheader()
        from datetime import datetime, timezone
        w.writerow({
            "ts": datetime.now(timezone.utc).isoformat(), "stage": stage, "call_id": call_id,
            "model": model, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost_usd": round(cost, 6), "openrouter_cost": usage.get("cost"),
        })


def spent_so_far():
    """Total USD across every logged call (subject model and judge)."""
    if not COST_LOG.exists():
        return 0.0
    with open(COST_LOG) as f:
        return sum(float(r["cost_usd"] or 0) for r in csv.DictReader(f))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
