"""Run GLM-5.3 trials. Append-only and resumable: trials already in the run file are skipped.

Every request offers one stub tool, write_file. The stub never touches the filesystem or runs
anything: it records the arguments and returns a fixed success message. A trial loops until the
model stops calling tools (at most MAX_TOOL_ROUNDS rounds).

    .venv/bin/python src/run.py stage0          # 3 API-check trials -> runs/stage0_v2.jsonl
    .venv/bin/python src/run.py smoke           # 20 trials across sources/conditions -> runs/smoke_v2.jsonl
    .venv/bin/python src/run.py full            # 2,160 pilot trials -> runs/pilot_v2.jsonl
"""
import argparse
import asyncio
import json
import random
import time
from datetime import datetime, timezone

import httpx
import yaml

from common import (CFG, ROOT, RUNS, api_key, append_jsonl, budget, call_cost, log_cost,
                    read_jsonl, spent_so_far)

API = CFG["api"]
SYSTEM = (ROOT / "prompts/system.txt").read_text()
COND = yaml.safe_load(open(ROOT / CFG["conditions_file"]))
VERSION = CFG["version"]
_items_path = ROOT / CFG["paths"]["items"]
ITEMS = [json.loads(l) for l in open(_items_path)] if _items_path.exists() else []
TOOLS = json.load(open(ROOT / "prompts/tools.json"))
MAX_TOOL_ROUNDS = 4


def stub_tool(call):
    """Fake tool execution: record the call, return a fixed message. No side effects."""
    fn = call.get("function") or {}
    if fn.get("name") != "write_file":
        return f"Error: unknown tool {fn.get('name')!r}."
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        return "Error: arguments are not valid JSON."
    if not isinstance(args, dict) or "path" not in args or "content" not in args:
        return "Error: write_file requires 'path' and 'content'."
    return f"Wrote {len(str(args['content']).encode())} bytes to {args['path']}."


FORMAT_KEY = {"mc": "format_mc", "free": "format_free", "rating": "format_rating", "reply": "format_reply"}


def build_messages(item, cond_key):
    """Item turns with the format line and condition instruction appended.

    v2: our system prompt first (byte-identical), instruction appended to the last user turn.
    v3: the item's own system prompt (or ours), instruction appended to the system prompt.
    """
    sep = "\n" if VERSION == "v3" else " "
    instr = f"{COND[FORMAT_KEY[item['type']]]}{sep}{COND[cond_key]}"
    msgs = [dict(m) for m in item["messages"]]
    if msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM}] + msgs
    assert msgs[-1]["role"] == "user"
    if CFG.get("instruction_location") == "system":
        msgs[0]["content"] += f"\n\n{instr}"
    else:
        msgs[-1]["content"] += f"\n\n{instr}"
    return msgs


def all_calls():
    calls = []
    for it in ITEMS:
        for cond in CFG["conditions"]:
            for para in CFG["paraphrases"]:
                for s in range(CFG["samples_per_cell"]):
                    calls.append({"call_id": f"{it['item_id']}|{cond}{para}|s{s}", "item": it,
                                  "condition": cond, "paraphrase": para, "sample": s,
                                  "effort": API["reasoning_effort"]})
    return calls


def smoke_calls(n=20):
    """n calls spread round-robin over sources and over the six condition paraphrases."""
    rng = random.Random(CFG["seed"])
    sources = sorted({it["source"] for it in ITEMS})
    cps = [(c, p) for c in CFG["conditions"] for p in CFG["paraphrases"]]
    calls = []
    for i in range(n):
        src = sources[i % len(sources)]
        cond, para = cps[i % len(cps)]
        it = rng.choice([x for x in ITEMS if x["source"] == src])
        calls.append({"call_id": f"{it['item_id']}|{cond}{para}|s0", "item": it,
                      "condition": cond, "paraphrase": para, "sample": 0,
                      "effort": API["reasoning_effort"]})
    return calls


def stage0_calls():
    """One item in each condition (paraphrase 1) to check tool calling and the reasoning field."""
    it = ITEMS[0]
    return [{"item": it, "sample": 0, "call_id": f"{it['item_id']}|{c}1|s0", "condition": c,
             "paraphrase": 1, "effort": API["reasoning_effort"]} for c in CFG["conditions"]]


class Runner:
    def __init__(self, stage, out_path):
        self.stage, self.out = stage, out_path
        self.key = api_key(API["key_env"])
        self.sem = asyncio.Semaphore(API["concurrency"])
        self.spent = spent_so_far()
        self.budget = budget()
        self.stopped = False

    def body(self, messages, effort):
        return {
            "model": API["model"],
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "reasoning": {"effort": effort},
            "max_tokens": API["max_tokens"],
            "provider": API["provider"],
            "usage": {"include": True},
        }

    async def post(self, client, body):
        delay = 2.0
        for attempt in range(API["max_retries"]):
            try:
                r = await client.post(f"{API['base_url']}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
                data = r.json()
                if r.status_code != 200 or "error" in data:
                    code = (data.get("error") or {}).get("code", r.status_code)
                    if code in (429, 502, 503, 504) or (isinstance(code, int) and code >= 500):
                        raise httpx.HTTPStatusError(f"error {code}", request=r.request, response=r)
                    return data, attempt  # non-retryable: keep the error in the record
                return data, attempt
            except (httpx.HTTPStatusError, httpx.TransportError, json.JSONDecodeError) as e:
                if attempt == API["max_retries"] - 1:
                    return {"error": {"message": f"gave up: {e!r}"}}, attempt
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 120)

    async def one(self, client, call):
        async with self.sem:
            if self.stopped or self.spent >= self.budget:
                self.stopped = True
                return
            messages = build_messages(call["item"], f"{call['condition']}{call['paraphrase']}")
            t0 = datetime.now(timezone.utc).isoformat()
            start = time.monotonic()
            turns, cost, retries, error = [], 0.0, 0, None
            for _ in range(MAX_TOOL_ROUNDS + 1):
                body = self.body(messages, call["effort"])
                data, n_retry = await self.post(client, body)
                retries += n_retry
                usage = data.get("usage") or {}
                c = call_cost(usage, CFG["pricing_per_million"])
                cost += c
                log_cost(self.stage, call["call_id"], API["model"], usage, c)
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                tool_calls = msg.get("tool_calls") or []
                turn = {"request_messages": list(messages), "reasoning": msg.get("reasoning"),
                        "reasoning_details": msg.get("reasoning_details"),
                        "content": msg.get("content"), "tool_calls": tool_calls,
                        "finish_reason": choice.get("finish_reason"),
                        "native_finish_reason": choice.get("native_finish_reason"),
                        "usage": usage, "provider": data.get("provider"),
                        "response_id": data.get("id"), "error": data.get("error"), "raw": data}
                if tool_calls:
                    turn["tool_results"] = [stub_tool(tc) for tc in tool_calls]
                turns.append(turn)
                if data.get("error"):
                    error = data["error"]
                    break
                if not tool_calls:
                    break
                assistant = {"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": tool_calls}
                if msg.get("reasoning_details"):
                    assistant["reasoning_details"] = msg["reasoning_details"]
                messages = messages + [assistant] + [
                    {"role": "tool", "tool_call_id": tc.get("id"), "content": res}
                    for tc, res in zip(tool_calls, turn["tool_results"])]
            self.spent += cost
            row = {
                "call_id": call["call_id"], "stage": self.stage,
                "item_id": call["item"]["item_id"], "source": call["item"]["source"],
                "condition": call["condition"], "paraphrase": call["paraphrase"],
                "sample": call["sample"], "effort": call["effort"],
                "tools": TOOLS, "temperature": "provider default (not sent)",
                "turns": turns, "n_turns": len(turns), "hit_round_limit": bool(turns[-1]["tool_calls"]),
                "cost_usd": cost, "error": error, "retries": retries,
                "t_start": t0, "latency_s": round(time.monotonic() - start, 2),
            }
            append_jsonl(self.out, row)
            out_toks = sum((t["usage"] or {}).get("completion_tokens") or 0 for t in turns)
            ntc = sum(len(t["tool_calls"]) for t in turns)
            flag = "ERR" if error else turns[-1]["finish_reason"]
            print(f"[{self.stage}] {call['call_id']:<55} {flag:<7} turns={len(turns)} tool_calls={ntc} "
                  f"out={out_toks} ${cost:.4f}  total=${self.spent:.2f}", flush=True)

    async def run(self, calls):
        done = {r["call_id"] for r in read_jsonl(self.out) if not r.get("error")}
        todo = [c for c in calls if c["call_id"] not in done]
        # Random order so a budget stop leaves a balanced partial sample.
        random.Random(CFG["seed"]).shuffle(todo)
        print(f"{len(calls)} calls, {len(calls) - len(todo)} already done, {len(todo)} to run; "
              f"spent so far ${self.spent:.2f} of ${self.budget:.2f}")
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
            await asyncio.gather(*(self.one(client, c) for c in todo))
        if self.stopped:
            print(f"STOPPED: budget ${self.budget:.2f} reached (spent ${self.spent:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage0", "smoke", "full"])
    ap.add_argument("--n", type=int, default=20, help="smoke-test size")
    ap.add_argument("--limit", type=int, help="run at most this many pending calls (full)")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    if a.stage == "stage0":
        calls, out = stage0_calls(), RUNS / f"stage0_{VERSION}.jsonl"
    elif a.stage == "smoke":
        calls, out = smoke_calls(a.n), RUNS / f"smoke_{VERSION}.jsonl"
    else:
        calls, out = all_calls(), RUNS / f"pilot_{VERSION}.jsonl"
        if a.limit:
            done = {r["call_id"] for r in read_jsonl(out) if not r.get("error")}
            calls = [c for c in calls if c["call_id"] not in done][: a.limit]
    asyncio.run(Runner(a.stage, out).run(calls))


if __name__ == "__main__":
    main()
