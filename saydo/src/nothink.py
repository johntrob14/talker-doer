"""E9b: does the talker's rule still control the doer when deliberation is removed?

Same 8 writing tasks and em-dash rule as E9, single turn.
  think=none  Together raw completions, GLM-5.3 template, prefill "</think>" (empty reasoning)
  think=low   Z.AI chat completions, reasoning effort low
Conditions: none (no rule) | system (rule in system prompt) | commit (user rule + a fixed confirmation
turn written by GLM in E9 pilot, then the task).

    python src/nothink.py --samples 2   # -> runs/e9b_nothink.jsonl
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import jinja2

sys.path.insert(0, str(Path(__file__).parent))
from emdash import RULE_MSG, SYSTEM, SYSTEM_RULE, TASKS, measure  # noqa: E402
from llm import MODEL, RAW_PROVIDER, ROOT, BudgetExceeded, _post, append_jsonl, chat, log_cost, read_jsonl, spent, BUDGET  # noqa: E402

TEMPLATE = ROOT.parent / "glm53_channel_pilot/data/raw/glm53_chat_template.jinja"
CONFIRM = ("Confirmed. I won't use em dashes in anything I write for you in this conversation. I'll use commas, "
           "colons, parentheses or separate sentences instead.")


def render(messages):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, ensure_ascii=False, **k: json.dumps(v, ensure_ascii=ensure_ascii)
    return env.from_string(TEMPLATE.read_text()).render(messages=messages, tools=None, reasoning_effort="high",
                                                        add_generation_prompt=True)


def messages(cond, task):
    if cond == "system":
        return [{"role": "system", "content": SYSTEM_RULE}, {"role": "user", "content": task}]
    msgs = [{"role": "system", "content": SYSTEM}]
    if cond == "commit":
        msgs += [{"role": "user", "content": RULE_MSG}, {"role": "assistant", "content": CONFIRM}]
    return msgs + [{"role": "user", "content": task}]


async def one(client, sem, out, think, cond, i, s):
    cid = f"{think}|{cond}|t{i}|s{s}"
    async with sem:
        msgs = messages(cond, TASKS[i])
        try:
            if think == "none":
                if spent() >= BUDGET:
                    raise BudgetExceeded
                prompt = render(msgs)
                assert prompt.endswith("<|assistant|><think>")
                body = {"model": MODEL, "prompt": prompt + "</think>", "max_tokens": 700, "temperature": 1.0,
                        "provider": RAW_PROVIDER, "usage": {"include": True}}
                d = await _post(client, "https://openrouter.ai/api/v1/completions", body)
                log_cost("e9b-nothink", cid, d.get("usage"))
                ch = (d.get("choices") or [{}])[0]
                content, reasoning = ch.get("text") or "", ch.get("reasoning") or ""
            else:
                d = await chat(client, "e9b-nothink", cid, msgs, effort="low")
                m = (d.get("choices") or [{}])[0].get("message") or {}
                content, reasoning = m.get("content") or "", m.get("reasoning") or ""
        except BudgetExceeded as e:
            print("BUDGET STOP", e)
            return
        append_jsonl(out, {"call_id": cid, "think": think, "cond": cond, "task": TASKS[i], "sample": s,
                           "messages": msgs, "content": content, "reasoning": reasoning, "error": d.get("error"),
                           "provider": d.get("provider"), **measure(content)})
        print(f"{cid:<26} em={measure(content)['em']} words={measure(content)['words']} reasoning_len={len(reasoning)}",
              flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--thinks", default="none,low")
    a = ap.parse_args()
    out = ROOT / "runs/e9b_nothink.jsonl"
    done = {r["call_id"] for r in read_jsonl(out) if not r.get("error")}
    sem = asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30)) as client:
        await asyncio.gather(*(one(client, sem, out, th, c, i, s) for s in range(a.samples) for th in a.thinks.split(",")
                               for c in ("none", "system", "commit") for i in range(len(TASKS))
                               if f"{th}|{c}|t{i}|s{s}" not in done))
    import pandas as pd
    df = pd.DataFrame(read_jsonl(out))
    df["has_em"] = df.em > 0
    print(df.groupby(["think", "cond"]).agg(n=("has_em", "size"), with_em=("has_em", "sum"), em_total=("em", "sum"),
                                            en_total=("en", "sum"), errors=("error", lambda x: x.notna().sum())))
    print(f"total new spend ${spent():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
