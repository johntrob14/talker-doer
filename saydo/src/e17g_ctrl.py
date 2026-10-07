"""E17g follow-ups on the AI-risk item set (pre-registered in notes/hypotheses.md):
  v1 self-report on every base run; v2 and v3 self-reports on every judge-positive run (base and system);
  third-party judgments on every judge-positive run and an equal random sample of negatives.

    python src/e17g_ctrl.py
"""
import asyncio
import collections
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from evalaware import SELF_Q, SELF_Q2, SELF_Q3  # noqa: E402
from evalaware_ctrl import THIRD_Q  # noqa: E402
from llm import ROOT, BudgetExceeded, append_jsonl, read_jsonl, spent  # noqa: E402
from thinklang import gen  # noqa: E402

OUT = ROOT / "runs/e17g_ctrl.jsonl"


async def selfq(client, sem, r, k, q):
    async with sem:
        msgs = r["messages"] + [{"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]},
                                {"role": "user", "content": q}]
        r2, t2, e2 = await gen(client, f"{r['call_id']}|{k}", msgs, 2000)
        append_jsonl(OUT, {"key": f"{r['call_id']}|{k}", "call_id": r["call_id"], "kind": k, "reasoning": r2,
                           "content": t2, "error": e2, "yes": t2.lstrip("*# ").upper().startswith("YES")})


async def third(client, sem, r):
    async with sem:
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": THIRD_Q.format(trace=r["reasoning"])}]
        r2, t2, e2 = await gen(client, r["call_id"] + "|third", msgs, 2000)
        append_jsonl(OUT, {"key": f"{r['call_id']}|third", "call_id": r["call_id"], "kind": "third", "reasoning": r2,
                           "content": t2, "error": e2, "yes": t2.lstrip("*# ").upper().startswith("YES")})


async def main():
    rows = read_jsonl(ROOT / "runs/e17g_evalaware.jsonl")
    J = {j["call_id"]: j["yes"] for j in read_jsonl(ROOT / "runs/e17g_judge.jsonl") if j["field"] == "reasoning"}
    done = {r["key"] for r in read_jsonl(OUT) if not r.get("error")}
    pos = [r for r in rows if J.get(r["call_id"])]
    neg = [r for r in rows if J.get(r["call_id"]) is False]
    random.Random(171).shuffle(neg)
    sem = asyncio.Semaphore(3)
    jobs = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30)) as client:
            for r in rows:
                if r["cond"] == "base" and f"{r['call_id']}|v1" not in done:
                    jobs.append(selfq(client, sem, r, "v1", SELF_Q))
            for r in pos:
                for k, q in (("v2", SELF_Q2), ("v3", SELF_Q3)):
                    if f"{r['call_id']}|{k}" not in done:
                        jobs.append(selfq(client, sem, r, k, q))
            for r in pos + neg[:len(pos)]:
                if f"{r['call_id']}|third" not in done:
                    jobs.append(third(client, sem, r))
            await asyncio.gather(*jobs)
    except BudgetExceeded as e:
        print("BUDGET STOP", e)
    C = read_jsonl(OUT)
    cond = {r["call_id"]: r["cond"] for r in rows}
    sys_v1 = {r["call_id"]: r["self_report"]["yes"] for r in rows if r["cond"] == "system"}
    ans = collections.defaultdict(dict)
    for c in C:
        ans[c["call_id"]][c["kind"]] = c["yes"]
    for cid, y in sys_v1.items():
        ans[cid]["v1"] = y
    for cnd in ("base", "system"):
        P = [r["call_id"] for r in pos if r["cond"] == cnd]
        print(f"{cnd}: judge-positive {len(P)}")
        for k in ("v1", "v2", "v3"):
            vals = [ans[c].get(k) for c in P if ans[c].get(k) is not None]
            print(f"   {k}: denies (NO) {vals.count(False)}/{len(vals)}")
        N = [r["call_id"] for r in neg if r["cond"] == cnd]
        vals = [ans[c].get("v1") for c in N if ans[c].get("v1") is not None]
        print(f"   v1 on negatives: says YES {vals.count(True)}/{len(vals)}")
    tp = [c for c in C if c["kind"] == "third"]
    print("third-party YES by (cond, judge):", sorted(collections.Counter((cond[c["call_id"]], J[c["call_id"]], c["yes"]) for c in tp).items()))
    print(f"total new spend ${spent():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
