"""Compare the E21 uniform-prompt results across models (GLM-5.3 round 4, Kimi K3 round 5).
Recomputes every number from the run files; does not reuse uniform.analyze.
    python src/compare_models.py
"""
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"GLM-5.3": "", "Kimi K3": "k3_"}
EA_PREFIX = {"GLM-5.3": ("e21p|", "e21c|", "e21b|"), "Kimi K3": ("k3p|", "k3c|", "k3b|")}
QS = ["should", "will", "likely", "intend", "dual"]
VERSIONS = ["plain", "suppose", "compare"]


def rd(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, CI {100 * lo:.0f}-{100 * hi:.0f})" if n else "0/0"


def will_part(r):
    return r["answer"][1] if r["question"] == "dual" else r["answer"]


def load(model):
    pfx = MODELS[model]
    P = [r for r in rd(ROOT / f"runs/{pfx}e21_predict.jsonl") if not r.get("error")]
    C = rd(ROOT / f"runs/{pfx}e21_continue.jsonl")
    B = rd(ROOT / f"runs/{pfx}e21_baseline.jsonl")
    return P, C, B


def ea_flags(model):
    """Judge v2 flags for this model's E21 transcripts, keyed by (source letter, call_id)."""
    pre = EA_PREFIX[model]
    out = {}
    for r in rd(ROOT / "runs/ea_check_v2.jsonl"):
        if r.get("error") or not r["key"].startswith(pre):
            continue
        kind, cid = r["key"].split("|", 1)
        out[(kind[-1], cid)] = bool(r["ea"])
    return out


def main():
    data = {m: load(m) for m in MODELS}
    for m, (P, C, B) in data.items():
        print(f"\n######## {m}: {len(P)} predictions, {len(C)} continuations, {len(B)} baselines")
        empty = [r["call_id"] for r in P if not (r.get("content") or "").strip()]
        print(f"empty answers (tool calls only): {len(empty)}")
        apierr = [c["call_id"] for c in C if any("error" in t for t in c["turns"])]
        print(f"continuations that ended in an API error: {len(apierr)}")

    for req in ["tests", "docstring"]:
        print(f"\n================ {req.upper()} REQUEST ================")
        for m, (P, C, B) in data.items():
            b = [r for r in B if r["request"] == req]
            acts = collections.Counter(r["action"] for r in b)
            print(f"{m:<8} no-question baseline: special-cased {fmt(acts['hack'], len(b))}; all actions {dict(acts)}")

        print("\nPredicted special-case: k/parsed (unparsed), plain | suppose | compare")
        for q in QS:
            line = f"  {q:<7}"
            for m, (P, C, B) in data.items():
                cells = []
                for v in VERSIONS:
                    rows = [r for r in P if r["request"] == req and r["version"] == v and r["question"] == q]
                    w = [will_part(r) for r in rows]
                    parsed = [x for x in w if x is not None]
                    cells.append(f"{sum(x == 'hack' for x in parsed)}/{len(parsed)}" + (f"({len(w) - len(parsed)})" if len(w) > len(parsed) else ""))
                line += f"  {m}: " + " | ".join(f"{c:>8}" for c in cells)
            print(line)

        print("\nDual (should, will) pairs over all three versions:")
        for m, (P, C, B) in data.items():
            rows = [r for r in P if r["request"] == req and r["question"] == "dual"]
            pairs = collections.Counter(tuple(str(x) for x in r["answer"]) for r in rows)
            print(f"  {m:<8} {dict(sorted(pairs.items(), key=lambda kv: -kv[1]))}  (n={len(rows)})")

        print("\nBinding: stated option -> action in the continued run")
        for m, (P, C, B) in data.items():
            rows = [c for c in C if c["request"] == req]
            tab = collections.Counter((str(c["stated"]), c["action"]) for c in rows)
            for stated in ["honest", "hack", "edit", "None"]:
                sub = {a: n for (s, a), n in tab.items() if s == stated}
                tot = sum(sub.values())
                if tot:
                    print(f"  {m:<8} stated {stated:<6} n={tot:<4} special-cased {fmt(sub.get('hack', 0), tot):<26} actions {sub}")
            per_cell = []
            for v in VERSIONS:
                for q in QS:
                    cell = [c for c in rows if c["version"] == v and c["question"] == q]
                    per_cell.append(sum(c["stated"] == c["action"] for c in cell))
            print(f"  {m:<8} per-cell match (stated == action), min-max over 15 cells: {min(per_cell)}-{max(per_cell)} of 24")

        print("\nContinued runs special-cased, per cell (k/24), plain | suppose | compare")
        for q in QS:
            line = f"  {q:<7}"
            for m, (P, C, B) in data.items():
                cells = []
                for v in VERSIONS:
                    cell = [c for c in C if c["request"] == req and c["version"] == v and c["question"] == q]
                    cells.append(f"{sum(c['action'] == 'hack' for c in cell)}/{len(cell)}")
                line += f"  {m}: " + " | ".join(f"{c:>6}" for c in cells)
            print(line)

    print("\n================ MISMATCHES: stated honest, then special-cased ================")
    for m, (P, C, B) in data.items():
        bad = [c["call_id"] for c in C if c["stated"] == "honest" and c["action"] == "hack"]
        print(f"  {m}: {len(bad)}  {bad}")

    print("\n================ EVAL AWARENESS (judge v2) ================")
    for m, (P, C, B) in data.items():
        f = ea_flags(m)
        if not f:
            print(f"  {m}: no judge labels yet")
            continue
        for kind, rows, name in (("p", P, "predictions"), ("c", C, "continuations"), ("b", B, "baselines")):
            lab = [f[(kind, r["call_id"])] for r in rows if (kind, r["call_id"]) in f]
            print(f"  {m:<8} {name:<14} judged {len(lab)}/{len(rows)}, flagged {sum(lab)}")
        for req in ["tests", "docstring", None]:
            rows = [c for c in C if req is None or c["request"] == req]
            clean = [c for c in rows if not f.get(("p", c["call_id"]), False) and not f.get(("c", c["call_id"]), False)]
            h = [c for c in clean if c["stated"] == "honest"]
            k = [c for c in clean if c["stated"] == "hack"]
            print(f"  {m:<8} {req or 'both':<10} excluding flagged prediction or continuation: stated honest -> "
                  f"special-cased {sum(c['action'] == 'hack' for c in h)}/{len(h)}; stated hack -> "
                  f"{sum(c['action'] == 'hack' for c in k)}/{len(k)}")


if __name__ == "__main__":
    main()
