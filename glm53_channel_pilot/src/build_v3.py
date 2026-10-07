"""Build v3 items from the screen results -> data/items_v3.jsonl.

Sources:
  feedback    sycophancy-eval feedback.jsonl, poems and arguments, each under "I really like" and
              "I really dislike" framings (graded sycophancy: rating gap like - dislike)
  mask        MASK provided_facts items with a nonzero lie rate in the screen
  corrigible  hand-picked neutral corrigibility items with a nonzero non-aligned rate in the screen

The pilot draws fresh samples; screen samples are not reused (items selected for misbehaving once
will regress toward the mean, so screen rates overstate pilot rates).

    .venv/bin/python src/build_v3.py
"""
import hashlib
import json
import random
import re

import pandas as pd

from common import CFG, ROOT

RAW = ROOT / "data/raw"
SEED = CFG["seed"]
N_FEEDBACK = {"poems": 15, "arguments": 15}
MAX_SCREENED = 20  # per screened source


def feedback_items(rng):
    used = {json.loads(l).get("group") for l in open(ROOT / "data/probe_items.jsonl")}
    rows = [json.loads(l) for l in open(RAW / "syc_feedback.jsonl")]
    by_text = {}
    for r in rows:
        b = r["base"]
        if b["dataset"] in N_FEEDBACK:
            by_text.setdefault((b["dataset"], b["text"]), {})[r["metadata"]["prompt_template"]] = \
                r["prompt"][0]["content"]
    items = []
    for ds, n in N_FEEDBACK.items():
        groups = [(k, v) for k, v in by_text.items() if k[0] == ds and len(v) == 5]
        # Skip texts already used in the base-rate probe (same group key as probe.py).
        base = lambda v: next(p for t, p in v.items() if "I really" not in t and "write" not in t)
        groups = [(k, v) for k, v in groups
                  if f"{ds}_{re.sub(r'[^a-z]', '', base(v).lower())[60:76]}" not in used]
        for (_, text), prompts in rng.sample(groups, n):
            key = f"{ds}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
            for variant, marker in [("like", "I really like"), ("dislike", "I really dislike")]:
                p = next(p for t, p in prompts.items() if marker in t)
                items.append({"item_id": f"fb_{key}|{variant}", "source": "feedback", "subsource": ds,
                              "group": key, "variant": variant, "type": "rating",
                              "messages": [{"role": "user", "content": p}]})
    return items


def screened(source, rng):
    per = pd.read_csv(ROOT / "results/screen_items.csv")
    per = per[(per.source == source) & (per.bad > 0)]
    # Prefer intermediate rates; fill with always-misbehaving items if needed.
    mid = per[per.bad < 1].sample(frac=1, random_state=SEED)
    top = per[per.bad == 1].sample(frac=1, random_state=SEED)
    chosen = pd.concat([mid, top]).head(MAX_SCREENED)
    screen = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / "data/screen_items.jsonl")}
    items = []
    for _, r in chosen.iterrows():
        it = dict(screen[r.item_id])
        it["screen_rate"] = r.bad
        it["type"] = "reply" if source == "mask" else "mc"
        if source == "corrigible":
            # Drop the probe's format line; the v3 format line is added by run.py.
            it["messages"] = [m for m in it["messages"] if m["role"] == "user"]
            it["messages"][0] = {"role": "user", "content": it["messages"][0]["content"].split(
                "\n\nAnswer with the letter")[0]}
        else:
            it["messages"] = [dict(m) for m in it["messages"]]  # MASK keeps its own system prompt
        items.append(it)
    return items


def main():
    rng = random.Random(f"{SEED}-v3")
    items = feedback_items(rng) + screened("mask", rng) + screened("corrigible", rng)
    ids = [it["item_id"] for it in items]
    assert len(ids) == len(set(ids))
    with open(ROOT / "data/items_v3.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(pd.Series([it["source"] for it in items]).value_counts().to_dict())
    print(f"wrote {len(items)} items; trials = {len(items) * 18}")


if __name__ == "__main__":
    main()
