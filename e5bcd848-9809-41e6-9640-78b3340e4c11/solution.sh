#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PY'
from pathlib import Path
import json, csv
from statistics import median

DATA = Path("/workdir/data")
OUT  = Path("/workdir/sol.csv")
COLS = ["node","delay_contribution_score","weighted_impact","normalized_impact"]

def fmt4(x: float) -> str:
    return f"{float(x):.4f}"

def as_int(x):
    try: return int(x)
    except: return 0

# Load
shipments = json.loads((DATA/"shipments.json").read_text(encoding="utf-8"))
capacities = json.loads((DATA/"capacities.json").read_text(encoding="utf-8"))

# Capacity index and default capacity
caps_idx = {}
pos_caps = []
for c in capacities:
    node = str(c.get("node") or "").strip()
    if node:
        caps_idx[node.casefold()] = {"node": node, "cap": float(c.get("daily_capacity") or 0)}
        capv = c.get("daily_capacity")
        if isinstance(capv,(int,float)) and capv>0:
            pos_caps.append(float(capv))
default_cap = float(int(round(median(pos_caps)))) if pos_caps else 1.0

def canon(n: str) -> str:
    n2 = (n or "").strip()
    k = n2.casefold()
    if k in caps_idx: return caps_idx[k]["node"]
    return n2.title()

# Delayed shipments
delayed = []
for s in shipments:
    exp = as_int(s.get("expected_days"))
    act = as_int(s.get("actual_days"))
    if act > exp:
        s2 = dict(s)
        s2["delay"] = act - exp
        delayed.append(s2)

# Aggregate
total_delay = {}
total_qty   = {}
for s in delayed:
    hops = [str(h) for h in (s.get("hops") or [])]
    dest = str(s.get("destination") or "")
    expanded = [*hops, dest]

    seen = set()
    uniq = []
    for n in expanded:
        k = (n or "").strip().casefold()
        if not k or k in seen: continue
        seen.add(k); uniq.append(n)

    qty = max(0.0, float(as_int(s.get("quantity"))))

    for n in uniq:
        name = canon(n)
        total_delay[name] = total_delay.get(name, 0.0) + float(s["delay"])
        total_qty[name]   = total_qty.get(name,   0.0) + qty

rows = []
for node in total_delay:
    cap = caps_idx.get(node.casefold(), {}).get("cap", default_cap)
    if cap <= 0: cap = default_cap
    dcs = total_delay[node] / cap
    wi  = dcs * total_qty.get(node, 0.0)
    rows.append({"node": node, "delay_contribution_score": dcs, "weighted_impact": wi})

wi_max = max((r["weighted_impact"] for r in rows), default=0.0)
for r in rows:
    r["normalized_impact"] = 0.0 if wi_max == 0 else (r["weighted_impact"] / wi_max)

rows.sort(key=lambda r: (-r["normalized_impact"], -r["weighted_impact"], r["node"].casefold()))

# Emit CSV with exact formatting
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for r in rows:
        w.writerow([
            r["node"],
            fmt4(r["delay_contribution_score"]),
            fmt4(r["weighted_impact"]),
            fmt4(r["normalized_impact"]),
        ])
PY
