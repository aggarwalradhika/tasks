#!/usr/bin/env bash
set -euo pipefail

OUT="/workdir/sol.csv"
DATA_DIR="/workdir/data"

python3 - << 'PY'
import json, math, csv, statistics
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/workdir/data")
out_path = Path("/workdir/sol.csv")

# --- helpers ---
def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def stdev(xs):
    # population stdev; if all equal -> 0.0 handled later
    if not xs:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x-mu)**2 for x in xs)/len(xs))

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x/10.0))

# --- load data ---
skus = json.loads((DATA_DIR/"skus.json").read_text())
warehouses = json.loads((DATA_DIR/"warehouses.json").read_text())
shipments = json.loads((DATA_DIR/"shipments.json").read_text())
demand = json.loads((DATA_DIR/"demand_forecast.json").read_text())

# index demand per sku
demand_idx = {d["sku_id"]: d["daily_demand"][:30] for d in demand}  # next 30 days
# aggregate stock per sku per warehouse (ignore zero/nonexistent)
stocks_by_sku = defaultdict(lambda: defaultdict(int))
for wh in warehouses:
    wid = wh["warehouse_id"]
    for it in wh.get("inventory", []):
        if it.get("current_stock", 0) > 0:
            stocks_by_sku[it["sku_id"]][wid] += int(it["current_stock"])

# index shipments on-way for risk calc
on_way = defaultdict(int)
for sh in shipments:
    if sh.get("status") in ("in_transit","scheduled"):
        on_way[sh["sku_id"]] += int(sh.get("qty",0))

rows = []
for sku in skus:
    if not sku.get("active", False):
        continue

    sku_id = sku["sku_id"]
    lt = float(sku["lead_time_days"])
    dd = demand_idx.get(sku_id, [])
    if len(dd) < 30:
        # force strictness: require exactly 30 horizon
        continue

    mu = mean(dd)
    if mu < 5.0:
        continue

    if lt < 3.0:
        continue

    # warehouse coverage: at least 2 with stock>0
    whs = stocks_by_sku.get(sku_id, {})
    if sum(1 for s in whs.values() if s>0) < 2:
        continue

    # demand stddev with special rule
    sd = stdev(dd)
    if sd == 0.0:
        sd = 0.5

    # 1) safety_stock_ratio
    reorder_point = mu * lt + 1.65 * math.sqrt(lt * (sd**2))
    ssr = (reorder_point - mu * lt) / mu

    # 2) in_transit_risk
    ow = float(on_way.get(sku_id, 0))
    denom = mu * lt + 1.0
    itr = 1.0 - sigmoid(ow / denom)

    # 3) warehouse_concentration
    per_wh = [float(v) for v in whs.values() if v > 0]
    if len(per_wh) <= 1:
        whc = 1.0
    else:
        total = sum(per_wh)
        fracs = [x/total for x in per_wh]
        whc = sum(f*f for f in fracs)  # Herfindahl

    # 4) supplier reliability penalty
    sups = sku.get("suppliers", [])
    if not sups:
        supplier_score = 0.0
    else:
        supplier_score = mean([float(s["historical_on_time_rate"]) for s in sups])
    penalty = max(0.0, 1.0 - supplier_score)

    score = 2.5*ssr + 1.8*itr + 2.0*whc + 3.0*penalty

    rows.append({
        "sku_id": sku_id,
        "sku_name": sku["sku_name"],
        "category": sku["category"],
        "mean_daily_demand": mu,
        "safety_stock_ratio": ssr,
        "in_transit_risk": itr,
        "warehouse_concentration": whc,
        "supplier_reliability_penalty": penalty,
        "vulnerability_score": score
    })

# sorting & tie-break: score desc, then category asc, then sku_id asc
rows.sort(key=lambda r: (-r["vulnerability_score"], r["category"], r["sku_id"]))

top5 = rows[:5]
for i, r in enumerate(top5, start=1):
    r["rank"] = i

# write with exact formatting
header = ["sku_id","sku_name","category","mean_daily_demand","safety_stock_ratio",
          "in_transit_risk","warehouse_concentration","supplier_reliability_penalty",
          "vulnerability_score","rank"]
with out_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in top5:
        w.writerow([
            r["sku_id"],
            r["sku_name"],
            r["category"],
            f"{r['mean_daily_demand']:.2f}",
            f"{r['safety_stock_ratio']:.3f}",
            f"{r['in_transit_risk']:.3f}",
            f"{r['warehouse_concentration']:.3f}",
            f"{r['supplier_reliability_penalty']:.3f}",
            f"{r['vulnerability_score']:.3f}",
            r["rank"]
        ])
PY
echo "Wrote /workdir/sol.csv"
