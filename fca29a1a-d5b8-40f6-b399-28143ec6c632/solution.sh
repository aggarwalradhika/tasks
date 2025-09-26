#!/usr/bin/env bash
set -euo pipefail

# Cross-Regional Financing Risk Assessment — reference solution
# Mirrors user's generator code that created tests/answers.csv
# Reads from /workdir/data and writes /workdir/sol.csv

python3 - << 'PY'
from pathlib import Path
import json, csv
from collections import defaultdict

DATA = Path('/workdir/data')
OUT  = Path('/workdir/sol.csv')

def load_json(name):
    return json.loads((DATA / name).read_text())

cars            = load_json('cars.json')["cars"]
dealers         = load_json('dealers.json')["dealers"]
financing_offers= load_json('financing_incentives.json')["financing_offers"]

SAT_THRESHOLD = 0.75
MIN_PRICE, MIN_SAFETY, MIN_MPG, MIN_DEALERS = 40000, 4, 25, 3
FED_BASELINE, DEFAULT_MAX_APR = 3.5, 8.5

SEG_MULT = {
  'luxury':1.5,'ultra_luxury':1.5,'premium':1.3,'mainstream':1.0,
  'economy':0.8,'specialty':1.2,'value':0.9,'performance':1.4,'family':0.95
}

def dep_risk(rate, seg):
    return float(rate) * SEG_MULT.get((seg or '').lower(), 1.0)

def max_apr(make, model, offers):
    mx = None
    for o in offers:
        if make in o.get('applicable_makes', []):
            models = o.get('applicable_models', [])
            if (model in models) or ('all_luxury_models' in models):
                for t in o.get('apr_tiers', []):
                    a = t.get('apr')
                    if a is not None:
                        mx = float(a) if mx is None else max(mx, float(a))
    return mx if mx is not None else DEFAULT_MAX_APR

def fin_exposure(mx): return max(0.0, float(mx) - FED_BASELINE)

def pop_var(vals):
    n = len(vals)
    if n == 0: return 0.0
    m = sum(vals)/n
    return sum((v-m)**2 for v in vals)/n

def market_vol(inv_counts):
    if len(inv_counts) <= 1: return 2.0
    mean = sum(inv_counts)/len(inv_counts)
    if mean == 0: return 2.5
    return pop_var(inv_counts)/mean

def inv_liq(turnover_days_list):
    if not turnover_days_list: return 2.0
    cleaned = [30.0 if d == 0 else float(d) for d in turnover_days_list]
    return (sum(cleaned)/len(cleaned))/45.0

# Index cars by (make,model,year) and by id
cars_by_id = {c['id']: c for c in cars}
keys = {(c['make'], c['model'], int(c['year'])) for c in cars}

results = []
for mk, md, yr in keys:
    rows = [c for c in cars if c['make']==mk and c['model']==md and int(c['year'])==yr]
    prices = [float(c['price']) for c in rows]
    if not prices: continue
    avg_price = sum(prices)/len(prices)

    c0 = rows[0]
    safety = float(c0['safety_rating'])
    mpg    = float(c0['mpg_combined'])
    car_ids = {c['id'] for c in rows}

    # ---- UNIQUE DEALERS PER MODEL ----
    unique_dealer_ids = set()
    inv_counts = []
    turnover_days = []
    for d in dealers:
        if float(d.get('customer_satisfaction', 0.0)) < SAT_THRESHOLD:
            continue
        # If any inventory item for any car_id of this model has positive qty, count this dealer once
        has_model_inventory = False
        for it in d.get('inventory', []):
            if it.get('car_id') in car_ids:
                qty = float(it.get('qty_in_stock',0)) + float(it.get('qty_in_transit',0))
                if qty > 0:
                    has_model_inventory = True
                    break
        if has_model_inventory:
            dealer_id = d.get('id') or id(d)  # fall back to object id if no id field in data
            if dealer_id not in unique_dealer_ids:
                unique_dealer_ids.add(dealer_id)
                # Representative inventory count for volatility: sum of positive quantities for this model at this dealer
                qty_sum = 0.0
                for it in d.get('inventory', []):
                    if it.get('car_id') in car_ids:
                        qty_sum += max(0.0, float(it.get('qty_in_stock',0)) + float(it.get('qty_in_transit',0)))
                inv_counts.append(qty_sum if qty_sum>0 else 0.0)
                turnover_days.append(float(d.get('inventory_turnover_days', 0)))

    # ---- Eligibility with UNIQUE dealers ----
    if not (avg_price >= MIN_PRICE and safety >= MIN_SAFETY and mpg >= MIN_MPG and len(unique_dealer_ids) >= MIN_DEALERS):
        continue

    dep = dep_risk(float(c0['depreciation_rate']), c0.get('market_segment'))
    mx  = max_apr(mk, md, financing_offers)
    fin = fin_exposure(mx)
    vol = market_vol(inv_counts)
    liq = inv_liq(turnover_days)
    total = 2.0*dep + 0.8*fin + 1.5*vol + liq

    results.append({
        'make': mk, 'model': md, 'year': yr,
        'avg_price': float(f'{avg_price:.2f}'),
        'depreciation_risk': float(f'{dep:.3f}'),
        'financing_exposure': float(f'{fin:.2f}'),
        'market_volatility': float(f'{vol:.2f}'),
        'inventory_liquidity_risk': float(f'{liq:.2f}'),
        'total_risk_score': float(f'{total:.3f}')
    })

# sort asc by total risk, tie-break by make/model/year, take 5
results.sort(key=lambda r: (r['total_risk_score'], r['make'], r['model'], r['year']))
top5 = results[:5]
for i, r in enumerate(top5, 1):
    r['rank'] = i

cols = ["make","model","year","avg_price","depreciation_risk","financing_exposure",
        "market_volatility","inventory_liquidity_risk","total_risk_score","rank"]
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in top5:
        w.writerow({k: r[k] for k in cols})
print(f"Wrote {len(top5)} rows to {OUT}")
PY
