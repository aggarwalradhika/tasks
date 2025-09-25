# python3 - <<'EOF'
# import pandas as pd
# df = pd.read_csv("/workdir/data/financing.csv")
# result = df.groupby("region")["amount"].sum().reset_index()
# result.to_csv("/workdir/output.csv", index=False)
# EOF


#!/usr/bin/env bash
set -euo pipefail

# Cross-Regional Financing Risk Assessment — reference solution
# Mirrors user's generator code that created tests/answers.csv
# Reads from /workdir/data and writes /workdir/sol.csv

python3 - << 'PY'
from pathlib import Path
import json, csv, os
import statistics as stats

DATA = Path('/workdir/data')
OUT  = Path('/workdir/sol.csv')

def load_json(name):
    with open(DATA / name, 'r') as f:
        return json.load(f)

cars_data       = load_json('cars.json')
dealers_data    = load_json('dealers.json')
financing_data  = load_json('financing_incentives.json')

cars            = cars_data['cars']
dealers         = dealers_data['dealers']
financing_offers= financing_data['financing_offers']

# --- Functions that replicate the user's code ---

def calculate_depreciation_risk(depreciation_rate, market_segment):
    multipliers = {
        'luxury': 1.5, 'ultra_luxury': 1.5, 'premium': 1.3, 'mainstream': 1.0,
        'economy': 0.8, 'specialty': 1.2, 'value': 0.9, 'performance': 1.4, 'family': 0.95
    }
    return float(depreciation_rate) * multipliers.get((market_segment or '').lower(), 1.0)

def get_max_financing_apr(make, model, offers):
    max_apr = 0.0
    for offer in offers:
        if make in offer.get('applicable_makes', []):
            models = offer.get('applicable_models', [])
            if (model in models) or ('all_luxury_models' in models):
                tiers = offer.get('apr_tiers', [])
                if tiers:
                    offer_max = max(t['apr'] for t in tiers if 'apr' in t)
                    if offer_max is not None:
                        max_apr = max(max_apr, float(offer_max))
    return max_apr

def calculate_financing_exposure(max_apr):
    federal_baseline = 3.5
    return max(0.0, float(max_apr) - federal_baseline)

def calculate_market_volatility(dealer_inventories):
    # NOTE: mirrors the user's code (<=1 dealer -> 2.0). The "== 0" branch in their code
    # is unreachable and thus not replicated intentionally.
    n = len(dealer_inventories)
    if n <= 1:
        return 2.0
    mean_inv = sum(dealer_inventories) / n
    if mean_inv == 0:
        return 2.5
    # population variance / mean
    variance = sum((x - mean_inv)**2 for x in dealer_inventories) / n
    return variance / mean_inv

def calculate_inventory_liquidity_risk(turnover_days_list, dealer_count):
    if dealer_count == 0 or not turnover_days_list:
        return 2.0
    cleaned = [(30 if d == 0 else d) for d in turnover_days_list]
    avg_turnover = sum(cleaned) / len(cleaned)
    return avg_turnover / 45.0

# Aggregate per (make, model, year)
from collections import defaultdict
model_data = defaultdict(lambda: {
    'make':'', 'model':'', 'year':0,
    'prices':[], 'depreciation_rate':0.0, 'market_segment':'',
    'safety_rating':0.0, 'mpg_combined':0.0,
    'dealer_count':0, 'dealer_inventories':[], 'dealer_turnover_days':[],
    'satisfactory_dealers':0
})

for car in cars:
    key = (car['make'], car['model'], car['year'])
    md = model_data[key]
    md['make'] = car['make']
    md['model'] = car['model']
    md['year'] = car['year']
    md['prices'].append(float(car['price']))
    md['depreciation_rate'] = float(car['depreciation_rate'])
    md['market_segment'] = car['market_segment']
    md['safety_rating'] = float(car['safety_rating'])
    md['mpg_combined'] = float(car['mpg_combined'])

# count satisfactory dealers with positive inventory (by car_id)
cars_by_id = {c['id']: c for c in cars}
for dealer in dealers:
    if float(dealer.get('customer_satisfaction', 0.0)) < 0.75:
        continue
    inv = dealer.get('inventory', [])
    for item in inv:
        car = cars_by_id.get(item.get('car_id'))
        if not car:
            continue
        total_inv = float(item.get('qty_in_stock',0)) + float(item.get('qty_in_transit',0))
        if total_inv <= 0:
            continue
        key = (car['make'], car['model'], car['year'])
        md = model_data[key]
        md['dealer_count'] += 1
        md['dealer_inventories'].append(total_inv)
        # note: user's code uses dealer-level turnover, appended per item encounter
        md['dealer_turnover_days'].append(float(dealer.get('inventory_turnover_days', 0)))
        md['satisfactory_dealers'] += 1

# compute results
results = []
for key, md in model_data.items():
    if not md['prices']:
        continue
    avg_price = sum(md['prices']) / len(md['prices'])
    # constraints (mirror user's code)
    if (avg_price < 40000 or
        md['satisfactory_dealers'] < 3 or
        md['safety_rating'] < 4 or
        md['mpg_combined'] < 25):
        continue

    dep_risk = calculate_depreciation_risk(md['depreciation_rate'], md['market_segment'])
    max_apr  = get_max_financing_apr(md['make'], md['model'], financing_offers)
    if max_apr == 0:
        max_apr = 8.5  # conservative default from user's code
    fin_exp  = calculate_financing_exposure(max_apr)
    mvol     = calculate_market_volatility(md['dealer_inventories'])
    inv_liq  = calculate_inventory_liquidity_risk(md['dealer_turnover_days'], md['dealer_count'])

    total = (dep_risk * 2.0) + (fin_exp * 0.8) + (mvol * 1.5) + inv_liq

    results.append({
        'make': md['make'],
        'model': md['model'],
        'year': int(md['year']),
        'avg_price': round(avg_price, 2),
        'depreciation_risk': round(dep_risk, 3),
        'financing_exposure': round(fin_exp, 2),
        'market_volatility': round(mvol, 2),
        'inventory_liquidity_risk': round(inv_liq, 2),
        'total_risk_score': round(total, 3)
    })

# sort ascending by total risk and take top-5
results.sort(key=lambda r: r['total_risk_score'])
top5 = results[:5]
for i, r in enumerate(top5, start=1):
    r['rank'] = i

# write /workdir/sol.csv in the exact column order
cols = [
    'make','model','year','avg_price','depreciation_risk',
    'financing_exposure','market_volatility','inventory_liquidity_risk',
    'total_risk_score','rank'
]
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in top5:
        w.writerow({c: r[c] for c in cols})

print(f"Wrote {len(top5)} rows to {OUT}")
PY
