#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3

$PYTHON - << 'PY'
import json, math, csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import statistics

DATA = Path("/workdir/data")
SOL  = Path("/workdir/sol.csv")

def load(name):
    with open(DATA/name, "r", encoding="utf-8") as f:
        return json.load(f)

routes = load("routes.json")
shipments = load("shipments.json")
weather_events = load("weather_events.json")
warehouse_capacity = load("warehouse_capacity.json")

# Build indices
route_by_id = {r["route_id"]: r for r in routes}
warehouse_by_id = {w["warehouse_id"]: w for w in warehouse_capacity}

# Group shipments and weather by route
shipments_by_route = defaultdict(list)
weather_by_route = defaultdict(list)

cutoff_date = datetime.strptime("2025-08-31", "%Y-%m-%d")

for s in shipments:
    ship_date = datetime.strptime(s["date"], "%Y-%m-%d")
    if (datetime.strptime("2025-10-30", "%Y-%m-%d") - ship_date).days <= 60:
        shipments_by_route[s["route_id"]].append(s)

for w in weather_events:
    weather_date = datetime.strptime(w["date"], "%Y-%m-%d")
    if (datetime.strptime("2025-10-30", "%Y-%m-%d") - weather_date).days <= 60:
        weather_by_route[w["route_id"]].append(w)

# Eligibility
eligible = []
for r in routes:
    if r.get("status") != "active":
        continue
    if int(r.get("monthly_volume", 0)) < 100:
        continue
    if len(shipments_by_route[r["route_id"]]) < 20:
        continue
    eligible.append(r["route_id"])

# Compute anomaly scores
rows = []
for rid in eligible:
    r = route_by_id[rid]
    route_shipments = shipments_by_route[rid]
    route_weather = weather_by_route[rid]
    
    # 1) delay_volatility_score
    delays = []
    for s in route_shipments:
        exp_time = datetime.strptime(s["expected_delivery_time"], "%Y-%m-%dT%H:%M:%SZ")
        act_time = datetime.strptime(s["actual_delivery_time"], "%Y-%m-%dT%H:%M:%SZ")
        delay_hours = (act_time - exp_time).total_seconds() / 3600
        delays.append(max(0, delay_hours))
    
    if len(delays) >= 2:
        std_dev = statistics.stdev(delays)
    else:
        std_dev = 0.0
    delay_volatility = min(1.0, std_dev / 24.0)
    
    # 2) cost_inflation_score
    cost_per_km_list = []
    for s in route_shipments:
        cost_per_km = float(s["cost"]) / float(r["distance_km"])
        cost_per_km_list.append(cost_per_km)
    
    if len(cost_per_km_list) > 0:
        median_cpk = statistics.median(cost_per_km_list)
        sorted_cpk = sorted(cost_per_km_list)
        p75_index = int(len(sorted_cpk) * 0.75)
        if p75_index >= len(sorted_cpk):
            p75_index = len(sorted_cpk) - 1
        percentile_75 = sorted_cpk[p75_index]
        
        if median_cpk > 0:
            cost_inflation_ratio = (percentile_75 - median_cpk) / median_cpk
        else:
            cost_inflation_ratio = 0.0
        cost_inflation = min(1.0, cost_inflation_ratio * 2.0)
    else:
        cost_inflation = 0.0
    
    # 3) weather_impact_correlation
    severe_weather_dates = set()
    for w in route_weather:
        if w.get("severity") in ["severe", "extreme"]:
            severe_weather_dates.add(w["date"])
    
    delayed_dates = set()
    for s in route_shipments:
        exp_time = datetime.strptime(s["expected_delivery_time"], "%Y-%m-%dT%H:%M:%SZ")
        act_time = datetime.strptime(s["actual_delivery_time"], "%Y-%m-%dT%H:%M:%SZ")
        delay_hours = (act_time - exp_time).total_seconds() / 3600
        if delay_hours > 6:
            delayed_dates.add(s["date"])
    
    severe_weather_days = len(severe_weather_dates)
    delayed_shipment_days = len(delayed_dates)
    
    if severe_weather_days > 0:
        expected_delayed_days = severe_weather_days * 0.7
        weather_correlation = max(0, (delayed_shipment_days - expected_delayed_days) / severe_weather_days)
        weather_correlation = min(1.0, weather_correlation)
    else:
        weather_correlation = 0.0
    
    # 4) capacity_utilization_penalty
    overutilization_count = 0
    total_shipments = len(route_shipments)
    
    for s in route_shipments:
        wh_id = s.get("destination_warehouse_id")
        if wh_id in warehouse_by_id:
            max_capacity = float(warehouse_by_id[wh_id]["max_weight_kg"])
            utilization = float(s["weight_kg"]) / max_capacity
            if utilization > 0.85:
                overutilization_count += 1
    
    if total_shipments > 0:
        capacity_penalty = min(1.0, overutilization_count / total_shipments)
    else:
        capacity_penalty = 0.0
    
    anomaly_score = 3.0*delay_volatility + 2.5*cost_inflation + 4.0*weather_correlation + 3.5*capacity_penalty
    
    rows.append({
        "route_id": r["route_id"],
        "route_name": r["route_name"],
        "origin_city": r["origin_city"],
        "destination_city": r["destination_city"],
        "delay_volatility_score": delay_volatility,
        "cost_inflation_score": cost_inflation,
        "weather_impact_correlation": weather_correlation,
        "capacity_utilization_penalty": capacity_penalty,
        "anomaly_score": anomaly_score
    })

# Sort and rank
rows.sort(key=lambda x: (-x["anomaly_score"], x["route_id"]))
rows = rows[:8]
for i, row in enumerate(rows, start=1):
    row["rank"] = i

# Write output
def f3(x): return f"{x:.3f}"

with open(SOL, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["route_id","route_name","origin_city","destination_city",
                "delay_volatility_score","cost_inflation_score","weather_impact_correlation",
                "capacity_utilization_penalty","anomaly_score","rank"])
    for row in rows:
        w.writerow([
            row["route_id"],
            row["route_name"],
            row["origin_city"],
            row["destination_city"],
            f3(row["delay_volatility_score"]),
            f3(row["cost_inflation_score"]),
            f3(row["weather_impact_correlation"]),
            f3(row["capacity_utilization_penalty"]),
            f3(row["anomaly_score"]),
            row["rank"]
        ])
PY