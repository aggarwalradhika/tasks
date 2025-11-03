#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3

$PYTHON - << 'PY'
#!/usr/bin/env python3
"""
Enhanced Supply Chain Anomaly Detection with Multi-Constraint Selection

This solution demonstrates the constraint satisfaction approach required
for the enhanced task.

CRITICAL FIX: Output contains ONLY header + 8 data rows (no validation summary)
"""

import json
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import statistics
from itertools import combinations

DATA = Path("/workdir/data")
SOL = Path("/workdir/sol.csv")

def load(name):
    """Load JSON file from data directory"""
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)

# Region definitions
WEST_CITIES = {"Los Angeles", "Phoenix", "Salt Lake City", "San Francisco", "San Diego"}
EAST_CITIES = {"Miami", "New York", "Boston", "Washington DC", "Tampa"}
# Everything else is Central

def get_region(city):
    """Determine region from destination city"""
    if city in WEST_CITIES:
        return "West"
    elif city in EAST_CITIES:
        return "East"
    else:
        return "Central"

def check_constraints(route_combination, route_data):
    """
    Check if a combination of 8 routes satisfies all constraints
    
    Returns: (is_valid, validation_details)
    """
    regions = {"West": 0, "East": 0, "Central": 0}
    total_volume = 0
    warehouse_counts = defaultdict(int)
    unique_cities = set()
    
    for route in route_combination:
        route_info = route_data[route["route_id"]]
        region = route_info["region"]
        regions[region] += 1
        total_volume += route_info["monthly_volume"]
        warehouse_counts[route_info["warehouse_id"]] += 1
        unique_cities.add(route_info["destination_city"])
    
    # Check all constraints
    constraint_1 = regions["West"] >= 2 and regions["East"] >= 2 and regions["Central"] >= 2
    constraint_2 = total_volume >= 2800
    constraint_3 = all(count <= 3 for count in warehouse_counts.values())
    constraint_4 = len(unique_cities) >= 6
    
    is_valid = constraint_1 and constraint_2 and constraint_3 and constraint_4
    
    details = {
        "regions": regions,
        "total_volume": total_volume,
        "max_warehouse_count": max(warehouse_counts.values()) if warehouse_counts else 0,
        "unique_cities": len(unique_cities),
        "warehouse_distribution": dict(warehouse_counts)
    }
    
    return is_valid, details

def main():
    # Load data
    routes = load("routes.json")
    shipments = load("shipments.json")
    weather_events = load("weather_events.json")
    warehouse_capacity = load("warehouse_capacity.json")

    # Build indices
    route_by_id = {r["route_id"]: r for r in routes}
    warehouse_by_id = {w["warehouse_id"]: w for w in warehouse_capacity}

    # Group shipments and weather by route (last 60 days)
    shipments_by_route = defaultdict(list)
    weather_by_route = defaultdict(list)
    warehouse_by_route = {}  # Most common warehouse for each route

    ref_date = datetime.strptime("2025-10-30", "%Y-%m-%d")

    for s in shipments:
        ship_date = datetime.strptime(s["date"], "%Y-%m-%d")
        if (ref_date - ship_date).days <= 60:
            shipments_by_route[s["route_id"]].append(s)
            # Track warehouse (use first one we see)
            if s["route_id"] not in warehouse_by_route:
                warehouse_by_route[s["route_id"]] = s["destination_warehouse_id"]

    for w in weather_events:
        weather_date = datetime.strptime(w["date"], "%Y-%m-%d")
        if (ref_date - weather_date).days <= 60:
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

    print(f"Eligible routes: {len(eligible)}")

    # Compute anomaly scores for all eligible routes
    all_routes = []
    for rid in eligible:
        r = route_by_id[rid]
        route_shipments = shipments_by_route[rid]
        route_weather = weather_by_route[rid]
        
        # Calculate all 4 component scores (using exact grader methods)
        
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
        
        # Calculate anomaly score
        anomaly_score = (3.0 * delay_volatility + 
                        2.5 * cost_inflation + 
                        4.0 * weather_correlation + 
                        3.5 * capacity_penalty)
        
        all_routes.append({
            "route_id": r["route_id"],
            "route_name": r["route_name"],
            "origin_city": r["origin_city"],
            "destination_city": r["destination_city"],
            "region": get_region(r["destination_city"]),
            "monthly_volume": r["monthly_volume"],
            "warehouse_id": warehouse_by_route.get(rid, "UNKNOWN"),
            "distance_km": r["distance_km"],
            "delay_volatility_score": delay_volatility,
            "cost_inflation_score": cost_inflation,
            "weather_impact_correlation": weather_correlation,
            "capacity_utilization_penalty": capacity_penalty,
            "anomaly_score": anomaly_score
        })

    # CRITICAL: Constraint satisfaction search
    print(f"\nSearching for optimal valid combination...")
    print(f"Total eligible routes: {len(all_routes)}")
    
    # Build lookup dict
    route_data = {r["route_id"]: r for r in all_routes}
    
    best_combination = None
    best_total_score = -1
    best_total_distance = -1
    valid_combinations_found = 0
    
    # Try all combinations of 8 routes
    for combo in combinations(all_routes, 8):
        is_valid, details = check_constraints(combo, route_data)
        
        if is_valid:
            valid_combinations_found += 1
            total_score = sum(r["anomaly_score"] for r in combo)
            total_distance = sum(r["distance_km"] for r in combo)
            
            # Check if this is better than current best
            if (total_score > best_total_score or 
                (total_score == best_total_score and total_distance > best_total_distance)):
                best_combination = combo
                best_total_score = total_score
                best_total_distance = total_distance
                best_details = details
    
    print(f"Valid combinations found: {valid_combinations_found}")
    print(f"Best total anomaly score: {best_total_score:.3f}")
    print(f"Best total distance: {best_total_distance} km")
    
    if best_combination is None:
        raise ValueError("No valid combination found that satisfies all constraints!")
    
    # Sort the best combination by anomaly score for ranking
    selected_routes = list(best_combination)
    selected_routes.sort(key=lambda x: (-x["anomaly_score"], x["route_id"]))
    
    # Assign ranks
    for i, route in enumerate(selected_routes, start=1):
        route["rank"] = i
    
    print(f"\nSelected routes:")
    for route in selected_routes:
        print(f"  {route['rank']}. {route['route_id']} - Score: {route['anomaly_score']:.3f}, Region: {route['region']}")
    
    print(f"\nValidation (for debugging - not in output):")
    print(f"  Total volume: {best_details['total_volume']} >= 2800")
    print(f"  Regional diversity: West={best_details['regions']['West']}, East={best_details['regions']['East']}, Central={best_details['regions']['Central']}")
    print(f"  Geographic diversity: {best_details['unique_cities']} cities")
    print(f"  Warehouse distribution: max {best_details['max_warehouse_count']} routes per warehouse")
    
    # Write output with exact formatting
    # CRITICAL FIX: NO validation summary - just header + 8 data rows
    def f3(x): 
        return f"{x:.3f}"
    
    with open(SOL, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Write header
        w.writerow([
            "route_id", "route_name", "origin_city", "destination_city",
            "destination_region", "monthly_volume", "destination_warehouse_id", "distance_km",
            "delay_volatility_score", "cost_inflation_score", "weather_impact_correlation",
            "capacity_utilization_penalty", "anomaly_score", "rank"
        ])
        # Write data rows ONLY - no validation summary
        for route in selected_routes:
            w.writerow([
                route["route_id"],
                route["route_name"],
                route["origin_city"],
                route["destination_city"],
                route["region"],
                route["monthly_volume"],
                route["warehouse_id"],
                route["distance_km"],
                f3(route["delay_volatility_score"]),
                f3(route["cost_inflation_score"]),
                f3(route["weather_impact_correlation"]),
                f3(route["capacity_utilization_penalty"]),
                f3(route["anomaly_score"]),
                route["rank"]
            ])
    
    print(f"\nOutput written to {SOL}")
    print(f"Total lines in file: 9 (1 header + 8 data rows)")

if __name__ == "__main__":
    main()
PY