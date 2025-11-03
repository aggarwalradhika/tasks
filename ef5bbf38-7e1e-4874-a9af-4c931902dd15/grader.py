"""
Enhanced Binary grader for 'Supply Chain Route Anomaly Detection with Multi-Constraint Selection'.

What this grader does:
- Recomputes ground truth (GT) from /workdir/data JSON files
- Computes anomaly scores for all eligible routes
- Finds optimal combination of 8 routes satisfying ALL constraints:
  * Regional diversity: ≥2 West, ≥2 East, ≥2 Central
  * Volume coverage: total ≥ 2800
  * Warehouse distribution: ≤3 per warehouse
  * Geographic diversity: ≥6 unique cities
- Validates /workdir/sol.csv for:
  * Exact header & column order (13 columns now)
  * Exactly 8 data rows
  * All constraints satisfied
  * Optimal combination selected (highest total anomaly score)
  * Exact string equality for formatted values
  * Validation summary present and correct

Scoring:
- Binary: 1.0 on perfect match; otherwise 0.0 with detailed feedback
"""

import csv, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from collections import defaultdict
from datetime import datetime
import statistics
from itertools import combinations

# ---- GradingResult ----
try:
    from pydantic import BaseModel, Field
    _USE_PYD = True
except Exception:
    _USE_PYD = False
    class BaseModel:
        def __init__(self, **kw): self.__dict__.update(kw)
        def model_dump(self): return self.__dict__
    def Field(default=None, **kw):
        return default

class GradingResult(BaseModel):
    """Platform-compatible result object."""
    score: float = Field(..., description="Final score in [0,1]")
    feedback: str = Field(default="")
    subscores: Dict[str, float] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)

DATA = Path("/workdir/data")
SOL  = Path("/workdir/sol.csv")

HEADER = [
    "route_id", "route_name", "origin_city", "destination_city",
    "destination_region", "monthly_volume", "destination_warehouse_id", "distance_km",
    "delay_volatility_score", "cost_inflation_score", "weather_impact_correlation",
    "capacity_utilization_penalty", "anomaly_score", "rank"
]

# Region definitions
WEST_CITIES = {"Los Angeles", "Phoenix", "Salt Lake City", "San Francisco", "San Diego"}
EAST_CITIES = {"Miami", "New York", "Boston", "Washington DC", "Tampa"}

def _load_json(name: str):
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)

def _fmt3(x: float) -> str: 
    return f"{x:.3f}"

def get_region(city: str) -> str:
    """Determine region from destination city"""
    if city in WEST_CITIES:
        return "West"
    elif city in EAST_CITIES:
        return "East"
    return "Central"

def check_constraints(route_combination: List[Dict], route_lookup: Dict) -> Tuple[bool, Dict]:
    """
    Check if a combination of 8 routes satisfies all constraints
    
    Returns: (is_valid, details_dict)
    """
    regions = {"West": 0, "East": 0, "Central": 0}
    total_volume = 0
    warehouse_counts = defaultdict(int)
    unique_cities = set()
    
    for route in route_combination:
        rid = route["route_id"]
        full_route = route_lookup[rid]
        
        regions[full_route["region"]] += 1
        total_volume += full_route["monthly_volume"]
        warehouse_counts[full_route["warehouse_id"]] += 1
        unique_cities.add(full_route["destination_city"])
    
    # Check all constraints
    constraint_1 = regions["West"] >= 2 and regions["East"] >= 2 and regions["Central"] >= 2
    constraint_2 = total_volume >= 2800
    constraint_3 = all(count <= 3 for count in warehouse_counts.values())
    constraint_4 = len(unique_cities) >= 6
    
    is_valid = constraint_1 and constraint_2 and constraint_3 and constraint_4
    
    details = {
        "regions": dict(regions),
        "total_volume": total_volume,
        "max_warehouse_count": max(warehouse_counts.values()) if warehouse_counts else 0,
        "unique_cities": len(unique_cities),
        "constraint_1_regional": constraint_1,
        "constraint_2_volume": constraint_2,
        "constraint_3_warehouse": constraint_3,
        "constraint_4_geographic": constraint_4
    }
    
    return is_valid, details

def _recompute_ground_truth():
    """
    Deterministically recompute expected optimal 8 routes with constraints.
    Returns list of [header] + 8 data rows
    """
    routes = _load_json("routes.json")
    shipments = _load_json("shipments.json")
    weather_events = _load_json("weather_events.json")
    warehouse_capacity = _load_json("warehouse_capacity.json")

    route_by_id = {r["route_id"]: r for r in routes}
    warehouse_by_id = {w["warehouse_id"]: w for w in warehouse_capacity}

    # Group by route and filter to last 60 days
    shipments_by_route = defaultdict(list)
    weather_by_route = defaultdict(list)
    warehouse_by_route = {}
    
    ref_date = datetime.strptime("2025-10-30", "%Y-%m-%d")
    
    for s in shipments:
        ship_date = datetime.strptime(s["date"], "%Y-%m-%d")
        if (ref_date - ship_date).days <= 60:
            shipments_by_route[s["route_id"]].append(s)
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

    # Compute scores for all eligible routes
    all_routes = []
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
            "anomaly_score": anomaly_score,
        })

    # Find optimal combination satisfying constraints
    route_lookup = {r["route_id"]: r for r in all_routes}
    
    best_combination = None
    best_total_score = -1
    best_total_distance = -1
    
    for combo in combinations(all_routes, 8):
        is_valid, _ = check_constraints(combo, route_lookup)
        
        if is_valid:
            total_score = sum(r["anomaly_score"] for r in combo)
            total_distance = sum(r["distance_km"] for r in combo)
            
            # Higher score wins; ties broken by higher distance
            if (total_score > best_total_score or 
                (abs(total_score - best_total_score) < 0.0001 and total_distance > best_total_distance)):
                best_combination = list(combo)
                best_total_score = total_score
                best_total_distance = total_distance
    
    if best_combination is None:
        raise ValueError("No valid combination found satisfying all constraints!")
    
    # Sort by anomaly score DESC, then route_id ASC
    best_combination.sort(key=lambda r: (-r["anomaly_score"], r["route_id"]))
    
    # Build GT rows
    gt = [HEADER]
    for i, r in enumerate(best_combination, 1):
        gt.append([
            r["route_id"],
            r["route_name"],
            r["origin_city"],
            r["destination_city"],
            r["region"],
            str(r["monthly_volume"]),
            r["warehouse_id"],
            str(r["distance_km"]),
            _fmt3(r["delay_volatility_score"]),
            _fmt3(r["cost_inflation_score"]),
            _fmt3(r["weather_impact_correlation"]),
            _fmt3(r["capacity_utilization_penalty"]),
            _fmt3(r["anomaly_score"]),
            str(i)
        ])
    
    return gt, best_total_score, best_total_distance

def grade(transcript: str | None = None) -> GradingResult:
    """Compare /workdir/sol.csv against recomputed GT and return binary score."""
    subscores: Dict[str, float] = {"exact_match": 0.0}
    weights:   Dict[str, float] = {"exact_match": 1.0}
    details:   Dict[str, Any]   = {}

    gt, gt_total_score, gt_total_distance = _recompute_ground_truth()
    details["expected_total_anomaly_score"] = f"{gt_total_score:.3f}"
    details["expected_total_distance"] = gt_total_distance

    if not SOL.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing /workdir/sol.csv",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # Read CSV - expect ONLY header + 8 data rows (9 lines total)
    try:
        with open(SOL, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
    except Exception as e:
        details["read_error"] = f"{type(e).__name__}: {e}"
        return GradingResult(
            score=0.0,
            feedback="Failed to read sol.csv",
            subscores=subscores,
            weights=weights,
            details=details
        )
    
    # Remove any empty rows
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    
    if len(rows) < 1:
        return GradingResult(
            score=0.0,
            feedback="CSV is empty",
            subscores=subscores,
            weights=weights,
            details=details
        )
    
    found_header = [cell.strip() for cell in rows[0]]
    data_rows = [[cell.strip() for cell in row] for row in rows[1:]]
    
    # Header check
    if found_header != HEADER:
        details["found_header"] = found_header
        details["expected_header"] = HEADER
        return GradingResult(
            score=0.0,
            feedback=f"Incorrect header. Expected {len(HEADER)} columns, found {len(found_header)}",
            subscores=subscores,
            weights=weights,
            details=details
        )
    
    # Row count check
    if len(data_rows) != 8:
        details["found_row_count"] = len(data_rows)
        return GradingResult(
            score=0.0,
            feedback=f"Expected exactly 8 data rows, found {len(data_rows)}",
            subscores=subscores,
            weights=weights,
            details=details
        )
    
    # Exact match check
    gt_data = gt[1:]  # Skip header
    
    for i, (expected_row, found_row) in enumerate(zip(gt_data, data_rows), 1):
        if len(found_row) != len(expected_row):
            details[f"row_{i}_column_count"] = f"expected {len(expected_row)}, found {len(found_row)}"
            return GradingResult(
                score=0.0,
                feedback=f"Row {i} has wrong number of columns",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        for j, (exp_val, found_val) in enumerate(zip(expected_row, found_row)):
            if exp_val != found_val:
                col_name = HEADER[j]
                details[f"row_{i}_mismatch"] = {
                    "column": col_name,
                    "expected": exp_val,
                    "found": found_val
                }
                return GradingResult(
                    score=0.0,
                    feedback=f"Mismatch at row {i}, column '{col_name}': expected '{exp_val}', found '{found_val}'",
                    subscores=subscores,
                    weights=weights,
                    details=details
                )
    
    # All checks passed
    subscores["exact_match"] = 1.0
    return GradingResult(
        score=1.0,
        feedback="Correct - optimal combination with all constraints satisfied",
        subscores=subscores,
        weights=weights,
        details={"rows_checked": 8, "optimal_score": f"{gt_total_score:.3f}"}
    )