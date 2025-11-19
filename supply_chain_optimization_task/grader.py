#!/usr/bin/env python3
"""
Supply Chain Optimization Task - Grader (Binary Scoring)

Validates demand forecasts and inventory policies against ground truth.

Scoring: Binary (1.0 if ALL checks pass, 0.0 otherwise)

Checks:
1. File existence and format
2. Forecast accuracy (WMAPE < 25%)
3. All 6 hard constraints satisfied
4. Cost computation matches claimed cost
5. Service levels meet requirements
6. Total cost within 1.25× of baseline
7. Simulation runs successfully for 12 weeks
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import csv
import math
import sys

# Fallback GradingResult
try:
    from apex_arena._types import GradingResult
except Exception:
    @dataclass
    class GradingResult:
        score: float
        feedback: str
        subscores: Dict[str, float] = field(default_factory=dict)
        details: Dict[str, Any] = field(default_factory=dict)
        weights: Dict[str, float] = field(default_factory=dict)

DATA_DIR = Path("/workdir/data")
FORECAST_FILE = Path("/workdir/demand_forecast.csv")
POLICY_FILE = Path("/workdir/inventory_policy.json")
ANSWER_FILE = Path("/workdir/ans.txt")

# Cost weights
ALPHA = [1.0, 5.0, 50.0, 3.0, 20.0]  # holding, ordering, stockout, transport, obsolescence

# Service level targets
SERVICE_TARGETS = {
    'A': 0.95,
    'B': 0.90,
    'C': 0.85
}

BUDGET_LIMIT = 5_000_000  # $5M working capital
WMAPE_THRESHOLD = 0.25  # 25% maximum WMAPE
COST_TOLERANCE = 0.01  # 1% tolerance for cost matching
OPTIMALITY_THRESHOLD = 1.25  # Must be within 125% of baseline


def load_data():
    """Load all data files"""
    data = {}
    
    # Load test demand (weeks 105-116) - this is the ground truth
    with open(DATA_DIR / 'test_demand.csv') as f:
        reader = csv.DictReader(f)
        data['test_demand'] = list(reader)
    
    # Load product catalog
    with open(DATA_DIR / 'product_catalog.json') as f:
        data['products'] = json.load(f)['products']
    
    # Load distribution network
    with open(DATA_DIR / 'distribution_network.json') as f:
        data['network'] = json.load(f)
    
    # Load service requirements
    with open(DATA_DIR / 'service_level_requirements.json') as f:
        data['service_reqs'] = json.load(f)
    
    return data


def calculate_wmape(actual_demand: List[Dict], forecast_df: List[Dict], products: List[Dict]) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error
    
    WMAPE = Σ(|actual - forecast| × product_value) / Σ(actual × product_value)
    """
    # Build product cost lookup
    product_costs = {p['product_id']: p['unit_cost'] for p in products}
    
    # Build forecast lookup
    forecast_map = {}
    for row in forecast_df:
        key = (row['product_id'], row['dc_id'], int(row['week']))
        forecast_map[key] = float(row['forecasted_demand'])
    
    total_weighted_error = 0.0
    total_weighted_actual = 0.0
    
    for actual_row in actual_demand:
        key = (actual_row['product_id'], actual_row['dc_id'], int(actual_row['week']))
        
        if key not in forecast_map:
            # Missing forecast - treat as 0
            forecast = 0.0
        else:
            forecast = forecast_map[key]
        
        actual = float(actual_row['units_demanded'])
        cost = product_costs[actual_row['product_id']]
        
        total_weighted_error += abs(actual - forecast) * cost
        total_weighted_actual += actual * cost
    
    if total_weighted_actual == 0:
        return 0.0
    
    return total_weighted_error / total_weighted_actual


def validate_constraints(policies: List[Dict], products: List[Dict], 
                        network: Dict) -> Tuple[bool, List[str]]:
    """Validate all hard constraints"""
    violations = []
    
    # Build product lookup
    product_map = {p['product_id']: p for p in products}
    
    # 1. Check MOQ constraints
    for policy in policies:
        product = product_map[policy['product_id']]
        moq = product['supplier_moq']
        order_qty = policy['order_quantity']
        
        if order_qty > 0 and order_qty < moq:
            violations.append(
                f"{policy['product_id']} at {policy['dc_id']}: "
                f"order_quantity {order_qty} < MOQ {moq}"
            )
    
    # 2. Check capacity constraints (at average inventory level)
    dc_volume = {dc['dc_id']: 0.0 for dc in network['distribution_centers']}
    dc_weight = {dc['dc_id']: 0.0 for dc in network['distribution_centers']}
    
    for policy in policies:
        product = product_map[policy['product_id']]
        dc_id = policy['dc_id']
        avg_inv = policy['average_inventory']
        
        dc_volume[dc_id] += avg_inv * product['volume_cubic_ft']
        dc_weight[dc_id] += avg_inv * product['weight_lbs']
    
    for dc in network['distribution_centers']:
        dc_id = dc['dc_id']
        if dc_volume[dc_id] > dc['volume_capacity_cubic_ft']:
            violations.append(
                f"{dc_id}: volume {dc_volume[dc_id]:.1f} > "
                f"capacity {dc['volume_capacity_cubic_ft']}"
            )
        if dc_weight[dc_id] > dc['weight_capacity_lbs']:
            violations.append(
                f"{dc_id}: weight {dc_weight[dc_id]:.1f} > "
                f"capacity {dc['weight_capacity_lbs']}"
            )
    
    # 3. Check shelf life constraints (perishables)
    for policy in policies:
        product = product_map[policy['product_id']]
        if product['is_perishable']:
            avg_inv = policy['average_inventory']
            # Estimate weekly demand from orders
            weekly_demand = policy['order_quantity'] / 12 * policy['orders_per_12_weeks']
            if weekly_demand > 0:
                turnover = avg_inv / weekly_demand
                if turnover > product['shelf_life_weeks']:
                    violations.append(
                        f"{policy['product_id']} at {policy['dc_id']}: "
                        f"turnover {turnover:.1f} weeks > shelf_life "
                        f"{product['shelf_life_weeks']} weeks"
                    )
    
    # 4. Check budget constraint
    total_inventory_value = 0.0
    for policy in policies:
        product = product_map[policy['product_id']]
        total_inventory_value += policy['average_inventory'] * product['unit_cost']
    
    if total_inventory_value > BUDGET_LIMIT:
        violations.append(
            f"Total inventory value ${total_inventory_value:,.0f} > "
            f"budget ${BUDGET_LIMIT:,.0f}"
        )
    
    return len(violations) == 0, violations


def simulate_12_weeks(policies: List[Dict], test_demand: List[Dict],
                      products: List[Dict], network: Dict) -> Tuple[int, Dict, Dict]:
    """
    Simulate 12 weeks of operations using the inventory policies
    and actual demand (from test set).
    
    Returns: (total_cost, cost_breakdown, service_levels)
    """
    # Build lookups
    product_map = {p['product_id']: p for p in products}
    
    # Build policy lookup: (product, dc) -> policy
    policy_map = {}
    for policy in policies:
        key = (policy['product_id'], policy['dc_id'])
        policy_map[key] = policy
    
    # Build DC lookup
    dc_map = {dc['dc_id']: dc for dc in network['distribution_centers']}
    
    # Build demand lookup: (product, dc, week) -> demand
    demand_map = {}
    for row in test_demand:
        key = (row['product_id'], row['dc_id'], int(row['week']))
        demand_map[key] = float(row['units_demanded'])
    
    # Initialize inventory state
    inventory = {}
    for policy in policies:
        key = (policy['product_id'], policy['dc_id'])
        # Start with safety stock + half order quantity
        inventory[key] = policy['safety_stock'] + policy['order_quantity'] / 2
    
    # Track costs
    holding_cost = 0.0
    ordering_cost = 0.0
    stockout_penalty = 0.0
    transportation_cost = 0.0
    obsolescence_cost = 0.0
    
    # Track service levels
    fulfilled = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    demanded = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    
    # Simulate week by week
    for week in range(105, 117):  # Weeks 105-116
        # Process demand
        for key, policy in policy_map.items():
            product_id, dc_id = key
            product = product_map[product_id]
            category = product['category']
            
            demand_key = (product_id, dc_id, week)
            demand = demand_map.get(demand_key, 0.0)
            
            demanded[category] += demand
            
            # Check inventory
            available = inventory.get(key, 0.0)
            
            if available >= demand:
                # Fulfill demand
                inventory[key] = available - demand
                fulfilled[category] += demand
            else:
                # Partial fulfillment / stockout
                fulfilled[category] += available
                stockout = demand - available
                inventory[key] = 0.0
                
                # Stockout penalty
                penalty_mult = SERVICE_TARGETS[category]  # Use as proxy
                stockout_penalty += stockout * product['unit_cost'] * (penalty_mult * 2)
        
        # Check reorder points and place orders
        for key, policy in policy_map.items():
            product_id, dc_id = key
            product = product_map[product_id]
            
            if inventory[key] <= policy['reorder_point']:
                # Place order
                order_qty = policy['order_quantity']
                ordering_cost += 200  # Fixed cost
                ordering_cost += 0  # Variable cost assumed in unit_cost
                
                # Order arrives after lead time (assume immediate for simplicity)
                inventory[key] += order_qty
        
        # Calculate holding cost for this week
        for key, policy in policy_map.items():
            product_id, dc_id = key
            product = product_map[product_id]
            dc = dc_map[dc_id]
            
            inv_level = inventory[key]
            weekly_holding_rate = dc['holding_cost_rate_annual'] / 52
            holding_cost += inv_level * product['unit_cost'] * weekly_holding_rate
        
        # Check for obsolescence (perishables)
        for key, policy in policy_map.items():
            product_id, dc_id = key
            product = product_map[product_id]
            
            if product['is_perishable']:
                inv_level = inventory[key]
                # Simplified: assume FIFO, and excess inventory expires
                turnover = policy['average_inventory'] / max(1, demand_map.get(
                    (product_id, dc_id, week), 1))
                if turnover > product['shelf_life_weeks']:
                    # Some inventory expires
                    expired = inv_level * 0.05  # 5% expiration rate
                    obsolescence_cost += expired * product['unit_cost']
                    inventory[key] -= expired
    
    # Calculate service levels
    service_levels = {}
    for category in ['A', 'B', 'C']:
        if demanded[category] > 0:
            service_levels[category] = fulfilled[category] / demanded[category]
        else:
            service_levels[category] = 1.0
    
    # Total cost
    total_cost = (ALPHA[0] * holding_cost +
                  ALPHA[1] * ordering_cost +
                  ALPHA[2] * stockout_penalty +
                  ALPHA[3] * transportation_cost +
                  ALPHA[4] * obsolescence_cost)
    
    cost_breakdown = {
        'holding_cost': int(holding_cost),
        'ordering_cost': int(ordering_cost),
        'stockout_penalty': int(stockout_penalty),
        'transportation_cost': int(transportation_cost),
        'obsolescence_cost': int(obsolescence_cost)
    }
    
    return int(total_cost), cost_breakdown, service_levels


def sophisticated_baseline(test_demand: List[Dict], products: List[Dict],
                           network: Dict) -> Tuple[int, List[Dict]]:
    """
    Generate a sophisticated baseline solution for comparison.
    
    Uses:
    - Simple moving average for forecasting
    - Newsvendor model for safety stock
    - EOQ for order quantities
    """
    # Build product map
    product_map = {p['product_id']: p for p in products}
    
    # Calculate average demand per product-DC
    demand_sums = {}
    demand_counts = {}
    
    for row in test_demand:
        key = (row['product_id'], row['dc_id'])
        demand = float(row['units_demanded'])
        demand_sums[key] = demand_sums.get(key, 0.0) + demand
        demand_counts[key] = demand_counts.get(key, 0) + 1
    
    # Generate baseline policies
    baseline_policies = []
    
    for key in demand_sums:
        product_id, dc_id = key
        product = product_map[product_id]
        
        avg_demand = demand_sums[key] / demand_counts[key]
        
        # Safety stock using simple heuristic
        z_scores = {'A': 1.645, 'B': 1.282, 'C': 0.842}
        safety_stock = z_scores[product['category']] * avg_demand * math.sqrt(
            product['base_lead_time_weeks'])
        
        # EOQ
        annual_demand = avg_demand * 52
        fixed_cost = 200
        holding_rate = 0.20  # Average
        
        eoq = math.sqrt((2 * annual_demand * fixed_cost) / 
                        (holding_rate * product['unit_cost']))
        
        order_quantity = max(eoq, product['supplier_moq'])
        
        reorder_point = avg_demand * product['base_lead_time_weeks'] + safety_stock
        
        avg_inventory = safety_stock + order_quantity / 2
        orders_per_12_weeks = max(1, int(12 * avg_demand / order_quantity))
        
        baseline_policies.append({
            'product_id': product_id,
            'dc_id': dc_id,
            'reorder_point': int(reorder_point),
            'order_quantity': int(order_quantity),
            'safety_stock': int(safety_stock),
            'expected_service_level': SERVICE_TARGETS[product['category']],
            'average_inventory': int(avg_inventory),
            'orders_per_12_weeks': orders_per_12_weeks
        })
    
    # Simulate baseline
    baseline_cost, _, _ = simulate_12_weeks(baseline_policies, test_demand,
                                            products, network)
    
    return baseline_cost, baseline_policies


def grade(transcript: str = None) -> GradingResult:
    """
    Grade the submission with binary scoring.
    
    Returns 1.0 if ALL checks pass, 0.0 otherwise.
    """
    subscores = {"all_passes": 0.0}
    weights = {"all_passes": 1.0}
    details = {}
    
    try:
        # Load data
        try:
            data = load_data()
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"Error loading data files: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # CHECK 1: Files exist
        if not FORECAST_FILE.exists():
            return GradingResult(
                score=0.0,
                feedback="FAIL: Missing /workdir/demand_forecast.csv",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        if not POLICY_FILE.exists():
            return GradingResult(
                score=0.0,
                feedback="FAIL: Missing /workdir/inventory_policy.json",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        if not ANSWER_FILE.exists():
            return GradingResult(
                score=0.0,
                feedback="FAIL: Missing /workdir/ans.txt",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # CHECK 2: Load and validate forecast format
        try:
            with open(FORECAST_FILE) as f:
                reader = csv.DictReader(f)
                forecast_df = list(reader)
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Error reading demand_forecast.csv: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        expected_rows = 40 * 8 * 12  # products × DCs × weeks
        if len(forecast_df) != expected_rows:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Expected {expected_rows} forecast rows, got {len(forecast_df)}",
                subscores=subscores,
                weights=weights,
                details={"expected_rows": expected_rows, "actual_rows": len(forecast_df)}
            )
        
        # Validate forecast columns
        required_cols = ['product_id', 'dc_id', 'week', 'forecasted_demand',
                        'prediction_interval_lower', 'prediction_interval_upper']
        if not all(col in forecast_df[0] for col in required_cols):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Missing required columns in forecast file",
                subscores=subscores,
                weights=weights,
                details={"required": required_cols, "found": list(forecast_df[0].keys())}
            )
        
        # CHECK 3: Calculate WMAPE
        try:
            wmape = calculate_wmape(data['test_demand'], forecast_df, data['products'])
            details['wmape'] = wmape
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Error calculating WMAPE: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        if wmape > WMAPE_THRESHOLD:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Forecast WMAPE {wmape:.2%} > threshold {WMAPE_THRESHOLD:.0%}",
                subscores=subscores,
                weights=weights,
                details={"wmape": wmape, "threshold": WMAPE_THRESHOLD}
            )
        
        # CHECK 4: Load and validate policy format
        try:
            with open(POLICY_FILE) as f:
                policy_data = json.load(f)
            policies = policy_data['policies']
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Error reading inventory_policy.json: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        required_fields = ['product_id', 'dc_id', 'reorder_point', 'order_quantity',
                          'safety_stock', 'expected_service_level', 'average_inventory',
                          'orders_per_12_weeks']
        
        for i, policy in enumerate(policies):
            for field in required_fields:
                if field not in policy:
                    return GradingResult(
                        score=0.0,
                        feedback=f"FAIL: Policy {i} missing field '{field}'",
                        subscores=subscores,
                        weights=weights,
                        details={"policy_index": i, "missing_field": field}
                    )
        
        # CHECK 5: Validate all hard constraints
        try:
            constraints_ok, violations = validate_constraints(policies, data['products'],
                                                             data['network'])
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Error validating constraints: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        if not constraints_ok:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Constraint violations: {violations[0]}",
                subscores=subscores,
                weights=weights,
                details={"violations": violations}
            )
        
        # CHECK 6: Simulate 12 weeks
        try:
            actual_cost, cost_breakdown, service_levels = simulate_12_weeks(
                policies, data['test_demand'], data['products'], data['network']
            )
            details['simulated_cost'] = actual_cost
            details['cost_breakdown'] = cost_breakdown
            details['service_levels'] = service_levels
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Simulation error: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # CHECK 7: Verify service levels meet requirements
        for category in ['A', 'B', 'C']:
            achieved = service_levels[category]
            target = SERVICE_TARGETS[category]
            if achieved < target:
                return GradingResult(
                    score=0.0,
                    feedback=f"FAIL: Category {category} service level {achieved:.2%} < target {target:.0%}",
                    subscores=subscores,
                    weights=weights,
                    details={
                        "category": category,
                        "achieved": achieved,
                        "target": target,
                        "service_levels": service_levels
                    }
                )
        
        # CHECK 8: Verify claimed cost matches simulated cost
        try:
            with open(ANSWER_FILE) as f:
                claimed_cost = int(f.read().strip())
            details['claimed_cost'] = claimed_cost
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Error reading ans.txt: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        if abs(actual_cost - claimed_cost) > COST_TOLERANCE * actual_cost:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Claimed cost {claimed_cost:,} != simulated cost {actual_cost:,}",
                subscores=subscores,
                weights=weights,
                details={
                    "claimed_cost": claimed_cost,
                    "simulated_cost": actual_cost,
                    "difference": abs(actual_cost - claimed_cost)
                }
            )
        
        # CHECK 9: Compare to baseline (near-optimality)
        try:
            baseline_cost, _ = sophisticated_baseline(data['test_demand'], data['products'],
                                                      data['network'])
            details['baseline_cost'] = baseline_cost
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Error computing baseline: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        threshold = baseline_cost * OPTIMALITY_THRESHOLD
        if actual_cost > threshold:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Cost {actual_cost:,} > {OPTIMALITY_THRESHOLD:.2f}× baseline {baseline_cost:,} = {threshold:,.0f}",
                subscores=subscores,
                weights=weights,
                details={
                    "actual_cost": actual_cost,
                    "baseline_cost": baseline_cost,
                    "threshold": threshold,
                    "ratio": actual_cost / baseline_cost
                }
            )
        
        # ALL CHECKS PASSED!
        subscores["all_passes"] = 1.0
        
        return GradingResult(
            score=1.0,
            feedback="Correct",
            subscores=subscores,
            weights=weights,
            details={
                "wmape": wmape,
                "simulated_cost": actual_cost,
                "baseline_cost": baseline_cost,
                "cost_ratio": actual_cost / baseline_cost,
                "service_levels": service_levels,
                "cost_breakdown": cost_breakdown
            }
        )
    
    except Exception as e:
        import traceback
        return GradingResult(
            score=0.0,
            feedback=f"UNEXPECTED ERROR: {e}",
            subscores=subscores,
            weights=weights,
            details={"error": str(e), "traceback": traceback.format_exc()}
        )