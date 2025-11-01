#!/usr/bin/env python3
"""
Grader for Bike Station Optimization Task

This grader validates that the agent has correctly:
1. Selected exactly 5 new bike stations from the candidate locations
2. Satisfied all 6 hard constraints (coverage, spacing, density, transit, balance, diversity)
3. Calculated all multipliers and projected ridership correctly
4. Calculated all penalties correctly
5. Achieved a network score within 50 points of the optimal benchmark

The grader independently recalculates all values to verify correctness and uses
appropriate tolerances for floating-point comparisons.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


class GradingResult:
    """Result of grading with score, feedback, subscores, weights, and details"""
    def __init__(self, score: float, feedback: str, subscores: Dict[str, float] = None,
                 weights: Dict[str, float] = None, details: Dict[str, Any] = None):
        self.score = score
        self.feedback = feedback
        self.subscores = subscores or {}
        self.weights = weights or {}
        self.details = details or {}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in miles.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
        
    Returns:
        Distance in miles
    """
    R = 3959  # Earth radius in miles
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def load_data() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], Dict]:
    """
    Load all required data files.
    
    Returns:
        Tuple of (existing_stations, candidates, demographics, pois, weather)
    """
    data_dir = Path('/workdir/data')
    
    with open(data_dir / 'existing_stations.json') as f:
        existing = json.load(f)['existing_stations']
    
    with open(data_dir / 'candidate_locations.json') as f:
        candidates = json.load(f)['candidate_locations']
    
    with open(data_dir / 'manhattan_demographics.json') as f:
        demographics = json.load(f)['census_tracts']
    
    with open(data_dir / 'poi_data.json') as f:
        pois = json.load(f)['points_of_interest']
    
    with open(data_dir / 'weather_ridership.json') as f:
        weather = json.load(f)['weather_zones']
    
    return existing, candidates, demographics, pois, weather


def calculate_demographic_multiplier(candidate: Dict, demographics: List[Dict]) -> float:
    """
    Calculate demographic multiplier for a candidate location.
    
    Formula:
        demographic_multiplier = 1.0 + (pop_density_factor × 0.3) + (commuter_density_factor × 0.4)
    where:
        pop_density_factor = (tract_pop_density - 20000) / 30000, capped at [0, 1]
        commuter_density_factor = (tract_commuter_density - 5000) / 10000, capped at [0, 1]
    """
    tract = next((t for t in demographics 
                  if t['census_tract_id'] == candidate['census_tract_id']), None)
    if not tract:
        return 1.0
    
    pop_density_factor = max(0, min(1, (tract['population_density'] - 20000) / 30000))
    commuter_density_factor = max(0, min(1, (tract['commuter_density'] - 5000) / 10000))
    
    return 1.0 + (pop_density_factor * 0.3) + (commuter_density_factor * 0.4)


def calculate_poi_proximity_multiplier(candidate: Dict, pois: List[Dict]) -> float:
    """
    Calculate POI proximity multiplier for a candidate location.
    
    Formula:
        poi_proximity_multiplier = 1.0 + Σ(poi_bonus for each POI within 0.15 miles)
        Maximum total bonus: +0.80 (cap at 1.80 multiplier)
    """
    poi_bonus = {
        'office_building': 0.15,
        'subway_station': 0.25,
        'park': 0.10,
        'university': 0.20,
        'hospital': 0.12,
        'retail_hub': 0.18
    }
    
    total_bonus = 0.0
    for poi in pois:
        dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                 poi['latitude'], poi['longitude'])
        if dist <= 0.15:
            total_bonus += poi_bonus.get(poi['type'], 0)
    
    return 1.0 + min(0.80, total_bonus)


def calculate_network_effect_multiplier(candidate: Dict, existing: List[Dict]) -> Tuple[float, int]:
    """
    Calculate network effect multiplier for a candidate location.
    
    Formula:
        network_effect_multiplier = 1.0 + (nearby_existing_stations_factor × 0.2)
    where:
        nearby_existing_stations_factor = min(count / 5, 1.0)
        count = existing stations within 0.5 miles (but > 0.25 miles)
        
    Returns:
        Tuple of (multiplier, count of nearby stations)
    """
    nearby_count = 0
    for station in existing:
        dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                 station['latitude'], station['longitude'])
        if 0.25 < dist <= 0.5:
            nearby_count += 1
    
    factor = min(nearby_count / 5, 1.0)
    return 1.0 + (factor * 0.2), nearby_count


def calculate_weather_adjustment(candidate: Dict, weather: Dict) -> float:
    """
    Calculate weather adjustment for a candidate location.
    
    Formula:
        weather_adjustment = average of seasonal adjustment factors
    """
    zone = candidate['weather_zone']
    if zone not in weather:
        return 1.0
    
    zone_data = weather[zone]
    return (zone_data['spring_multiplier'] + zone_data['summer_multiplier'] +
            zone_data['fall_multiplier'] + zone_data['winter_multiplier']) / 4


def is_high_density(candidate: Dict, demographics: List[Dict]) -> bool:
    """Check if candidate is in a high density census tract."""
    tract = next((t for t in demographics 
                  if t['census_tract_id'] == candidate['census_tract_id']), None)
    return tract and tract['density_category'] == 'High Density'


def is_near_transit(candidate: Dict, pois: List[Dict]) -> bool:
    """Check if candidate is within 0.1 miles of a subway station."""
    for poi in pois:
        if poi['type'] == 'subway_station':
            dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                     poi['latitude'], poi['longitude'])
            if dist <= 0.1:
                return True
    return False


def is_isolated(candidate: Dict, existing: List[Dict]) -> bool:
    """Check if candidate has < 2 existing stations within 0.4 miles."""
    count = 0
    for station in existing:
        dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                 station['latitude'], station['longitude'])
        if dist <= 0.4:
            count += 1
    return count < 2


def check_constraints(selection: List[Dict], existing: List[Dict],
                     demographics: List[Dict], pois: List[Dict]) -> Tuple[bool, Dict[str, bool], str]:
    """
    Check all 6 hard constraints for the selected stations.
    
    Returns:
        Tuple of (all_satisfied, constraints_dict, error_message)
    """
    constraints = {}
    errors = []
    
    # 1. Coverage Constraint: Each new station ≥ 0.25 miles from existing
    min_dist_to_existing = float('inf')
    for candidate in selection:
        for station in existing:
            dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                     station['latitude'], station['longitude'])
            min_dist_to_existing = min(min_dist_to_existing, dist)
    
    constraints['coverage_constraint'] = min_dist_to_existing >= 0.25
    if not constraints['coverage_constraint']:
        errors.append(f"Coverage constraint failed: minimum distance to existing station is "
                     f"{min_dist_to_existing:.4f} miles (required ≥ 0.25)")
    
    # 2. Spacing Constraint: New stations ≥ 0.15 miles apart
    min_dist_between = float('inf')
    for i in range(len(selection)):
        for j in range(i+1, len(selection)):
            dist = haversine_distance(selection[i]['latitude'], selection[i]['longitude'],
                                     selection[j]['latitude'], selection[j]['longitude'])
            min_dist_between = min(min_dist_between, dist)
    
    constraints['spacing_constraint'] = min_dist_between >= 0.15
    if not constraints['spacing_constraint']:
        errors.append(f"Spacing constraint failed: minimum distance between new stations is "
                     f"{min_dist_between:.4f} miles (required ≥ 0.15)")
    
    # 3. Density Constraint: At least 3 in "High Density" census tracts
    high_density_count = sum(1 for c in selection if is_high_density(c, demographics))
    constraints['density_constraint'] = high_density_count >= 3
    if not constraints['density_constraint']:
        errors.append(f"Density constraint failed: only {high_density_count} stations in "
                     f"high density areas (required ≥ 3)")
    
    # 4. Transit Integration: At least 2 within 0.1 miles of subway_station POI
    near_transit_count = sum(1 for c in selection if is_near_transit(c, pois))
    constraints['transit_integration'] = near_transit_count >= 2
    if not constraints['transit_integration']:
        errors.append(f"Transit integration failed: only {near_transit_count} stations near "
                     f"transit (required ≥ 2)")
    
    # 5. Network Balance: CV ≤ 0.35 (checked separately with ridership values)
    constraints['network_balance'] = True  # Placeholder, checked in main grading
    
    # 6. Geographic Diversity: At least 3 different neighborhoods
    neighborhoods = set(c['neighborhood'] for c in selection)
    constraints['geographic_diversity'] = len(neighborhoods) >= 3
    if not constraints['geographic_diversity']:
        errors.append(f"Geographic diversity failed: only {len(neighborhoods)} neighborhoods "
                     f"(required ≥ 3): {list(neighborhoods)}")
    
    error_message = "; ".join(errors) if errors else ""
    return all(constraints.values()), constraints, error_message


def calculate_spacing_penalty(selection: List[Dict]) -> float:
    """
    Calculate spacing penalty for stations that are too close.
    
    Formula:
        spacing_penalty = Σ max(0, (0.20 - distance) × 1000)
        for all pairs where distance < 0.20 miles
    """
    penalty = 0.0
    for i in range(len(selection)):
        for j in range(i+1, len(selection)):
            dist = haversine_distance(selection[i]['latitude'], selection[i]['longitude'],
                                     selection[j]['latitude'], selection[j]['longitude'])
            if dist < 0.20:
                penalty += max(0, (0.20 - dist) * 1000)
    return penalty


def calculate_imbalance_penalty(ridership_values: List[float]) -> Tuple[float, float]:
    """
    Calculate imbalance penalty based on coefficient of variation.
    
    Formula:
        imbalance_penalty = (CV - 0.25) × 500 if CV > 0.25, else 0
        where CV = standard_deviation / mean
        
    Returns:
        Tuple of (penalty, coefficient_of_variation)
    """
    if not ridership_values:
        return 0.0, 0.0
    
    mean = sum(ridership_values) / len(ridership_values)
    if mean == 0:
        return 0.0, 0.0
    
    variance = sum((x - mean)**2 for x in ridership_values) / len(ridership_values)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean
    
    penalty = max(0, (cv - 0.25) * 500)
    return penalty, cv


def calculate_isolation_penalty(selection: List[Dict], existing: List[Dict]) -> Tuple[float, int]:
    """
    Calculate isolation penalty for stations with poor network connectivity.
    
    Formula:
        isolation_penalty = isolated_station_count × 200
        where isolated = < 2 existing stations within 0.4 miles
        
    Returns:
        Tuple of (penalty, count of isolated stations)
    """
    isolated_count = sum(1 for c in selection if is_isolated(c, existing))
    return isolated_count * 200, isolated_count


def grade() -> GradingResult:
    """
    Main grading function that validates the solution and returns a score.
    
    Returns:
        GradingResult with score (1.0 or 0.0) and detailed feedback
    """
    # Check if solution.json exists
    solution_path = Path('/workdir/solution.json')
    
    if not solution_path.exists():
        return GradingResult(
            score=0.0,
            feedback="FAIL: solution.json not found at /workdir/solution.json",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": "solution.json file missing"}
        )
    
    # Load data files
    try:
        existing, candidates, demographics, pois, weather = load_data()
    except Exception as e:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Error loading data files: {str(e)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Data loading error: {str(e)}"}
        )
    
    # Create lookup dictionaries
    candidates_dict = {c['station_id']: c for c in candidates}
    
    # Parse solution.json
    try:
        with open(solution_path, 'r') as f:
            solution = json.load(f)
    except Exception as e:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Error reading solution.json: {str(e)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"JSON parsing error: {str(e)}"}
        )
    
    # Validate solution structure
    if 'selected_stations' not in solution:
        return GradingResult(
            score=0.0,
            feedback="FAIL: solution.json missing 'selected_stations' field",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": "Missing 'selected_stations' field"}
        )
    
    if 'network_summary' not in solution:
        return GradingResult(
            score=0.0,
            feedback="FAIL: solution.json missing 'network_summary' field",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": "Missing 'network_summary' field"}
        )
    
    rows = solution['selected_stations']
    summary = solution['network_summary']
    
    # Check exactly 5 stations
    if len(rows) != 5:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: selected_stations must have exactly 5 stations, found {len(rows)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Wrong number of stations: {len(rows)} instead of 5"}
        )
    
    # Validate required fields in each station
    required_fields = [
        'station_id', 'location_name', 'latitude', 'longitude', 'neighborhood',
        'census_tract_id', 'projected_daily_ridership', 'demographic_multiplier',
        'poi_proximity_multiplier', 'network_effect_multiplier', 'weather_adjustment',
        'nearby_existing_count', 'is_high_density', 'near_transit', 'is_isolated'
    ]
    
    for i, row in enumerate(rows):
        missing_fields = [field for field in required_fields if field not in row]
        if missing_fields:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Station {i+1} missing required fields: {missing_fields}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Missing fields in station {i+1}: {missing_fields}"}
            )
    
    # Validate station IDs and build selection
    selection = []
    for row in rows:
        station_id = row['station_id']
        if station_id not in candidates_dict:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Invalid station_id '{station_id}' not found in candidate_locations.json",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid station_id: {station_id}"}
            )
        selection.append(candidates_dict[station_id])
    
    # Check for duplicate stations
    if len(set(row['station_id'] for row in rows)) != 5:
        return GradingResult(
            score=0.0,
            feedback="FAIL: Duplicate station_ids found in selected_stations",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": "Duplicate station IDs in solution"}
        )
    
    # Validate network_summary structure
    required_summary_fields = [
        'total_projected_ridership', 'spacing_penalty', 'imbalance_penalty',
        'isolation_penalty', 'network_score', 'ridership_coefficient_of_variation',
        'constraints_satisfied', 'min_distance_between_new_stations',
        'min_distance_to_existing', 'neighborhoods_represented'
    ]
    
    missing_fields = [f for f in required_summary_fields if f not in summary]
    if missing_fields:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Missing required fields in network_summary: {missing_fields}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Missing summary fields: {missing_fields}"}
        )
    
    # Verify constraints
    constraints_valid, constraints_dict, constraint_error = check_constraints(
        selection, existing, demographics, pois
    )
    
    # Verify and calculate all values for each station
    tolerance_multiplier = 0.01  # Tolerance for multiplier calculations
    tolerance_ridership = 0.1    # Tolerance for ridership calculations
    
    calculated_ridership = []
    validation_details = {
        "selected_stations": [s['station_id'] for s in selection],
        "station_validations": []
    }
    
    for i, row in enumerate(rows):
        candidate = selection[i]
        station_detail = {"station_id": candidate['station_id']}
        
        # Calculate expected values
        expected_demo_mult = calculate_demographic_multiplier(candidate, demographics)
        expected_poi_mult = calculate_poi_proximity_multiplier(candidate, pois)
        expected_network_mult, expected_nearby_count = calculate_network_effect_multiplier(
            candidate, existing
        )
        expected_weather_adj = calculate_weather_adjustment(candidate, weather)
        
        expected_ridership = (
            candidate['baseline_demand'] * 
            expected_demo_mult * 
            expected_poi_mult * 
            expected_network_mult * 
            expected_weather_adj
        )
        
        calculated_ridership.append(expected_ridership)
        station_detail["expected_ridership"] = round(expected_ridership, 1)
        
        # Verify demographic multiplier
        try:
            actual_demo_mult = float(row['demographic_multiplier'])
            if abs(actual_demo_mult - expected_demo_mult) > tolerance_multiplier:
                return GradingResult(
                    score=0.0,
                    feedback=f"FAIL: Incorrect demographic_multiplier for {candidate['station_id']}: "
                            f"expected {expected_demo_mult:.3f}, got {actual_demo_mult:.3f}",
                    subscores={"all_passes": 0.0},
                    weights={"all_passes": 1.0},
                    details={"error": f"Wrong demographic_multiplier for {candidate['station_id']}",
                            "expected": expected_demo_mult, "actual": actual_demo_mult}
                )
        except (ValueError, TypeError):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Invalid demographic_multiplier value for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid demographic_multiplier format for {candidate['station_id']}"}
            )
        
        # Verify POI proximity multiplier
        try:
            actual_poi_mult = float(row['poi_proximity_multiplier'])
            if abs(actual_poi_mult - expected_poi_mult) > tolerance_multiplier:
                return GradingResult(
                    score=0.0,
                    feedback=f"FAIL: Incorrect poi_proximity_multiplier for {candidate['station_id']}: "
                            f"expected {expected_poi_mult:.3f}, got {actual_poi_mult:.3f}",
                    subscores={"all_passes": 0.0},
                    weights={"all_passes": 1.0},
                    details={"error": f"Wrong poi_proximity_multiplier for {candidate['station_id']}",
                            "expected": expected_poi_mult, "actual": actual_poi_mult}
                )
        except (ValueError, TypeError):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Invalid poi_proximity_multiplier value for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid poi_proximity_multiplier format for {candidate['station_id']}"}
            )
        
        # Verify network effect multiplier
        try:
            actual_network_mult = float(row['network_effect_multiplier'])
            if abs(actual_network_mult - expected_network_mult) > tolerance_multiplier:
                return GradingResult(
                    score=0.0,
                    feedback=f"FAIL: Incorrect network_effect_multiplier for {candidate['station_id']}: "
                            f"expected {expected_network_mult:.3f}, got {actual_network_mult:.3f}",
                    subscores={"all_passes": 0.0},
                    weights={"all_passes": 1.0},
                    details={"error": f"Wrong network_effect_multiplier for {candidate['station_id']}",
                            "expected": expected_network_mult, "actual": actual_network_mult}
                )
        except (ValueError, TypeError):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Invalid network_effect_multiplier value for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid network_effect_multiplier format for {candidate['station_id']}"}
            )
        
        # Verify weather adjustment
        try:
            actual_weather_adj = float(row['weather_adjustment'])
            if abs(actual_weather_adj - expected_weather_adj) > tolerance_multiplier:
                return GradingResult(
                    score=0.0,
                    feedback=f"FAIL: Incorrect weather_adjustment for {candidate['station_id']}: "
                            f"expected {expected_weather_adj:.3f}, got {actual_weather_adj:.3f}",
                    subscores={"all_passes": 0.0},
                    weights={"all_passes": 1.0},
                    details={"error": f"Wrong weather_adjustment for {candidate['station_id']}",
                            "expected": expected_weather_adj, "actual": actual_weather_adj}
                )
        except (ValueError, TypeError):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Invalid weather_adjustment value for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid weather_adjustment format for {candidate['station_id']}"}
            )
        
        # Verify nearby existing count
        try:
            actual_nearby_count = int(row['nearby_existing_count'])
            if actual_nearby_count != expected_nearby_count:
                return GradingResult(
                    score=0.0,
                    feedback=f"FAIL: Incorrect nearby_existing_count for {candidate['station_id']}: "
                            f"expected {expected_nearby_count}, got {actual_nearby_count}",
                    subscores={"all_passes": 0.0},
                    weights={"all_passes": 1.0},
                    details={"error": f"Wrong nearby_existing_count for {candidate['station_id']}",
                            "expected": expected_nearby_count, "actual": actual_nearby_count}
                )
        except (ValueError, TypeError):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Invalid nearby_existing_count value for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid nearby_existing_count format for {candidate['station_id']}"}
            )
        
        # Verify projected ridership
        try:
            actual_ridership = float(row['projected_daily_ridership'])
            if abs(actual_ridership - expected_ridership) > tolerance_ridership:
                return GradingResult(
                    score=0.0,
                    feedback=f"FAIL: Incorrect projected_daily_ridership for {candidate['station_id']}: "
                            f"expected {expected_ridership:.1f}, got {actual_ridership:.1f}",
                    subscores={"all_passes": 0.0},
                    weights={"all_passes": 1.0},
                    details={"error": f"Wrong projected_daily_ridership for {candidate['station_id']}",
                            "expected": expected_ridership, "actual": actual_ridership}
                )
        except (ValueError, TypeError):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Invalid projected_daily_ridership value for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid projected_daily_ridership format for {candidate['station_id']}"}
            )
        
        # Verify boolean fields
        expected_high_density = is_high_density(candidate, demographics)
        if not isinstance(row['is_high_density'], bool):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: is_high_density must be boolean for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid is_high_density format for {candidate['station_id']}"}
            )
        if row['is_high_density'] != expected_high_density:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect is_high_density for {candidate['station_id']}: "
                        f"expected {expected_high_density}, got {row['is_high_density']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Wrong is_high_density for {candidate['station_id']}",
                        "expected": expected_high_density, "actual": row['is_high_density']}
            )
        
        expected_near_transit = is_near_transit(candidate, pois)
        if not isinstance(row['near_transit'], bool):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: near_transit must be boolean for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid near_transit format for {candidate['station_id']}"}
            )
        if row['near_transit'] != expected_near_transit:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect near_transit for {candidate['station_id']}: "
                        f"expected {expected_near_transit}, got {row['near_transit']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Wrong near_transit for {candidate['station_id']}",
                        "expected": expected_near_transit, "actual": row['near_transit']}
            )
        
        expected_isolated = is_isolated(candidate, existing)
        if not isinstance(row['is_isolated'], bool):
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: is_isolated must be boolean for {candidate['station_id']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Invalid is_isolated format for {candidate['station_id']}"}
            )
        if row['is_isolated'] != expected_isolated:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect is_isolated for {candidate['station_id']}: "
                        f"expected {expected_isolated}, got {row['is_isolated']}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": f"Wrong is_isolated for {candidate['station_id']}",
                        "expected": expected_isolated, "actual": row['is_isolated']}
            )
        
        station_detail["validations_passed"] = True
        validation_details["station_validations"].append(station_detail)
    
    # Calculate expected penalties and network score
    tolerance_penalty = 1.0  # Tolerance for penalty calculations
    
    expected_total_ridership = sum(calculated_ridership)
    expected_spacing_penalty = calculate_spacing_penalty(selection)
    expected_imbalance_penalty, expected_cv = calculate_imbalance_penalty(calculated_ridership)
    expected_isolation_penalty, isolated_count = calculate_isolation_penalty(selection, existing)
    
    expected_network_score = (
        expected_total_ridership - 
        expected_spacing_penalty - 
        expected_imbalance_penalty - 
        expected_isolation_penalty
    )
    
    # Add calculation details
    validation_details["expected_metrics"] = {
        "total_ridership": round(expected_total_ridership, 1),
        "spacing_penalty": round(expected_spacing_penalty, 1),
        "imbalance_penalty": round(expected_imbalance_penalty, 1),
        "isolation_penalty": round(expected_isolation_penalty, 1),
        "network_score": round(expected_network_score, 1),
        "cv": round(expected_cv, 3)
    }
    
    # Check network balance constraint with CV
    constraints_dict['network_balance'] = expected_cv <= 0.35
    if not constraints_dict['network_balance']:
        constraint_error = (f"Network balance constraint failed: CV = {expected_cv:.4f} "
                          f"(required ≤ 0.35)")
        constraints_valid = False
    
    # Verify total projected ridership
    try:
        actual_total = float(summary['total_projected_ridership'])
        if abs(actual_total - expected_total_ridership) > tolerance_ridership:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect total_projected_ridership: expected "
                        f"{expected_total_ridership:.1f}, got {actual_total:.1f}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": "Wrong total_projected_ridership",
                        "expected": expected_total_ridership, "actual": actual_total}
            )
    except (ValueError, KeyError, TypeError) as e:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Invalid total_projected_ridership: {str(e)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Invalid total_projected_ridership: {str(e)}"}
        )
    
    # Verify spacing penalty
    try:
        actual_spacing = float(summary['spacing_penalty'])
        if abs(actual_spacing - expected_spacing_penalty) > tolerance_penalty:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect spacing_penalty: expected {expected_spacing_penalty:.1f}, "
                        f"got {actual_spacing:.1f}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": "Wrong spacing_penalty",
                        "expected": expected_spacing_penalty, "actual": actual_spacing}
            )
    except (ValueError, KeyError, TypeError) as e:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Invalid spacing_penalty: {str(e)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Invalid spacing_penalty: {str(e)}"}
        )
    
    # Verify imbalance penalty
    try:
        actual_imbalance = float(summary['imbalance_penalty'])
        if abs(actual_imbalance - expected_imbalance_penalty) > tolerance_penalty:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect imbalance_penalty: expected {expected_imbalance_penalty:.1f}, "
                        f"got {actual_imbalance:.1f}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": "Wrong imbalance_penalty",
                        "expected": expected_imbalance_penalty, "actual": actual_imbalance}
            )
    except (ValueError, KeyError, TypeError) as e:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Invalid imbalance_penalty: {str(e)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Invalid imbalance_penalty: {str(e)}"}
        )
    
    # Verify isolation penalty
    try:
        actual_isolation = float(summary['isolation_penalty'])
        if abs(actual_isolation - expected_isolation_penalty) > tolerance_penalty:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect isolation_penalty: expected {expected_isolation_penalty:.1f}, "
                        f"got {actual_isolation:.1f}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": "Wrong isolation_penalty",
                        "expected": expected_isolation_penalty, "actual": actual_isolation}
            )
    except (ValueError, KeyError, TypeError) as e:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Invalid isolation_penalty: {str(e)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Invalid isolation_penalty: {str(e)}"}
        )
    
    # Verify network score
    try:
        actual_score = float(summary['network_score'])
        if abs(actual_score - expected_network_score) > tolerance_penalty:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect network_score: expected {expected_network_score:.1f}, "
                        f"got {actual_score:.1f}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": "Wrong network_score",
                        "expected": expected_network_score, "actual": actual_score}
            )
    except (ValueError, KeyError, TypeError) as e:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Invalid network_score: {str(e)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Invalid network_score: {str(e)}"}
        )
    
    # Verify CV
    try:
        actual_cv = float(summary['ridership_coefficient_of_variation'])
        if abs(actual_cv - expected_cv) > tolerance_multiplier:
            return GradingResult(
                score=0.0,
                feedback=f"FAIL: Incorrect ridership_coefficient_of_variation: expected "
                        f"{expected_cv:.3f}, got {actual_cv:.3f}",
                subscores={"all_passes": 0.0},
                weights={"all_passes": 1.0},
                details={"error": "Wrong ridership_coefficient_of_variation",
                        "expected": expected_cv, "actual": actual_cv}
            )
    except (ValueError, KeyError, TypeError) as e:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Invalid ridership_coefficient_of_variation: {str(e)}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": f"Invalid ridership_coefficient_of_variation: {str(e)}"}
        )
    
    # Check if all constraints are satisfied
    if not constraints_valid:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Constraints not satisfied. {constraint_error}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": constraint_error,
                    "constraints": constraints_dict}
        )
    
    # Verify constraints in summary match
    if summary['constraints_satisfied'] != constraints_dict:
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: constraints_satisfied in network_summary does not match actual "
                    f"constraint status. Expected: {constraints_dict}, Got: {summary['constraints_satisfied']}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": "Constraint mismatch",
                    "expected": constraints_dict,
                    "actual": summary['constraints_satisfied']}
        )
    
    # Check network score quality
    # The optimal benchmark is the score that can be achieved with the best possible selection
    # Using 50-point tolerance as specified in the task
    OPTIMAL_BENCHMARK = 3700.0
    SCORE_TOLERANCE = 50.0
    
    if expected_network_score < (OPTIMAL_BENCHMARK - SCORE_TOLERANCE):
        return GradingResult(
            score=0.0,
            feedback=f"FAIL: Network score {expected_network_score:.1f} is too low. "
                    f"Required: ≥ {OPTIMAL_BENCHMARK - SCORE_TOLERANCE:.1f}",
            subscores={"all_passes": 0.0},
            weights={"all_passes": 1.0},
            details={"error": "Network score too low",
                    "network_score": expected_network_score,
                    "minimum_required": OPTIMAL_BENCHMARK - SCORE_TOLERANCE}
        )
    
    # All checks passed
    validation_details["constraints"] = constraints_dict
    validation_details["actual_metrics"] = {
        "total_ridership": round(actual_total, 1),
        "spacing_penalty": round(actual_spacing, 1),
        "imbalance_penalty": round(actual_imbalance, 1),
        "isolation_penalty": round(actual_isolation, 1),
        "network_score": round(actual_score, 1),
        "cv": round(actual_cv, 3)
    }
    validation_details["score_quality"] = {
        "optimal_benchmark": OPTIMAL_BENCHMARK,
        "score_tolerance": SCORE_TOLERANCE,
        "minimum_required": OPTIMAL_BENCHMARK - SCORE_TOLERANCE,
        "achieved": expected_network_score,
        "margin": expected_network_score - (OPTIMAL_BENCHMARK - SCORE_TOLERANCE)
    }
    
    feedback = (
        f"PASS: All checks passed!\n"
        f"Network Score: {expected_network_score:.2f}\n"
        f"Total Ridership: {expected_total_ridership:.1f}\n"
        f"Spacing Penalty: {expected_spacing_penalty:.1f}\n"
        f"Imbalance Penalty: {expected_imbalance_penalty:.1f}\n"
        f"Isolation Penalty: {expected_isolation_penalty:.1f}\n"
        f"CV: {expected_cv:.3f}\n"
        f"Constraints: All satisfied\n"
        f"Selected Stations: {[s['station_id'] for s in selection]}"
    )
    
    return GradingResult(
        score=1.0,
        feedback=feedback,
        subscores={"all_passes": 1.0},
        weights={"all_passes": 1.0},
        details=validation_details
    )


def main():
    """Main entry point for the grader."""
    try:
        result = grade()
        print(result.feedback)
        return 0 if result.score == 1.0 else 1
    except Exception as e:
        print(f"FAIL: Grader error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())