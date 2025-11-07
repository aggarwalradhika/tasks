#!/usr/bin/env python3
"""
Bike Station Optimization Task - Grader

Validates /workdir/sol.csv and /workdir/network_summary.json against computed ground truth.

Checks:
1. File existence and format
2. Exactly 5 stations selected
3. All 6 constraints satisfied
4. Correct multiplier calculations
5. Correct penalty calculations
6. Network score within acceptable range

Scoring: Binary (1.0 if all checks pass, 0.0 otherwise)
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
    from apex_arena._types import GradingResult  # type: ignore
except Exception:
    @dataclass
    class GradingResult:
        score: float
        feedback: str
        subscores: Dict[str, float] = field(default_factory=dict)
        details: Dict[str, Any] = field(default_factory=dict)
        weights: Dict[str, float] = field(default_factory=dict)

DATA_DIR = Path("/workdir/data")
SUBMISSION_CSV = Path("/workdir/sol.csv")
SUBMISSION_JSON = Path("/workdir/network_summary.json")

REQUIRED_CSV_COLUMNS = [
    'station_id', 'location_name', 'latitude', 'longitude', 'neighborhood',
    'census_tract_id', 'projected_daily_ridership', 'demographic_multiplier',
    'poi_proximity_multiplier', 'network_effect_multiplier', 'weather_adjustment',
    'nearby_existing_count', 'is_high_density', 'near_transit', 'is_isolated'
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in miles"""
    R = 3959  # Earth radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def load_data():
    """Load all JSON data files"""
    data = {}
    
    with open(DATA_DIR / 'existing_stations.json') as f:
        data['existing_stations'] = json.load(f)['existing_stations']
    
    with open(DATA_DIR / 'candidate_locations.json') as f:
        data['candidate_locations'] = json.load(f)['candidate_locations']
    
    with open(DATA_DIR / 'manhattan_demographics.json') as f:
        data['demographics'] = json.load(f)['census_tracts']
    
    with open(DATA_DIR / 'poi_data.json') as f:
        data['pois'] = json.load(f)['points_of_interest']
    
    with open(DATA_DIR / 'weather_ridership.json') as f:
        data['weather'] = json.load(f)['weather_zones']
    
    return data


def calculate_demographic_multiplier(candidate: Dict, demographics: List[Dict]) -> float:
    """Calculate demographic multiplier for a candidate location"""
    tract = next((t for t in demographics if t['census_tract_id'] == candidate['census_tract_id']), None)
    if not tract:
        return 1.0
    
    pop_density_factor = max(0, min(1, (tract['population_density'] - 20000) / 30000))
    commuter_density_factor = max(0, min(1, (tract['commuter_density'] - 5000) / 10000))
    
    return 1.0 + (pop_density_factor * 0.3) + (commuter_density_factor * 0.4)


def calculate_poi_proximity_multiplier(candidate: Dict, pois: List[Dict]) -> float:
    """Calculate POI proximity multiplier"""
    poi_bonus_map = {
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
            total_bonus += poi_bonus_map.get(poi['type'], 0)
    
    return 1.0 + min(0.80, total_bonus)


def calculate_network_effect_multiplier(candidate: Dict, existing_stations: List[Dict]) -> Tuple[float, int]:
    """Calculate network effect multiplier and return count of nearby stations"""
    nearby_count = 0
    for station in existing_stations:
        dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                 station['latitude'], station['longitude'])
        if 0.25 < dist <= 0.5:
            nearby_count += 1
    
    factor = min(nearby_count / 5, 1.0)
    return 1.0 + (factor * 0.2), nearby_count


def calculate_weather_adjustment(candidate: Dict, weather_data: Dict) -> float:
    """Calculate weather adjustment"""
    zone = candidate['weather_zone']
    if zone not in weather_data:
        return 1.0
    
    zone_data = weather_data[zone]
    avg = (zone_data['spring_multiplier'] + zone_data['summer_multiplier'] +
           zone_data['fall_multiplier'] + zone_data['winter_multiplier']) / 4
    return avg


def is_isolated(candidate: Dict, existing_stations: List[Dict]) -> bool:
    """Check if station is isolated (< 2 existing stations within 0.4 miles)"""
    count = 0
    for station in existing_stations:
        dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                 station['latitude'], station['longitude'])
        if dist <= 0.4:
            count += 1
    return count < 2


def is_near_transit(candidate: Dict, pois: List[Dict]) -> bool:
    """Check if within 0.1 miles of subway station"""
    for poi in pois:
        if poi['type'] == 'subway_station':
            dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                     poi['latitude'], poi['longitude'])
            if dist <= 0.1:
                return True
    return False


def is_high_density(candidate: Dict, demographics: List[Dict]) -> bool:
    """Check if in high density census tract"""
    tract = next((t for t in demographics if t['census_tract_id'] == candidate['census_tract_id']), None)
    if not tract:
        return False
    return tract['density_category'] == 'High Density'


def validate_constraints(selected_candidates: List[Dict], data: Dict) -> Tuple[bool, Dict[str, bool], str]:
    """Validate all constraints and return (all_satisfied, constraints_dict, error_message)"""
    constraints = {}
    
    # 1. Coverage constraint: >= 0.25 miles from existing
    min_dist_to_existing = float('inf')
    for candidate in selected_candidates:
        for existing in data['existing_stations']:
            dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                     existing['latitude'], existing['longitude'])
            min_dist_to_existing = min(min_dist_to_existing, dist)
            if dist < 0.25:
                constraints['coverage_constraint'] = False
                break
        if not constraints.get('coverage_constraint', True):
            break
    else:
        constraints['coverage_constraint'] = True
    
    # 2. Spacing constraint: >= 0.15 miles apart
    min_dist_between_new = float('inf')
    for i in range(len(selected_candidates)):
        for j in range(i+1, len(selected_candidates)):
            dist = haversine_distance(selected_candidates[i]['latitude'], selected_candidates[i]['longitude'],
                                     selected_candidates[j]['latitude'], selected_candidates[j]['longitude'])
            min_dist_between_new = min(min_dist_between_new, dist)
            if dist < 0.15:
                constraints['spacing_constraint'] = False
                break
        if not constraints.get('spacing_constraint', True):
            break
    else:
        constraints['spacing_constraint'] = True
    
    # 3. Density constraint: >= 3 high density
    high_density_count = sum(1 for c in selected_candidates 
                            if is_high_density(c, data['demographics']))
    constraints['density_constraint'] = high_density_count >= 3
    
    # 4. Transit integration: >= 2 near subway
    near_transit_count = sum(1 for c in selected_candidates 
                            if is_near_transit(c, data['pois']))
    constraints['transit_integration'] = near_transit_count >= 2
    
    # 5. Network balance: CV <= 0.35 (will be checked after calculating ridership)
    constraints['network_balance'] = True  # Placeholder, checked later
    
    # 6. Geographic diversity: >= 3 neighborhoods
    neighborhoods = set(c['neighborhood'] for c in selected_candidates)
    constraints['geographic_diversity'] = len(neighborhoods) >= 3
    
    all_satisfied = all(constraints.values())
    
    if not all_satisfied:
        failed = [k for k, v in constraints.items() if not v]
        error_msg = f"Constraints not satisfied: {', '.join(failed)}"
    else:
        error_msg = ""
    
    return all_satisfied, constraints, error_msg


def read_submission():
    """Read and validate submission files"""
    if not SUBMISSION_CSV.exists():
        raise FileNotFoundError("Missing /workdir/sol.csv")
    
    if not SUBMISSION_JSON.exists():
        raise FileNotFoundError("Missing /workdir/network_summary.json")
    
    # Read CSV
    with open(SUBMISSION_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        solution_rows = list(reader)
    
    # Read JSON
    with open(SUBMISSION_JSON, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    return solution_rows, summary


def grade(transcript: str = None) -> GradingResult:
    """Grade the submission"""
    subscores = {"validation": 0.0}
    weights = {"validation": 1.0}
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
        
        # Read submission
        try:
            solution_rows, summary = read_submission()
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"Error reading submission files: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # Check exactly 5 rows
        if len(solution_rows) != 5:
            return GradingResult(
                score=0.0,
                feedback=f"Expected exactly 5 stations, found {len(solution_rows)}",
                subscores=subscores,
                weights=weights,
                details={"found_rows": len(solution_rows)}
            )
        
        # Check required columns
        for col in REQUIRED_CSV_COLUMNS:
            if col not in solution_rows[0]:
                return GradingResult(
                    score=0.0,
                    feedback=f"Missing required column: {col}",
                    subscores=subscores,
                    weights=weights,
                    details={"missing_column": col}
                )
        
        # Create candidate lookup
        candidate_map = {c['station_id']: c for c in data['candidate_locations']}
        
        # Validate station IDs exist
        selected_candidates = []
        for row in solution_rows:
            if row['station_id'] not in candidate_map:
                return GradingResult(
                    score=0.0,
                    feedback=f"Invalid station_id: {row['station_id']}",
                    subscores=subscores,
                    weights=weights,
                    details={"invalid_station_id": row['station_id']}
                )
            selected_candidates.append(candidate_map[row['station_id']])
        
        # Validate constraints (initial check, excluding network_balance)
        all_satisfied, constraints, error_msg = validate_constraints(selected_candidates, data)
        
        # Calculate ridership and validate calculations
        total_ridership = 0
        ridership_values = []
        
        for i, row in enumerate(solution_rows):
            candidate = selected_candidates[i]
            
            # Calculate expected values
            expected_demo = calculate_demographic_multiplier(candidate, data['demographics'])
            expected_poi = calculate_poi_proximity_multiplier(candidate, data['pois'])
            expected_network, expected_nearby = calculate_network_effect_multiplier(candidate, data['existing_stations'])
            expected_weather = calculate_weather_adjustment(candidate, data['weather'])
            
            base_ridership = candidate['baseline_demand']
            expected_ridership = (base_ridership * expected_demo * expected_poi * 
                                expected_network * expected_weather)
            
            # Validate submitted values
            try:
                submitted_demo = float(row['demographic_multiplier'])
                submitted_poi = float(row['poi_proximity_multiplier'])
                submitted_network = float(row['network_effect_multiplier'])
                submitted_weather = float(row['weather_adjustment'])
                submitted_ridership = float(row['projected_daily_ridership'])
            except ValueError as e:
                return GradingResult(
                    score=0.0,
                    feedback=f"Invalid numeric value in row {i+1}: {e}",
                    subscores=subscores,
                    weights=weights,
                    details={"row": i+1, "error": str(e)}
                )
            
            # Check multipliers with tolerance
            if abs(submitted_demo - expected_demo) > 0.01:
                return GradingResult(
                    score=0.0,
                    feedback=f"Incorrect demographic_multiplier for {row['station_id']}: expected {expected_demo:.3f}, got {submitted_demo:.3f}",
                    subscores=subscores,
                    weights=weights,
                    details={"station_id": row['station_id'], "expected": expected_demo, "got": submitted_demo}
                )
            
            if abs(submitted_poi - expected_poi) > 0.01:
                return GradingResult(
                    score=0.0,
                    feedback=f"Incorrect poi_proximity_multiplier for {row['station_id']}: expected {expected_poi:.3f}, got {submitted_poi:.3f}",
                    subscores=subscores,
                    weights=weights,
                    details={"station_id": row['station_id'], "expected": expected_poi, "got": submitted_poi}
                )
            
            if abs(submitted_network - expected_network) > 0.01:
                return GradingResult(
                    score=0.0,
                    feedback=f"Incorrect network_effect_multiplier for {row['station_id']}: expected {expected_network:.3f}, got {submitted_network:.3f}",
                    subscores=subscores,
                    weights=weights,
                    details={"station_id": row['station_id'], "expected": expected_network, "got": submitted_network}
                )
            
            if abs(submitted_weather - expected_weather) > 0.01:
                return GradingResult(
                    score=0.0,
                    feedback=f"Incorrect weather_adjustment for {row['station_id']}: expected {expected_weather:.3f}, got {submitted_weather:.3f}",
                    subscores=subscores,
                    weights=weights,
                    details={"station_id": row['station_id'], "expected": expected_weather, "got": submitted_weather}
                )
            
            if abs(submitted_ridership - expected_ridership) > 0.1:
                return GradingResult(
                    score=0.0,
                    feedback=f"Incorrect projected_daily_ridership for {row['station_id']}: expected {expected_ridership:.1f}, got {submitted_ridership:.1f}",
                    subscores=subscores,
                    weights=weights,
                    details={"station_id": row['station_id'], "expected": expected_ridership, "got": submitted_ridership}
                )
            
            total_ridership += expected_ridership
            ridership_values.append(expected_ridership)
        
        # Check network balance constraint
        mean_ridership = sum(ridership_values) / len(ridership_values)
        variance = sum((x - mean_ridership)**2 for x in ridership_values) / len(ridership_values)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_ridership if mean_ridership > 0 else 0
        constraints['network_balance'] = cv <= 0.35
        
        # Re-check all constraints including network_balance
        all_satisfied = all(constraints.values())
        if not all_satisfied:
            failed = [k for k, v in constraints.items() if not v]
            return GradingResult(
                score=0.0,
                feedback=f"Constraints not satisfied: {', '.join(failed)}",
                subscores=subscores,
                weights=weights,
                details={"failed_constraints": failed, "cv": cv}
            )
        
        # Calculate penalties
        spacing_penalty = 0
        for i in range(len(selected_candidates)):
            for j in range(i+1, len(selected_candidates)):
                dist = haversine_distance(selected_candidates[i]['latitude'], selected_candidates[i]['longitude'],
                                         selected_candidates[j]['latitude'], selected_candidates[j]['longitude'])
                if dist < 0.20:
                    spacing_penalty += max(0, (0.20 - dist) * 1000)
        
        imbalance_penalty = 0
        if cv > 0.25:
            imbalance_penalty = (cv - 0.25) * 500
        
        isolation_penalty = 0
        for candidate in selected_candidates:
            if is_isolated(candidate, data['existing_stations']):
                isolation_penalty += 200
        
        expected_network_score = total_ridership - spacing_penalty - imbalance_penalty - isolation_penalty
        
        # Validate summary values
        if abs(summary['total_projected_ridership'] - total_ridership) > 1.0:
            return GradingResult(
                score=0.0,
                feedback=f"Incorrect total_projected_ridership in summary: expected {total_ridership:.1f}, got {summary['total_projected_ridership']:.1f}",
                subscores=subscores,
                weights=weights,
                details={"expected": total_ridership, "got": summary['total_projected_ridership']}
            )
        
        if abs(summary['spacing_penalty'] - spacing_penalty) > 1.0:
            return GradingResult(
                score=0.0,
                feedback=f"Incorrect spacing_penalty in summary: expected {spacing_penalty:.1f}, got {summary['spacing_penalty']:.1f}",
                subscores=subscores,
                weights=weights,
                details={"expected": spacing_penalty, "got": summary['spacing_penalty']}
            )
        
        if abs(summary['imbalance_penalty'] - imbalance_penalty) > 1.0:
            return GradingResult(
                score=0.0,
                feedback=f"Incorrect imbalance_penalty in summary: expected {imbalance_penalty:.1f}, got {summary['imbalance_penalty']:.1f}",
                subscores=subscores,
                weights=weights,
                details={"expected": imbalance_penalty, "got": summary['imbalance_penalty']}
            )
        
        if abs(summary['isolation_penalty'] - isolation_penalty) > 1.0:
            return GradingResult(
                score=0.0,
                feedback=f"Incorrect isolation_penalty in summary: expected {isolation_penalty:.1f}, got {summary['isolation_penalty']:.1f}",
                subscores=subscores,
                weights=weights,
                details={"expected": isolation_penalty, "got": summary['isolation_penalty']}
            )
        
        if abs(summary['network_score'] - expected_network_score) > 1.0:
            return GradingResult(
                score=0.0,
                feedback=f"Incorrect network_score in summary: expected {expected_network_score:.1f}, got {summary['network_score']:.1f}",
                subscores=subscores,
                weights=weights,
                details={"expected": expected_network_score, "got": summary['network_score']}
            )
        
        # Check score is within acceptable range
        OPTIMAL_BENCHMARK = 3700
        if expected_network_score < OPTIMAL_BENCHMARK - 50:
            return GradingResult(
                score=0.0,
                feedback=f"Network score {expected_network_score:.1f} is not within 50 points of optimal solution (>= {OPTIMAL_BENCHMARK - 50})",
                subscores=subscores,
                weights=weights,
                details={"network_score": expected_network_score, "minimum_required": OPTIMAL_BENCHMARK - 50}
            )
        
        # All checks passed
        subscores["validation"] = 1.0
        return GradingResult(
            score=1.0,
            feedback="Correct",
            subscores=subscores,
            weights=weights,
            details={
                "network_score": expected_network_score,
                "total_ridership": total_ridership,
                "constraints_satisfied": constraints
            }
        )
    
    except Exception as e:
        return GradingResult(
            score=0.0,
            feedback=f"Error during grading: {e}",
            subscores=subscores,
            weights=weights,
            details={"error": str(e), "type": type(e).__name__}
        )