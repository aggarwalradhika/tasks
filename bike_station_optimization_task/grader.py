from pathlib import Path
import pandas as pd
import json
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

# ---- GradingResult with fallback -----------------
try:
    from apex_arena._types import GradingResult  # type: ignore
except Exception:
    @dataclass
    class GradingResult:
        """Fallback grading result when Apex's type is unavailable."""
        score: float
        feedback: str
        subscores: Dict[str, float] = field(default_factory=dict)
        details: Dict[str, Any] = field(default_factory=dict)
        weights: Dict[str, float] = field(default_factory=dict)


def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in miles."""
    R = 3959
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def load_all_data():
    """Load all data files."""
    with open('/workdir/data/existing_stations.json', 'r') as f:
        existing = json.load(f)['existing_stations']
    
    with open('/workdir/data/candidate_locations.json', 'r') as f:
        candidates = json.load(f)['candidate_locations']
    
    with open('/workdir/data/manhattan_demographics.json', 'r') as f:
        demographics = json.load(f)['census_tracts']
    
    with open('/workdir/data/poi_data.json', 'r') as f:
        pois = json.load(f)['points_of_interest']
    
    with open('/workdir/data/weather_ridership.json', 'r') as f:
        weather = json.load(f)['weather_zones']
    
    # Create lookup dictionaries
    candidate_dict = {c['station_id']: c for c in candidates}
    demo_dict = {d['census_tract_id']: d for d in demographics}
    
    return {
        'existing': existing,
        'candidates': candidate_dict,
        'demographics': demo_dict,
        'pois': pois,
        'weather': weather
    }


def calculate_demographic_multiplier(candidate, demo_dict):
    """Calculate demographic multiplier for a candidate."""
    tract = demo_dict.get(candidate['census_tract_id'])
    if not tract:
        return 1.0
    
    pop_density = tract['population_density']
    commuter_density = tract['commuter_density']
    
    pop_factor = max(0, min(1, (pop_density - 20000) / 30000))
    commuter_factor = max(0, min(1, (commuter_density - 5000) / 10000))
    
    return 1.0 + (pop_factor * 0.3) + (commuter_factor * 0.4)


def calculate_poi_multiplier(candidate, pois):
    """Calculate POI proximity multiplier."""
    poi_bonuses = {
        'office_building': 0.15,
        'subway_station': 0.25,
        'park': 0.10,
        'university': 0.20,
        'hospital': 0.12,
        'retail_hub': 0.18
    }
    
    total_bonus = 0.0
    for poi in pois:
        dist = haversine_miles(
            candidate['latitude'], candidate['longitude'],
            poi['latitude'], poi['longitude']
        )
        if dist <= 0.15:
            total_bonus += poi_bonuses.get(poi['type'], 0)
    
    total_bonus = min(total_bonus, 0.80)
    return 1.0 + total_bonus


def calculate_network_effect_multiplier(candidate, existing_stations):
    """Calculate network effect multiplier."""
    count = 0
    for station in existing_stations:
        dist = haversine_miles(
            candidate['latitude'], candidate['longitude'],
            station['latitude'], station['longitude']
        )
        if 0.25 < dist <= 0.5:
            count += 1
    
    factor = min(count / 5, 1.0)
    return 1.0 + (factor * 0.2)


def calculate_weather_adjustment(candidate, weather_zones):
    """Calculate weather adjustment."""
    zone_data = weather_zones.get(candidate['weather_zone'])
    if not zone_data:
        return 1.0
    
    avg = (zone_data['spring_multiplier'] + 
           zone_data['summer_multiplier'] + 
           zone_data['fall_multiplier'] + 
           zone_data['winter_multiplier']) / 4
    return avg


def calculate_projected_ridership(candidate, data):
    """Calculate projected daily ridership for a candidate."""
    demo_mult = calculate_demographic_multiplier(candidate, data['demographics'])
    poi_mult = calculate_poi_multiplier(candidate, data['pois'])
    network_mult = calculate_network_effect_multiplier(candidate, data['existing'])
    weather_adj = calculate_weather_adjustment(candidate, data['weather'])
    
    base = candidate['baseline_demand']
    projected = base * demo_mult * poi_mult * network_mult * weather_adj
    
    return {
        'projected_ridership': projected,
        'demographic_multiplier': demo_mult,
        'poi_proximity_multiplier': poi_mult,
        'network_effect_multiplier': network_mult,
        'weather_adjustment': weather_adj
    }


def check_constraints(solution_df, data):
    """Check all constraints."""
    constraints = {
        'coverage_constraint': True,
        'spacing_constraint': True,
        'density_constraint': True,
        'transit_integration': True,
        'network_balance': True,
        'geographic_diversity': True
    }
    
    details = {}
    
    # Coverage constraint
    min_dist_existing = float('inf')
    for _, row in solution_df.iterrows():
        for station in data['existing']:
            dist = haversine_miles(
                row['latitude'], row['longitude'],
                station['latitude'], station['longitude']
            )
            min_dist_existing = min(min_dist_existing, dist)
            if dist < 0.25:
                constraints['coverage_constraint'] = False
                details['coverage_violation'] = f"Station {row['station_id']} too close to existing"
                break
    
    # Spacing constraint
    min_dist_new = float('inf')
    coords = solution_df[['latitude', 'longitude']].values
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = haversine_miles(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            min_dist_new = min(min_dist_new, dist)
            if dist < 0.15:
                constraints['spacing_constraint'] = False
                details['spacing_violation'] = f"Stations too close: {dist:.4f} miles"
                break
    
    # Density constraint
    high_density_count = sum(1 for _, row in solution_df.iterrows() if row['is_high_density'] == True)
    if high_density_count < 3:
        constraints['density_constraint'] = False
        details['density_violation'] = f"Only {high_density_count} high density stations"
    
    # Transit integration
    near_transit_count = sum(1 for _, row in solution_df.iterrows() if row['near_transit'] == True)
    if near_transit_count < 2:
        constraints['transit_integration'] = False
        details['transit_violation'] = f"Only {near_transit_count} stations near transit"
    
    # Network balance
    ridership_values = solution_df['projected_daily_ridership'].values
    mean_ridership = ridership_values.mean()
    std_ridership = ridership_values.std()
    cv = std_ridership / mean_ridership if mean_ridership > 0 else 0
    if cv > 0.35:
        constraints['network_balance'] = False
        details['balance_violation'] = f"CV = {cv:.3f} > 0.35"
    
    # Geographic diversity
    unique_neighborhoods = solution_df['neighborhood'].nunique()
    if unique_neighborhoods < 3:
        constraints['geographic_diversity'] = False
        details['diversity_violation'] = f"Only {unique_neighborhoods} neighborhoods"
    
    details['min_dist_to_existing'] = min_dist_existing
    details['min_dist_between_new'] = min_dist_new
    details['cv'] = cv
    
    return constraints, details


def calculate_penalties(solution_df, data):
    """Calculate all penalties."""
    # Spacing penalty
    spacing_penalty = 0.0
    coords = solution_df[['latitude', 'longitude']].values
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = haversine_miles(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if dist < 0.20:
                spacing_penalty += max(0, (0.20 - dist) * 1000)
    
    # Imbalance penalty
    ridership_values = solution_df['projected_daily_ridership'].values
    mean_ridership = ridership_values.mean()
    std_ridership = ridership_values.std()
    cv = std_ridership / mean_ridership if mean_ridership > 0 else 0
    imbalance_penalty = max(0, (cv - 0.25) * 500)
    
    # Isolation penalty
    isolated_count = sum(1 for _, row in solution_df.iterrows() if row['is_isolated'] == True)
    isolation_penalty = isolated_count * 200
    
    return spacing_penalty, imbalance_penalty, isolation_penalty, cv


def grade(transcript: str) -> GradingResult:
    """Grade the NYC Bike Station Network Optimization task."""
    weights = {"all_passes": 1.0}
    subscores = {"all_passes": 0.0}
    details = {}
    
    sol_path = Path("/workdir/sol.csv")
    summary_path = Path("/workdir/network_summary.json")
    gt_path = Path("/tests/answers.csv")
    
    # Check file existence
    if not sol_path.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing solution file: /workdir/sol.csv",
            subscores=subscores,
            weights=weights,
            details={"error": "Solution file not found"}
        )
    
    if not summary_path.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing network_summary.json file",
            subscores=subscores,
            weights=weights,
            details={"error": "Summary file not found"}
        )
    
    try:
        sol = pd.read_csv(sol_path)
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        data = load_all_data()
        
        # FORMAT VALIDATION
        if len(sol) != 5:
            return GradingResult(
                score=0.0,
                feedback=f"Expected exactly 5 rows, got {len(sol)}",
                subscores=subscores,
                weights=weights,
                details={"row_count": len(sol)}
            )
        
        required_cols = {
            'station_id', 'location_name', 'latitude', 'longitude', 'neighborhood',
            'census_tract_id', 'projected_daily_ridership', 'demographic_multiplier',
            'poi_proximity_multiplier', 'network_effect_multiplier', 'weather_adjustment',
            'nearby_existing_count', 'is_high_density', 'near_transit', 'is_isolated'
        }
        missing = required_cols - set(sol.columns)
        if missing:
            return GradingResult(
                score=0.0,
                feedback=f"Missing columns: {sorted(missing)}",
                subscores=subscores,
                weights=weights,
                details={"missing_columns": sorted(missing)}
            )
        
        # STATION VALIDATION
        for _, row in sol.iterrows():
            if row['station_id'] not in data['candidates']:
                return GradingResult(
                    score=0.0,
                    feedback=f"Invalid station_id: {row['station_id']}",
                    subscores=subscores,
                    weights=weights,
                    details={"invalid_id": row['station_id']}
                )
        
        # CONSTRAINT VALIDATION
        constraints, constraint_details = check_constraints(sol, data)
        if not all(constraints.values()):
            failed = [k for k, v in constraints.items() if not v]
            return GradingResult(
                score=0.0,
                feedback=f"Constraints violated: {failed}",
                subscores=subscores,
                weights=weights,
                details={"constraints": constraints, "details": constraint_details}
            )
        
        # CALCULATION VALIDATION
        total_projected = 0.0
        for _, row in sol.iterrows():
            candidate = data['candidates'][row['station_id']]
            expected = calculate_projected_ridership(candidate, data)
            
            # Check each multiplier
            if abs(row['demographic_multiplier'] - expected['demographic_multiplier']) > 0.01:
                return GradingResult(
                    score=0.0,
                    feedback=f"Demographic multiplier error for {row['station_id']}",
                    subscores=subscores,
                    weights=weights,
                    details={
                        "expected": round(expected['demographic_multiplier'], 3),
                        "got": row['demographic_multiplier']
                    }
                )
            
            if abs(row['poi_proximity_multiplier'] - expected['poi_proximity_multiplier']) > 0.01:
                return GradingResult(
                    score=0.0,
                    feedback=f"POI multiplier error for {row['station_id']}",
                    subscores=subscores,
                    weights=weights,
                    details={
                        "expected": round(expected['poi_proximity_multiplier'], 3),
                        "got": row['poi_proximity_multiplier']
                    }
                )
            
            if abs(row['network_effect_multiplier'] - expected['network_effect_multiplier']) > 0.01:
                return GradingResult(
                    score=0.0,
                    feedback=f"Network effect multiplier error for {row['station_id']}",
                    subscores=subscores,
                    weights=weights,
                    details={
                        "expected": round(expected['network_effect_multiplier'], 3),
                        "got": row['network_effect_multiplier']
                    }
                )
            
            if abs(row['weather_adjustment'] - expected['weather_adjustment']) > 0.01:
                return GradingResult(
                    score=0.0,
                    feedback=f"Weather adjustment error for {row['station_id']}",
                    subscores=subscores,
                    weights=weights,
                    details={
                        "expected": round(expected['weather_adjustment'], 3),
                        "got": row['weather_adjustment']
                    }
                )
            
            if abs(row['projected_daily_ridership'] - expected['projected_ridership']) > 0.1:
                return GradingResult(
                    score=0.0,
                    feedback=f"Projected ridership error for {row['station_id']}",
                    subscores=subscores,
                    weights=weights,
                    details={
                        "expected": round(expected['projected_ridership'], 1),
                        "got": row['projected_daily_ridership']
                    }
                )
            
            total_projected += expected['projected_ridership']
        
        # PENALTY VALIDATION
        spacing_pen, imbalance_pen, isolation_pen, cv = calculate_penalties(sol, data)
        
        if abs(summary['spacing_penalty'] - spacing_pen) > 1.0:
            return GradingResult(
                score=0.0,
                feedback="Spacing penalty calculation error",
                subscores=subscores,
                weights=weights,
                details={"expected": round(spacing_pen, 1), "got": summary['spacing_penalty']}
            )
        
        if abs(summary['imbalance_penalty'] - imbalance_pen) > 1.0:
            return GradingResult(
                score=0.0,
                feedback="Imbalance penalty calculation error",
                subscores=subscores,
                weights=weights,
                details={"expected": round(imbalance_pen, 1), "got": summary['imbalance_penalty']}
            )
        
        if abs(summary['isolation_penalty'] - isolation_pen) > 1.0:
            return GradingResult(
                score=0.0,
                feedback="Isolation penalty calculation error",
                subscores=subscores,
                weights=weights,
                details={"expected": round(isolation_pen, 1), "got": summary['isolation_penalty']}
            )
        
        # NETWORK SCORE VALIDATION
        expected_score = total_projected - spacing_pen - imbalance_pen - isolation_pen
        if abs(summary['network_score'] - expected_score) > 1.0:
            return GradingResult(
                score=0.0,
                feedback="Network score calculation error",
                subscores=subscores,
                weights=weights,
                details={"expected": round(expected_score, 1), "got": summary['network_score']}
            )
        
        # OPTIMALITY CHECK
        if gt_path.exists():
            gt = pd.read_csv(gt_path)
            gt_summary_path = Path("/tests/ground_truth_summary.json")
            with open(gt_summary_path, 'r') as f:
                gt_summary = json.load(f)
            
            gt_score = gt_summary['network_score']
            if summary['network_score'] < gt_score - 50:
                return GradingResult(
                    score=0.0,
                    feedback=f"Solution suboptimal: {summary['network_score']:.1f} < {gt_score:.1f} - 50",
                    subscores=subscores,
                    weights=weights,
                    details={
                        "ground_truth_score": gt_score,
                        "solution_score": summary['network_score'],
                        "gap": round(gt_score - summary['network_score'], 1)
                    }
                )
        
        # ALL CHECKS PASSED
        subscores["all_passes"] = 1.0
        details["success"] = {
            "network_score": summary['network_score'],
            "total_ridership": total_projected,
            "constraints_satisfied": constraints
        }
        
        return GradingResult(
            score=1.0,
            feedback="All requirements met - optimal network found",
            subscores=subscores,
            weights=weights,
            details=details
        )
    
    except Exception as e:
        return GradingResult(
            score=0.0,
            feedback=f"Error during grading: {e}",
            subscores=subscores,
            weights=weights,
            details={"exception": str(e), "type": type(e).__name__}
        )