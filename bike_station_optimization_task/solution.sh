#!/bin/bash
set -e

# Solution for Bike Station Optimization Task
# Outputs a single solution.json file
python3 << 'EOF'
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in miles"""
    R = 3959
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def load_data():
    """Load all data files"""
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
    """Calculate demographic multiplier"""
    tract = next((t for t in demographics if t['census_tract_id'] == candidate['census_tract_id']), None)
    if not tract:
        return 1.0
    
    pop_density_factor = max(0, min(1, (tract['population_density'] - 20000) / 30000))
    commuter_density_factor = max(0, min(1, (tract['commuter_density'] - 5000) / 10000))
    
    return 1.0 + (pop_density_factor * 0.3) + (commuter_density_factor * 0.4)


def calculate_poi_proximity_multiplier(candidate: Dict, pois: List[Dict]) -> float:
    """Calculate POI proximity multiplier"""
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
    """Calculate network effect multiplier"""
    nearby_count = 0
    for station in existing:
        dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                 station['latitude'], station['longitude'])
        if 0.25 < dist <= 0.5:
            nearby_count += 1
    
    factor = min(nearby_count / 5, 1.0)
    return 1.0 + (factor * 0.2), nearby_count


def calculate_weather_adjustment(candidate: Dict, weather: Dict) -> float:
    """Calculate weather adjustment"""
    zone = candidate['weather_zone']
    if zone not in weather:
        return 1.0
    
    zone_data = weather[zone]
    return (zone_data['spring_multiplier'] + zone_data['summer_multiplier'] +
            zone_data['fall_multiplier'] + zone_data['winter_multiplier']) / 4


def is_isolated(candidate: Dict, existing: List[Dict]) -> bool:
    """Check if station is isolated"""
    count = 0
    for station in existing:
        dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                 station['latitude'], station['longitude'])
        if dist <= 0.4:
            count += 1
    return count < 2


def is_near_transit(candidate: Dict, pois: List[Dict]) -> bool:
    """Check if near transit"""
    for poi in pois:
        if poi['type'] == 'subway_station':
            dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                     poi['latitude'], poi['longitude'])
            if dist <= 0.1:
                return True
    return False


def is_high_density(candidate: Dict, demographics: List[Dict]) -> bool:
    """Check if high density"""
    tract = next((t for t in demographics if t['census_tract_id'] == candidate['census_tract_id']), None)
    return tract and tract['density_category'] == 'High Density'


def check_constraints(selection: List[Dict], existing: List[Dict], 
                     demographics: List[Dict], pois: List[Dict]) -> Tuple[bool, Dict]:
    """Check all constraints"""
    constraints = {}
    
    # Coverage constraint
    min_dist_to_existing = float('inf')
    for candidate in selection:
        for station in existing:
            dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                     station['latitude'], station['longitude'])
            min_dist_to_existing = min(min_dist_to_existing, dist)
    constraints['coverage_constraint'] = min_dist_to_existing >= 0.25
    
    # Spacing constraint
    min_dist_between = float('inf')
    for i in range(len(selection)):
        for j in range(i+1, len(selection)):
            dist = haversine_distance(selection[i]['latitude'], selection[i]['longitude'],
                                     selection[j]['latitude'], selection[j]['longitude'])
            min_dist_between = min(min_dist_between, dist)
    constraints['spacing_constraint'] = min_dist_between >= 0.15
    
    # Density constraint
    high_density_count = sum(1 for c in selection if is_high_density(c, demographics))
    constraints['density_constraint'] = high_density_count >= 3
    
    # Transit integration
    near_transit_count = sum(1 for c in selection if is_near_transit(c, pois))
    constraints['transit_integration'] = near_transit_count >= 2
    
    # Geographic diversity
    neighborhoods = set(c['neighborhood'] for c in selection)
    constraints['geographic_diversity'] = len(neighborhoods) >= 3
    
    return all(constraints.values()), constraints


def calculate_network_score(selection: List[Dict], existing: List[Dict],
                           demographics: List[Dict], pois: List[Dict], 
                           weather: Dict) -> Tuple[float, Dict]:
    """Calculate network score and return details"""
    ridership_values = []
    station_details = []
    
    for candidate in selection:
        demo_mult = calculate_demographic_multiplier(candidate, demographics)
        poi_mult = calculate_poi_proximity_multiplier(candidate, pois)
        network_mult, nearby_count = calculate_network_effect_multiplier(candidate, existing)
        weather_adj = calculate_weather_adjustment(candidate, weather)
        
        base = candidate['baseline_demand']
        ridership = base * demo_mult * poi_mult * network_mult * weather_adj
        ridership_values.append(ridership)
        
        station_details.append({
            'station_id': candidate['station_id'],
            'location_name': candidate['location_name'],
            'latitude': round(candidate['latitude'], 4),
            'longitude': round(candidate['longitude'], 4),
            'neighborhood': candidate['neighborhood'],
            'census_tract_id': candidate['census_tract_id'],
            'projected_daily_ridership': round(ridership, 1),
            'demographic_multiplier': round(demo_mult, 3),
            'poi_proximity_multiplier': round(poi_mult, 3),
            'network_effect_multiplier': round(network_mult, 3),
            'weather_adjustment': round(weather_adj, 3),
            'nearby_existing_count': nearby_count,
            'is_high_density': is_high_density(candidate, demographics),
            'near_transit': is_near_transit(candidate, pois),
            'is_isolated': is_isolated(candidate, existing)
        })
    
    total_ridership = sum(ridership_values)
    
    # Spacing penalty
    spacing_penalty = 0
    for i in range(len(selection)):
        for j in range(i+1, len(selection)):
            dist = haversine_distance(selection[i]['latitude'], selection[i]['longitude'],
                                     selection[j]['latitude'], selection[j]['longitude'])
            if dist < 0.20:
                spacing_penalty += max(0, (0.20 - dist) * 1000)
    
    # Imbalance penalty
    mean = sum(ridership_values) / len(ridership_values)
    variance = sum((x - mean)**2 for x in ridership_values) / len(ridership_values)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean if mean > 0 else 0
    imbalance_penalty = max(0, (cv - 0.25) * 500)
    
    # Isolation penalty
    isolation_penalty = sum(200 for detail in station_details if detail['is_isolated'])
    
    network_score = total_ridership - spacing_penalty - imbalance_penalty - isolation_penalty
    
    return network_score, {
        'station_details': station_details,
        'total_ridership': total_ridership,
        'spacing_penalty': spacing_penalty,
        'imbalance_penalty': imbalance_penalty,
        'isolation_penalty': isolation_penalty,
        'cv': cv
    }


def optimize_stations(existing, candidates, demographics, pois, weather):
    """Optimize station selection using greedy + local search"""
    
    # Pre-filter candidates that meet basic constraints
    valid_candidates = []
    for candidate in candidates:
        # Check minimum distance to existing
        min_dist = min(haversine_distance(candidate['latitude'], candidate['longitude'],
                                         station['latitude'], station['longitude'])
                      for station in existing)
        if min_dist >= 0.25:
            valid_candidates.append(candidate)
    
    print(f"Valid candidates after coverage filter: {len(valid_candidates)}")
    
    # Greedy initialization: select high-value candidates
    best_score = -float('inf')
    best_selection = None
    best_info = None
    
    # Calculate single-station metrics for sorting
    candidate_scores = []
    for candidate in valid_candidates:
        demo_mult = calculate_demographic_multiplier(candidate, demographics)
        poi_mult = calculate_poi_proximity_multiplier(candidate, pois)
        network_mult, _ = calculate_network_effect_multiplier(candidate, existing)
        weather_adj = calculate_weather_adjustment(candidate, weather)
        
        score = candidate['baseline_demand'] * demo_mult * poi_mult * network_mult * weather_adj
        
        # Bonus for high density and near transit
        if is_high_density(candidate, demographics):
            score *= 1.1
        if is_near_transit(candidate, pois):
            score *= 1.1
        
        candidate_scores.append((score, candidate))
    
    # Sort by score
    candidate_scores.sort(reverse=True)
    
    # Try top candidates with greedy approach
    print("Starting greedy search...")
    for start_idx in range(min(50, len(candidate_scores))):
        selection = [candidate_scores[start_idx][1]]
        
        # Greedily add 4 more stations
        for _ in range(4):
            best_addition = None
            best_addition_score = -float('inf')
            
            for score, candidate in candidate_scores:
                if candidate in selection:
                    continue
                
                # Check spacing with current selection
                valid_spacing = all(
                    haversine_distance(candidate['latitude'], candidate['longitude'],
                                     s['latitude'], s['longitude']) >= 0.15
                    for s in selection
                )
                
                if not valid_spacing:
                    continue
                
                test_selection = selection + [candidate]
                valid, _ = check_constraints(test_selection, existing, demographics, pois)
                
                if valid or len(test_selection) < 5:
                    test_score, _ = calculate_network_score(test_selection, existing, 
                                                           demographics, pois, weather)
                    if test_score > best_addition_score:
                        best_addition_score = test_score
                        best_addition = candidate
            
            if best_addition:
                selection.append(best_addition)
            else:
                break
        
        if len(selection) == 5:
            valid, constraints = check_constraints(selection, existing, demographics, pois)
            if valid:
                score, info = calculate_network_score(selection, existing, demographics, pois, weather)
                if score > best_score:
                    best_score = score
                    best_selection = selection
                    best_info = info
                    print(f"Found valid solution with score: {score:.2f}")
    
    # If no solution found, try random sampling
    if best_selection is None:
        print("Greedy search failed, trying broader search...")
        import random
        random.seed(42)
        
        for _ in range(1000):
            sample = random.sample(candidate_scores[:100], min(100, len(candidate_scores[:100])))
            
            selection = []
            for score, candidate in sample:
                if len(selection) >= 5:
                    break
                
                valid_addition = True
                for s in selection:
                    dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                            s['latitude'], s['longitude'])
                    if dist < 0.15:
                        valid_addition = False
                        break
                
                if valid_addition:
                    selection.append(candidate)
            
            if len(selection) == 5:
                valid, _ = check_constraints(selection, existing, demographics, pois)
                if valid:
                    score, info = calculate_network_score(selection, existing, demographics, pois, weather)
                    if score > best_score:
                        best_score = score
                        best_selection = selection
                        best_info = info
                        print(f"Found valid solution with score: {score:.2f}")
    
    return best_selection, best_score, best_info


def main():
    print("Loading data...")
    existing, candidates, demographics, pois, weather = load_data()
    
    print(f"Loaded {len(existing)} existing stations")
    print(f"Loaded {len(candidates)} candidate locations")
    
    print("\nOptimizing station placement...")
    best_selection, best_score, best_info = optimize_stations(
        existing, candidates, demographics, pois, weather
    )
    
    if best_selection is None:
        print("ERROR: Could not find valid solution")
        return
    
    print(f"\nBest network score: {best_score:.2f}")
    
    # Calculate final metrics
    valid, constraints = check_constraints(best_selection, existing, demographics, pois)
    
    min_dist_between = float('inf')
    for i in range(len(best_selection)):
        for j in range(i+1, len(best_selection)):
            dist = haversine_distance(best_selection[i]['latitude'], best_selection[i]['longitude'],
                                     best_selection[j]['latitude'], best_selection[j]['longitude'])
            min_dist_between = min(min_dist_between, dist)
    
    min_dist_to_existing = float('inf')
    for candidate in best_selection:
        for station in existing:
            dist = haversine_distance(candidate['latitude'], candidate['longitude'],
                                     station['latitude'], station['longitude'])
            min_dist_to_existing = min(min_dist_to_existing, dist)
    
    neighborhoods = list(set(c['neighborhood'] for c in best_selection))
    
    # Build complete solution JSON
    solution = {
        'selected_stations': best_info['station_details'],
        'network_summary': {
            'total_projected_ridership': round(best_info['total_ridership'], 1),
            'spacing_penalty': round(best_info['spacing_penalty'], 1),
            'imbalance_penalty': round(best_info['imbalance_penalty'], 1),
            'isolation_penalty': round(best_info['isolation_penalty'], 1),
            'network_score': round(best_score, 1),
            'ridership_coefficient_of_variation': round(best_info['cv'], 3),
            'constraints_satisfied': constraints,
            'min_distance_between_new_stations': round(min_dist_between, 3),
            'min_distance_to_existing': round(min_dist_to_existing, 3),
            'neighborhoods_represented': neighborhoods
        }
    }
    
    # Write single solution file
    with open('/workdir/solution.json', 'w') as f:
        json.dump(solution, f, indent=2)
    
    print("\nSolution saved to /workdir/solution.json")
    print(f"Constraints satisfied: {all(constraints.values())}")
    print(f"Network score: {best_score:.2f}")


if __name__ == '__main__':
    main()

EOF