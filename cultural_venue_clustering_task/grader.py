from pathlib import Path
import pandas as pd
import json
import math
from dataclasses import dataclass, field
from typing import Dict, Any

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


def to_float(x):
    """Convert value to float, return None on error."""
    try:
        return float(str(x).strip())
    except Exception:
        return None


def to_int(x):
    """Convert value to int, return None on error."""
    try:
        return int(str(x).strip())
    except Exception:
        return None


def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in miles between two coordinates."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def get_museum_founded_year(name):
    """Get founding year for museums based on name."""
    museum_years = {
        'Metropolitan Museum': 1870,
        'MoMA': 1929,
        'Modern Art': 1929,
        'Natural History': 1869,
        'Guggenheim': 1959,
        'Whitney': 1930,
        'Frick': 1935,
        'September 11': 2011,
        'Tenement': 1988,
        'City of New York': 1923
    }
    
    for key, year in museum_years.items():
        if key in name:
            return year
    return 1900


def get_michelin_stars(accolades):
    """Extract Michelin stars from restaurant accolades."""
    if not accolades:
        return 0
    
    # Check in order: 3 stars, 2 stars, 1 star
    for acc in accolades:
        if '3 Michelin Star' in acc:
            return 3
    
    for acc in accolades:
        if '2 Michelin Star' in acc:
            return 2
    
    for acc in accolades:
        if 'Michelin Star' in acc or '1 Michelin Star' in acc:
            return 1
    
    return 0


def load_venues():
    """Load all venue data from JSON files and return unified dictionary."""
    venues = {}

    # Load theaters
    with open('/workdir/data/manhattan_theaters.json', 'r') as f:
        data = json.load(f)
        for t in data['manhattan_theaters']['theaters']:
            venues[t['name']] = {
                'type': 'theater',
                'lat': t['location']['coordinates']['latitude'],
                'lon': t['location']['coordinates']['longitude'],
                'neighborhood': t['location']['neighborhood'],
                'founded': t.get('opened', 1950),
                'base_value': 30,
                'michelin_stars': 0
            }

    # Load restaurants
    with open('/workdir/data/manhattan_restaurants.json', 'r') as f:
        data = json.load(f)
        for r in data['manhattan_restaurants']['restaurants']:
            stars = get_michelin_stars(r.get('accolades', []))
            venues[r['name']] = {
                'type': 'restaurant',
                'lat': r['location']['coordinates']['latitude'],
                'lon': r['location']['coordinates']['longitude'],
                'neighborhood': r['location']['neighborhood'],
                'founded': r.get('founded', 1980),
                'base_value': 25,
                'michelin_stars': stars
            }

    # Load museums
    with open('/workdir/data/manhattan_museums.json', 'r') as f:
        data = json.load(f)
        for m in data['nyc_museums']['museums']:
            if m['location']['borough'] == 'Manhattan':
                founded_year = get_museum_founded_year(m['name'])
                venues[m['name']] = {
                    'type': 'museum',
                    'lat': m['location']['coordinates']['latitude'],
                    'lon': m['location']['coordinates']['longitude'],
                    'neighborhood': m['location']['neighborhood'],
                    'founded': founded_year,
                    'base_value': 40,
                    'michelin_stars': 0
                }

    # Load bookstores
    with open('/workdir/data/manhattan_bookstores.json', 'r') as f:
        data = json.load(f)
        for b in data['manhattan_bookstores']['bookstores']:
            venues[b['name']] = {
                'type': 'bookstore',
                'lat': b['location']['coordinates']['latitude'],
                'lon': b['location']['coordinates']['longitude'],
                'neighborhood': b['location']['neighborhood'],
                'founded': b.get('founded', 1980),
                'base_value': 20,
                'michelin_stars': 0
            }

    # Load coffee shops
    with open('/workdir/data/manhattan_coffee_shops.json', 'r') as f:
        data = json.load(f)
        for c in data['manhattan_coffee_shops']['coffee_shops']:
            venues[c['name']] = {
                'type': 'coffee',
                'lat': c['location']['coordinates']['latitude'],
                'lon': c['location']['coordinates']['longitude'],
                'neighborhood': c['location']['neighborhood'],
                'founded': c.get('founded', 2010),
                'base_value': 15,
                'michelin_stars': 0
            }

    return venues


def grade(transcript: str) -> GradingResult:
    """
    Grade the Manhattan Cultural Venue Clustering task.
    
    Returns 1.0 (PASS) if ALL conditions are met:
    1. Exactly 1 data row in sol.csv
    2. All required columns present
    3. All 3 venues exist in datasets with correct names (exact match)
    4. Venue types are correct (exact match)
    5. Venue neighborhoods are correct (exact match)
    6. All 3 venues from different neighborhoods
    7. At least 2 different venue types
    8. Cluster diameter ≤ 1.5 miles and correctly calculated
    9. Cultural density correctly calculated
    10. Quality bonus correctly calculated
    11. Cluster score correctly calculated
    12. Cluster score >= ground_truth - 10
    
    Returns 0.0 (FAIL) if any condition is not met.
    """
    gt_path = Path("/tests/answers.csv")
    pred_path = Path("/workdir/sol.csv")
    
    weights = {"all_passes": 1.0}
    subscores = {"all_passes": 0.0}
    details = {}

    # Check file existence
    if not gt_path.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing ground truth file",
            subscores=subscores,
            weights=weights,
            details={"error": "Ground truth file not found"}
        )
    
    if not pred_path.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing solution file: /workdir/sol.csv",
            subscores=subscores,
            weights=weights,
            details={"error": "Solution file not found"}
        )

    try:
        gt = pd.read_csv(gt_path)
        sol = pd.read_csv(pred_path)
        venues = load_venues()

        # FORMAT VALIDATION
        if len(sol) != 1:
            details["format"] = {"row_count": len(sol), "expected": 1}
            return GradingResult(
                score=0.0,
                feedback=f"Expected exactly 1 data row, got {len(sol)}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        required_cols = {
            "venue_1_name", "venue_1_type", "venue_1_neighborhood",
            "venue_2_name", "venue_2_type", "venue_2_neighborhood",
            "venue_3_name", "venue_3_type", "venue_3_neighborhood",
            "cluster_center_lat", "cluster_center_lon", "cluster_diameter_miles",
            "cultural_density", "quality_bonus", "cluster_score"
        }
        missing = required_cols - set(sol.columns)
        if missing:
            details["format"] = {"missing_columns": sorted(missing)}
            return GradingResult(
                score=0.0,
                feedback=f"Missing columns: {sorted(missing)}",
                subscores=subscores,
                weights=weights,
                details=details
            )

        row = sol.iloc[0]
        venue_names = [row["venue_1_name"], row["venue_2_name"], row["venue_3_name"]]
        venue_types = [row["venue_1_type"], row["venue_2_type"], row["venue_3_type"]]
        venue_neighborhoods = [row["venue_1_neighborhood"], row["venue_2_neighborhood"], row["venue_3_neighborhood"]]

        # VENUE VALIDATION
        venue_details = []
        for i, name in enumerate(venue_names):
            if name not in venues:
                details["venues"] = {"error": f"Venue '{name}' not found"}
                return GradingResult(
                    score=0.0,
                    feedback=f"Venue '{name}' not found in datasets",
                    subscores=subscores,
                    weights=weights,
                    details=details
                )
            
            v = venues[name]
            if v['type'] != venue_types[i]:
                details["venues"] = {
                    "venue": name,
                    "expected_type": v['type'],
                    "got_type": venue_types[i]
                }
                return GradingResult(
                    score=0.0,
                    feedback=f"Venue '{name}' type mismatch: expected '{v['type']}', got '{venue_types[i]}'",
                    subscores=subscores,
                    weights=weights,
                    details=details
                )
            
            if v['neighborhood'] != venue_neighborhoods[i]:
                details["venues"] = {
                    "venue": name,
                    "expected_neighborhood": v['neighborhood'],
                    "got_neighborhood": venue_neighborhoods[i]
                }
                return GradingResult(
                    score=0.0,
                    feedback=f"Venue '{name}' neighborhood mismatch: expected '{v['neighborhood']}', got '{venue_neighborhoods[i]}'",
                    subscores=subscores,
                    weights=weights,
                    details=details
                )
            
            venue_details.append({"name": name, "type": v['type'], "neighborhood": v['neighborhood']})

        # CONSTRAINT VALIDATION
        unique_neighborhoods = set(venue_neighborhoods)
        if len(unique_neighborhoods) != 3:
            details["constraints"] = {
                "neighborhoods": venue_neighborhoods,
                "unique_count": len(unique_neighborhoods)
            }
            return GradingResult(
                score=0.0,
                feedback=f"All venues must be from different neighborhoods. Got: {venue_neighborhoods}",
                subscores=subscores,
                weights=weights,
                details=details
            )

        unique_types = set(venue_types)
        if len(unique_types) < 2:
            details["constraints"] = {
                "types": venue_types,
                "unique_count": len(unique_types)
            }
            return GradingResult(
                score=0.0,
                feedback=f"Need at least 2 different venue types. Got: {venue_types}",
                subscores=subscores,
                weights=weights,
                details=details
            )

        # CALCULATION VALIDATION
        v1, v2, v3 = [venues[n] for n in venue_names]
        
        # Calculate diameter
        d12 = haversine_miles(v1['lat'], v1['lon'], v2['lat'], v2['lon'])
        d13 = haversine_miles(v1['lat'], v1['lon'], v3['lat'], v3['lon'])
        d23 = haversine_miles(v2['lat'], v2['lon'], v3['lat'], v3['lon'])
        actual_diameter = max(d12, d13, d23)
        reported_diameter = to_float(row['cluster_diameter_miles'])
        
        if reported_diameter is None:
            details["calculations"] = {"error": "Invalid cluster_diameter_miles value"}
            return GradingResult(
                score=0.0,
                feedback="Invalid cluster_diameter_miles value",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        if abs(actual_diameter - reported_diameter) > 0.01:
            details["calculations"] = {
                "diameter_expected": round(actual_diameter, 3),
                "diameter_reported": reported_diameter,
                "error": abs(actual_diameter - reported_diameter)
            }
            return GradingResult(
                score=0.0,
                feedback=f"Diameter calculation error: expected {actual_diameter:.3f}, got {reported_diameter}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        if actual_diameter > 1.5:
            details["calculations"] = {
                "diameter": actual_diameter,
                "max_allowed": 1.5
            }
            return GradingResult(
                score=0.0,
                feedback=f"Cluster diameter {actual_diameter:.3f} exceeds 1.5 mile limit",
                subscores=subscores,
                weights=weights,
                details=details
            )

        # Calculate expected values
        base_sum = v1['base_value'] + v2['base_value'] + v3['base_value']
        diversity_mult = 1.5 if len(unique_types) == 3 else 1.0
        historic_count = sum(1 for v in [v1, v2, v3] if v['founded'] < 1950)
        michelin_total = v1['michelin_stars'] + v2['michelin_stars'] + v3['michelin_stars']
        expected_quality_bonus = (michelin_total * 5) + (historic_count * 8)
        expected_cultural_density = (base_sum * diversity_mult) + expected_quality_bonus
        expected_cluster_score = expected_cultural_density - (actual_diameter * 50)

        reported_quality_bonus = to_int(row['quality_bonus'])
        reported_cultural_density = to_float(row['cultural_density'])
        reported_cluster_score = to_float(row['cluster_score'])

        # Validate quality bonus
        if reported_quality_bonus != expected_quality_bonus:
            details["calculations"] = {
                "quality_bonus_expected": expected_quality_bonus,
                "quality_bonus_reported": reported_quality_bonus
            }
            return GradingResult(
                score=0.0,
                feedback=f"Quality bonus error: expected {expected_quality_bonus}, got {reported_quality_bonus}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # Validate cultural density
        if abs(reported_cultural_density - expected_cultural_density) > 0.1:
            details["calculations"] = {
                "cultural_density_expected": round(expected_cultural_density, 1),
                "cultural_density_reported": reported_cultural_density
            }
            return GradingResult(
                score=0.0,
                feedback=f"Cultural density error: expected {expected_cultural_density:.1f}, got {reported_cultural_density}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # Validate cluster score
        if abs(reported_cluster_score - expected_cluster_score) > 0.1:
            details["calculations"] = {
                "cluster_score_expected": round(expected_cluster_score, 1),
                "cluster_score_reported": reported_cluster_score
            }
            return GradingResult(
                score=0.0,
                feedback=f"Cluster score error: expected {expected_cluster_score:.1f}, got {reported_cluster_score}",
                subscores=subscores,
                weights=weights,
                details=details
            )

        # OPTIMALITY CHECK
        gt_score = to_float(gt.iloc[0]["cluster_score"])
        if reported_cluster_score < gt_score - 10:
            details["optimality"] = {
                "ground_truth_score": gt_score,
                "solution_score": reported_cluster_score,
                "gap": round(gt_score - reported_cluster_score, 1)
            }
            return GradingResult(
                score=0.0,
                feedback=f"Solution suboptimal: {reported_cluster_score:.1f} < {gt_score:.1f} - 10",
                subscores=subscores,
                weights=weights,
                details=details
            )

        # ALL CHECKS PASSED
        subscores["all_passes"] = 1.0
        details["success"] = {
            "venues": venue_details,
            "cluster_score": reported_cluster_score,
            "ground_truth_score": gt_score,
            "quality": "optimal" if reported_cluster_score >= gt_score else "near-optimal"
        }
        
        return GradingResult(
            score=1.0,
            feedback="All requirements met - optimal cluster found",
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