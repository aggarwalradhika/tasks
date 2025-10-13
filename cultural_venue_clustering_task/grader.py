def get_museum_founded_year(name):
    """Get founding year for museums based on name (matches task.yaml spec)."""
    if 'Metropolitan Museum' in name:
        return 1870
    elif 'MoMA' in name or 'Modern Art' in name:
        return 1929
    elif 'Natural History' in name:
        return 1869
    elif 'Guggenheim' in name:
        return 1959
    elif 'Whitney' in name:
        return 1930
    elif 'Frick' in name:
        return 1935
    elif 'September 11' in name:
        return 2011
    elif 'Tenement' in name:
        return 1988
    elif 'City of New York' in name:
        return 1923
    else:
        return 1900


def get_michelin_stars(accolades):
    """Extract Michelin stars from restaurant accolades (matches task.yaml spec)."""
    if not accolades:
        return 0
    
    # Check for 3 stars first
    for acc in accolades:
        if '3 Michelin Star' in acc:
            return 3
    
    # Then check for 2 stars
    for acc in accolades:
        if '2 Michelin Star' in acc:
            return 2
    
    # Finally check for 1 star
    for acc in accolades:
        if 'Michelin Star' in acc or '1 Michelin Star' in acc:
            return 1
    
    return 0

from pathlib import Path
import pandas as pd
import json
import math
from apex_arena._types import GradingResult


def to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def to_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None


def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in miles."""
    R = 3959
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def load_venues():
    """Load all venue data from JSON files."""
    venues = {}

    # Theaters
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

    # Restaurants
    with open('/workdir/data/manhattan_restaurants.json', 'r') as f:
        data = json.load(f)
        for r in data['manhattan_restaurants']['restaurants']:
            stars = 0
            if 'accolades' in r:
                for acc in r['accolades']:
                    if '3 Michelin Star' in acc:
                        stars = 3
                        break
                    elif '2 Michelin Star' in acc:
                        stars = 2
                        break
                    elif 'Michelin Star' in acc:
                        stars = max(stars, 1)
            venues[r['name']] = {
                'type': 'restaurant',
                'lat': r['location']['coordinates']['latitude'],
                'lon': r['location']['coordinates']['longitude'],
                'neighborhood': r['location']['neighborhood'],
                'founded': r.get('founded', 1980),
                'base_value': 25,
                'michelin_stars': stars
            }

    # Museums
    with open('/workdir/data/manhattan_museums_json.json', 'r') as f:
        data = json.load(f)
        for m in data['nyc_museums']['museums']:
            if m['location']['borough'] == 'Manhattan':
                founded = 1900
                name = m['name']
                if 'Metropolitan Museum' in name:
                    founded = 1870
                elif 'MoMA' in name or 'Modern Art' in name:
                    founded = 1929
                elif 'Natural History' in name:
                    founded = 1869
                elif 'Guggenheim' in name:
                    founded = 1959
                elif 'Whitney' in name:
                    founded = 1930
                elif 'Frick' in name:
                    founded = 1935
                elif 'September 11' in name:
                    founded = 2011
                elif 'Tenement' in name:
                    founded = 1988
                elif 'City of New York' in name:
                    founded = 1923

                venues[name] = {
                    'type': 'museum',
                    'lat': m['location']['coordinates']['latitude'],
                    'lon': m['location']['coordinates']['longitude'],
                    'neighborhood': m['location']['neighborhood'],
                    'founded': founded,
                    'base_value': 40,
                    'michelin_stars': 0
                }

    # Bookstores
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

    # Coffee shops
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
    PASS if:
    1. Exactly 1 data row in sol.csv
    2. All required columns present
    3. All 3 venues exist in datasets with correct names
    4. Venue types are correct
    5. All 3 venues from different neighborhoods
    6. At least 2 different venue types
    7. Cluster diameter ≤ 1.5 miles and correctly calculated
    8. Cultural density correctly calculated
    9. Quality bonus correctly calculated
    10. Cluster score correctly calculated
    11. Cluster score >= ground truth - 10
    """
    gt_path = Path("/tests/answers.csv")
    pred_path = Path("/workdir/sol.csv")

    if not gt_path.exists():
        return GradingResult(score=0.0, feedback="Missing ground truth file")
    if not pred_path.exists():
        return GradingResult(score=0.0, feedback="Missing solution file: /workdir/sol.csv")

    try:
        gt = pd.read_csv(gt_path)
        sol = pd.read_csv(pred_path)
        venues = load_venues()

        # Check row count
        if len(sol) != 1:
            return GradingResult(score=0.0, feedback=f"Expected exactly 1 data row, got {len(sol)}")

        # Check required columns
        required_cols = {
            "venue_1_name", "venue_1_type", "venue_1_neighborhood",
            "venue_2_name", "venue_2_type", "venue_2_neighborhood",
            "venue_3_name", "venue_3_type", "venue_3_neighborhood",
            "cluster_center_lat", "cluster_center_lon", "cluster_diameter_miles",
            "cultural_density", "quality_bonus", "cluster_score"
        }
        missing = required_cols - set(sol.columns)
        if missing:
            return GradingResult(score=0.0, feedback=f"Missing columns: {sorted(missing)}")

        row = sol.iloc[0]
        errors = []

        # Extract venue info
        venue_names = [row["venue_1_name"], row["venue_2_name"], row["venue_3_name"]]
        venue_types = [row["venue_1_type"], row["venue_2_type"], row["venue_3_type"]]
        venue_neighborhoods = [row["venue_1_neighborhood"], row["venue_2_neighborhood"], row["venue_3_neighborhood"]]

        # Validate venues exist
        for i, name in enumerate(venue_names):
            if name not in venues:
                errors.append(f"Venue '{name}' not found in datasets")
            else:
                v = venues[name]
                if v['type'] != venue_types[i]:
                    errors.append(f"Venue '{name}' type mismatch: expected '{v['type']}', got '{venue_types[i]}'")
                if v['neighborhood'] != venue_neighborhoods[i]:
                    errors.append(f"Venue '{name}' neighborhood mismatch: expected '{v['neighborhood']}', got '{venue_neighborhoods[i]}'")

        if errors:
            return GradingResult(score=0.0, feedback="; ".join(errors))

        # Check different neighborhoods
        if len(set(venue_neighborhoods)) != 3:
            return GradingResult(score=0.0, feedback=f"All venues must be from different neighborhoods. Got: {venue_neighborhoods}")

        # Check venue type diversity
        if len(set(venue_types)) < 2:
            return GradingResult(score=0.0, feedback=f"Need at least 2 different venue types. Got: {venue_types}")

        # Validate calculations
        v1, v2, v3 = [venues[n] for n in venue_names]
        
        # Calculate actual diameter
        d12 = haversine_miles(v1['lat'], v1['lon'], v2['lat'], v2['lon'])
        d13 = haversine_miles(v1['lat'], v1['lon'], v3['lat'], v3['lon'])
        d23 = haversine_miles(v2['lat'], v2['lon'], v3['lat'], v3['lon'])
        actual_diameter = max(d12, d13, d23)
        
        reported_diameter = to_float(row['cluster_diameter_miles'])
        if reported_diameter is None or abs(actual_diameter - reported_diameter) > 0.01:
            return GradingResult(score=0.0, feedback=f"Diameter calculation error: expected {actual_diameter:.3f}, got {reported_diameter}")
        
        if actual_diameter > 1.5:
            return GradingResult(score=0.0, feedback=f"Cluster diameter {actual_diameter:.3f} exceeds 1.5 mile limit")

        # Calculate expected values
        base_sum = v1['base_value'] + v2['base_value'] + v3['base_value']
        diversity_mult = 1.5 if len(set(venue_types)) == 3 else 1.0
        historic_count = sum(1 for v in [v1, v2, v3] if v['founded'] < 1950)
        michelin_total = v1['michelin_stars'] + v2['michelin_stars'] + v3['michelin_stars']
        expected_quality_bonus = (michelin_total * 5) + (historic_count * 8)
        expected_cultural_density = (base_sum * diversity_mult) + expected_quality_bonus
        expected_cluster_score = expected_cultural_density - (actual_diameter * 50)

        reported_quality_bonus = to_int(row['quality_bonus'])
        reported_cultural_density = to_float(row['cultural_density'])
        reported_cluster_score = to_float(row['cluster_score'])

        if reported_quality_bonus != expected_quality_bonus:
            return GradingResult(score=0.0, feedback=f"Quality bonus error: expected {expected_quality_bonus}, got {reported_quality_bonus}")
        
        if abs(reported_cultural_density - expected_cultural_density) > 0.1:
            return GradingResult(score=0.0, feedback=f"Cultural density error: expected {expected_cultural_density:.1f}, got {reported_cultural_density}")
        
        if abs(reported_cluster_score - expected_cluster_score) > 0.1:
            return GradingResult(score=0.0, feedback=f"Cluster score error: expected {expected_cluster_score:.1f}, got {reported_cluster_score}")

        # Compare with ground truth (10 point tolerance)
        gt_score = to_float(gt.iloc[0]["cluster_score"])
        if reported_cluster_score < gt_score - 10:
            return GradingResult(score=0.0, feedback=f"Solution suboptimal: {reported_cluster_score:.1f} < {gt_score:.1f} - 10")

        return GradingResult(score=1.0, feedback="All requirements met")

    except Exception as e:
        return GradingResult(score=0.0, feedback=f"Error during grading: {e}")  