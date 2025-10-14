#!/bin/bash

# Manhattan Cultural Venue Clustering - Reference Solution
# This script finds the optimal 3-venue cluster

python3 << 'PYTHON_SCRIPT'
import json
import math
import csv
from itertools import combinations


def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in miles."""
    R = 3959
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def get_museum_founded_year(name):
    """Get founding year for museums based on name."""
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
    """Extract Michelin stars from restaurant accolades."""
    if not accolades:
        return 0
    
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
    """Load all venues from JSON files."""
    venues = []

    # Theaters
    with open('/workdir/data/manhattan_theaters.json', 'r') as f:
        data = json.load(f)
        for t in data['manhattan_theaters']['theaters']:
            venues.append({
                'name': t['name'],
                'type': 'theater',
                'lat': t['location']['coordinates']['latitude'],
                'lon': t['location']['coordinates']['longitude'],
                'neighborhood': t['location']['neighborhood'],
                'founded': t.get('opened', 1950),
                'base_value': 30,
                'michelin_stars': 0
            })

    # Restaurants
    with open('/workdir/data/manhattan_restaurants.json', 'r') as f:
        data = json.load(f)
        for r in data['manhattan_restaurants']['restaurants']:
            stars = get_michelin_stars(r.get('accolades', []))
            venues.append({
                'name': r['name'],
                'type': 'restaurant',
                'lat': r['location']['coordinates']['latitude'],
                'lon': r['location']['coordinates']['longitude'],
                'neighborhood': r['location']['neighborhood'],
                'founded': r.get('founded', 1980),
                'base_value': 25,
                'michelin_stars': stars
            })

    # Museums (Manhattan only)
    with open('/workdir/data/manhattan_museums.json', 'r') as f:
        data = json.load(f)
        for m in data['nyc_museums']['museums']:
            if m['location']['borough'] == 'Manhattan':
                founded_year = get_museum_founded_year(m['name'])
                venues.append({
                    'name': m['name'],
                    'type': 'museum',
                    'lat': m['location']['coordinates']['latitude'],
                    'lon': m['location']['coordinates']['longitude'],
                    'neighborhood': m['location']['neighborhood'],
                    'founded': founded_year,
                    'base_value': 40,
                    'michelin_stars': 0
                })

    # Bookstores
    with open('/workdir/data/manhattan_bookstores.json', 'r') as f:
        data = json.load(f)
        for b in data['manhattan_bookstores']['bookstores']:
            venues.append({
                'name': b['name'],
                'type': 'bookstore',
                'lat': b['location']['coordinates']['latitude'],
                'lon': b['location']['coordinates']['longitude'],
                'neighborhood': b['location']['neighborhood'],
                'founded': b.get('founded', 1980),
                'base_value': 20,
                'michelin_stars': 0
            })

    # Coffee shops
    with open('/workdir/data/manhattan_coffee_shops.json', 'r') as f:
        data = json.load(f)
        for c in data['manhattan_coffee_shops']['coffee_shops']:
            venues.append({
                'name': c['name'],
                'type': 'coffee',
                'lat': c['location']['coordinates']['latitude'],
                'lon': c['location']['coordinates']['longitude'],
                'neighborhood': c['location']['neighborhood'],
                'founded': c.get('founded', 2010),
                'base_value': 15,
                'michelin_stars': 0
            })

    return venues


def calculate_cluster_metrics(v1, v2, v3):
    """Calculate all cluster metrics and score."""
    # Calculate diameter (max of 3 pairwise distances)
    d12 = haversine_miles(v1['lat'], v1['lon'], v2['lat'], v2['lon'])
    d13 = haversine_miles(v1['lat'], v1['lon'], v3['lat'], v3['lon'])
    d23 = haversine_miles(v2['lat'], v2['lon'], v3['lat'], v3['lon'])
    diameter = max(d12, d13, d23)

    # Calculate center (average coordinates)
    center_lat = (v1['lat'] + v2['lat'] + v3['lat']) / 3
    center_lon = (v1['lon'] + v2['lon'] + v3['lon']) / 3

    # Calculate cultural density
    base_sum = v1['base_value'] + v2['base_value'] + v3['base_value']
    
    types = {v1['type'], v2['type'], v3['type']}
    diversity_mult = 1.5 if len(types) == 3 else 1.0

    historic_count = sum(1 for v in [v1, v2, v3] if v['founded'] < 1950)
    michelin_total = v1['michelin_stars'] + v2['michelin_stars'] + v3['michelin_stars']
    quality_bonus = (michelin_total * 5) + (historic_count * 8)

    cultural_density = (base_sum * diversity_mult) + quality_bonus
    cluster_score = cultural_density - (diameter * 50)

    return {
        'diameter': diameter,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'cultural_density': cultural_density,
        'quality_bonus': quality_bonus,
        'cluster_score': cluster_score
    }


def find_optimal_cluster():
    """Find the optimal 3-venue cluster by exhaustive search."""
    venues = load_venues()
    best_cluster = None
    best_score = float('-inf')

    # Evaluate all 3-venue combinations
    for combo in combinations(venues, 3):
        v1, v2, v3 = combo

        # Check constraint: different neighborhoods
        neighborhoods = {v1['neighborhood'], v2['neighborhood'], v3['neighborhood']}
        if len(neighborhoods) < 3:
            continue

        # Check constraint: at least 2 different types
        types = {v1['type'], v2['type'], v3['type']}
        if len(types) < 2:
            continue

        # Calculate metrics
        metrics = calculate_cluster_metrics(v1, v2, v3)
        
        # Check constraint: diameter <= 1.5 miles
        if metrics['diameter'] > 1.5:
            continue

        # Update best if this is better
        if metrics['cluster_score'] > best_score:
            best_score = metrics['cluster_score']
            best_cluster = {
                'venues': [v1, v2, v3],
                'metrics': metrics
            }

    return best_cluster


def main():
    """Main function to find and save optimal cluster."""
    cluster = find_optimal_cluster()

    if cluster is None:
        print("ERROR: No valid cluster found!")
        return

    # Write to CSV
    with open('/workdir/sol.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'venue_1_name', 'venue_1_type', 'venue_1_neighborhood',
            'venue_2_name', 'venue_2_type', 'venue_2_neighborhood',
            'venue_3_name', 'venue_3_type', 'venue_3_neighborhood',
            'cluster_center_lat', 'cluster_center_lon', 'cluster_diameter_miles',
            'cultural_density', 'quality_bonus', 'cluster_score'
        ])

        # Data row
        v1, v2, v3 = cluster['venues']
        m = cluster['metrics']
        writer.writerow([
            v1['name'], v1['type'], v1['neighborhood'],
            v2['name'], v2['type'], v2['neighborhood'],
            v3['name'], v3['type'], v3['neighborhood'],
            round(m['center_lat'], 4),
            round(m['center_lon'], 4),
            round(m['diameter'], 3),
            round(m['cultural_density'], 1),
            int(m['quality_bonus']),
            round(m['cluster_score'], 1)
        ])

    print(f"Optimal cluster found with score: {m['cluster_score']:.1f}")
    print(f"Venues: {v1['name']}, {v2['name']}, {v3['name']}")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT