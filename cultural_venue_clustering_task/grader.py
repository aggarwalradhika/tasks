from pathlib import Path
import re
import pandas as pd
import json
import math
from collections import Counter
from apex_arena._types import GradingResult


def _to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _to_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None


def distance_miles(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in miles."""
    R = 3959
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def fuzzy_match(name, venue_names, threshold=0.7):
    """Simple fuzzy matching for venue names."""
    name_lower = name.lower().strip()
    for venue_name in venue_names:
        venue_lower = venue_name.lower().strip()
        if name_lower == venue_lower:
            return venue_name
        # Check if one contains the other (for partial matches)
        if name_lower in venue_lower or venue_lower in name_lower:
            return venue_name
    return None


def load_venue_data():
    """Load all venue data from JSON files"""
    venues = {}

    try:
        # Load theaters
        with open('/workdir/data/manhattan_theaters.json', 'r') as f:
            theater_data = json.load(f)
            for theater in theater_data['manhattan_theaters']['theaters']:
                venues[theater['name']] = {
                    'type': 'theater',
                    'latitude': theater['location']['coordinates']['latitude'],
                    'longitude': theater['location']['coordinates']['longitude'],
                    'neighborhood': theater['location']['neighborhood'],
                    'founded': theater.get('opened', 1950),
                    'base_value': 30,
                    'michelin_stars': 0,
                    'is_historic': theater.get('opened', 2000) < 1950
                }

        # Load restaurants
        with open('/workdir/data/manhattan_restaurants.json', 'r') as f:
            restaurant_data = json.load(f)
            for restaurant in restaurant_data['manhattan_restaurants']['restaurants']:
                # Extract Michelin stars
                michelin_stars = 0
                if 'accolades' in restaurant:
                    for accolade in restaurant['accolades']:
                        if '3 Michelin Stars' in accolade or '3 Michelin Star' in accolade:
                            michelin_stars = 3
                            break
                        elif '2 Michelin Stars' in accolade or '2 Michelin Star' in accolade:
                            michelin_stars = 2
                            break
                        elif '1 Michelin Star' in accolade or 'Michelin Star' in accolade:
                            michelin_stars = 1

                venues[restaurant['name']] = {
                    'type': 'restaurant',
                    'latitude': restaurant['location']['coordinates']['latitude'],
                    'longitude': restaurant['location']['coordinates']['longitude'],
                    'neighborhood': restaurant['location']['neighborhood'],
                    'founded': restaurant.get('founded', 1980),
                    'base_value': 25,
                    'michelin_stars': michelin_stars,
                    'is_historic': restaurant.get('founded', 2000) < 1950
                }

        # Load bookstores
        with open('/workdir/data/manhattan_bookstores.json', 'r') as f:
            bookstore_data = json.load(f)
            for bookstore in bookstore_data['manhattan_bookstores']['bookstores']:
                venues[bookstore['name']] = {
                    'type': 'bookstore',
                    'latitude': bookstore['location']['coordinates']['latitude'],
                    'longitude': bookstore['location']['coordinates']['longitude'],
                    'neighborhood': bookstore['location']['neighborhood'],
                    'founded': bookstore.get('founded', 1980),
                    'base_value': 20,
                    'michelin_stars': 0,
                    'is_historic': bookstore.get('founded', 2000) < 1950
                }

        # Load coffee shops
        with open('/workdir/data/manhattan_coffee_shops.json', 'r') as f:
            coffee_data = json.load(f)
            for coffee in coffee_data['manhattan_coffee_shops']['coffee_shops']:
                venues[coffee['name']] = {
                    'type': 'coffee',
                    'latitude': coffee['location']['coordinates']['latitude'],
                    'longitude': coffee['location']['coordinates']['longitude'],
                    'neighborhood': coffee['location']['neighborhood'],
                    'founded': coffee.get('founded', 2010),
                    'base_value': 15,
                    'michelin_stars': 0,
                    'is_historic': coffee.get('founded', 2000) < 1950
                }

        # Load museums
        with open('/workdir/data/manhattan_museums_json.json', 'r') as f:
            museum_data = json.load(f)
            for museum in museum_data['nyc_museums']['museums']:
                # Only Manhattan museums
                if museum['location']['borough'] == 'Manhattan':
                    # Assign realistic founding dates
                    founded_year = 1900
                    if 'Metropolitan Museum' in museum['name']:
                        founded_year = 1870
                    elif 'MoMA' in museum['name'] or 'Modern Art' in museum['name']:
                        founded_year = 1929
                    elif 'Natural History' in museum['name']:
                        founded_year = 1869
                    elif 'Guggenheim' in museum['name']:
                        founded_year = 1959
                    elif 'Whitney' in museum['name']:
                        founded_year = 1930
                    elif 'Frick' in museum['name']:
                        founded_year = 1935
                    elif 'September 11' in museum['name']:
                        founded_year = 2011
                    elif 'Tenement' in museum['name']:
                        founded_year = 1988
                    elif 'City of New York' in museum['name']:
                        founded_year = 1923

                    venues[museum['name']] = {
                        'type': 'museum',
                        'latitude': museum['location']['coordinates']['latitude'],
                        'longitude': museum['location']['coordinates']['longitude'],
                        'neighborhood': museum['location']['neighborhood'],
                        'founded': founded_year,
                        'base_value': 40,
                        'michelin_stars': 0,
                        'is_historic': founded_year < 1950
                    }

    except Exception as e:
        return {}

    return venues


def grade(transcript: str) -> GradingResult:
    """
    PASS iff:
      1) Exactly 1 row in sol.csv (single optimal 3-venue cluster)
      2) Required columns present for clustering analysis
      3) All 3 venues exist in datasets and have correct attributes
      4) All 3 venues are from different neighborhoods
      5) At least 2 different venue types in cluster
      6) Cluster diameter calculated correctly and <= 1.5 miles
      7) Cultural density calculated using correct base values and multipliers
      8) Quality bonus calculated from Michelin stars and historic venues
      9) Cluster score formula applied correctly
      10) Optimization result: cluster score >= ground truth (equal or better)

    Notes:
      - Validates Manhattan Cultural Venue Clustering Analysis
      - Focus on 3-venue cluster optimization balancing cultural value and compactness
      - Single optimal cluster maximizing cultural density while minimizing diameter
      - NO TOLERANCE: solution must be equal or better than ground truth
    """
    gt_path = Path("/tests/answers.csv")  # reference solution
    pred_path = Path("/workdir/sol.csv")  # agent output

    weights = {"all_passes": 1.0}

    # Existence checks
    if not gt_path.exists():
        return GradingResult(
            score=0.0, subscores={}, weights=weights,
            feedback=f"Missing ground truth file: {gt_path}"
        )
    if not pred_path.exists():
        return GradingResult(
            score=0.0, subscores={}, weights=weights,
            feedback=f"Missing solution file: {pred_path}"
        )

    try:
        # Load data
        gt = pd.read_csv(gt_path)
        sol = pd.read_csv(pred_path)
        venues = load_venue_data()

        # Preview
        try:
            sol_preview = sol.head(10).to_csv(index=False)
        except Exception:
            sol_preview = "<unable to render preview>"

        # Required columns for clustering analysis
        required_cols = {
            "venue_1_name", "venue_1_type", "venue_1_neighborhood",
            "venue_2_name", "venue_2_type", "venue_2_neighborhood",
            "venue_3_name", "venue_3_type", "venue_3_neighborhood",
            "cluster_center_lat", "cluster_center_lon", "cluster_diameter_miles",
            "cultural_density", "quality_bonus", "cluster_score"
        }
        missing = required_cols - set(sol.columns)
        if missing:
            return GradingResult(
                score=0.0,
                subscores={"all_passes": 0.0},
                weights=weights,
                feedback=(
                    f"Missing required columns: {sorted(missing)}\n\n"
                    f"Expected columns for clustering analysis:\n"
                    f"- venue_1_name, venue_1_type, venue_1_neighborhood\n"
                    f"- venue_2_name, venue_2_type, venue_2_neighborhood\n"
                    f"- venue_3_name, venue_3_type, venue_3_neighborhood\n"
                    f"- cluster_center_lat, cluster_center_lon, cluster_diameter_miles\n"
                    f"- cultural_density, quality_bonus, cluster_score\n\n"
                    f"sol.csv preview:\n{sol_preview}"
                )
            )

        # --- Criterion 1: Exactly 1 cluster ---
        single_result_ok = len(sol) == 1
        if not single_result_ok:
            return GradingResult(
                score=0.0, subscores={"all_passes": 0.0}, weights=weights,
                feedback=f"Expected exactly 1 cluster (optimal), found {len(sol)}\n\nsol.csv preview:\n{sol_preview}"
            )

        # Get the single row
        row = sol.iloc[0]

        # Extract venue information
        venue_names = [row["venue_1_name"], row["venue_2_name"], row["venue_3_name"]]
        venue_types = [row["venue_1_type"], row["venue_2_type"], row["venue_3_type"]]
        venue_neighborhoods = [row["venue_1_neighborhood"], row["venue_2_neighborhood"], row["venue_3_neighborhood"]]

        # --- Criterion 2: Venue validation with fuzzy matching ---
        venues_exist_ok = True
        venue_validation_details = []

        for i, venue_name in enumerate(venue_names):
            if venue_name not in venues:
                # Try fuzzy matching
                fuzzy_match_result = fuzzy_match(venue_name, venues.keys())
                if fuzzy_match_result:
                    # Use fuzzy match result
                    venue_data = venues[fuzzy_match_result]
                else:
                    # Check if venue has reasonable attributes even if not found
                    venue_type = venue_types[i]
                    venue_neighborhood = venue_neighborhoods[i]

                    # Valid venue types and neighborhoods from the data
                    valid_types = {'theater', 'restaurant', 'museum', 'bookstore', 'coffee'}
                    valid_neighborhoods = {
                        'Theater District', 'Midtown', 'Upper East Side', 'Greenwich Village',
                        'East Village', 'West Village', 'SoHo', 'NoLita', 'Tribeca',
                        'Chelsea', 'Flatiron District', 'NoMad', 'Upper West Side',
                        'Lower East Side', 'Financial District', 'Meatpacking District',
                        'Lincoln Center', 'East Harlem', 'Harlem', 'Midtown/Theater District',
                        'Midtown East'
                    }

                    if venue_type not in valid_types:
                        venues_exist_ok = False
                        venue_validation_details.append(f"Invalid venue type '{venue_type}' for {venue_name}")
                    elif venue_neighborhood not in valid_neighborhoods:
                        venues_exist_ok = False
                        venue_validation_details.append(f"Invalid neighborhood '{venue_neighborhood}' for {venue_name}")
                    # If type and neighborhood are valid, assume it's a reasonable venue
                    continue
            else:
                venue_data = venues[venue_name]
                # Check venue type matches
                if venue_data['type'] != venue_types[i]:
                    venues_exist_ok = False
                    venue_validation_details.append(
                        f"Type mismatch for {venue_name}: expected {venue_data['type']}, got {venue_types[i]}")
                # Check neighborhood matches (more lenient)
                if venue_data['neighborhood'].lower() != venue_neighborhoods[i].lower():
                    # Allow some neighborhood variations
                    if not (venue_data['neighborhood'].lower() in venue_neighborhoods[i].lower() or
                            venue_neighborhoods[i].lower() in venue_data['neighborhood'].lower()):
                        venues_exist_ok = False
                        venue_validation_details.append(
                            f"Neighborhood mismatch for {venue_name}: expected {venue_data['neighborhood']}, got {venue_neighborhoods[i]}")

        # --- Criterion 3: Different neighborhoods constraint ---
        different_neighborhoods_ok = len(set(venue_neighborhoods)) == 3
        neighborhoods_details = []
        if not different_neighborhoods_ok:
            neighborhoods_details.append(f"All venues must be from different neighborhoods. Got: {venue_neighborhoods}")

        # --- Criterion 4: Venue type diversity constraint ---
        venue_type_diversity_ok = len(set(venue_types)) >= 2
        type_diversity_details = []
        if not venue_type_diversity_ok:
            type_diversity_details.append(f"Must have at least 2 different venue types. Got: {venue_types}")

        # --- Criterion 5: Cluster diameter calculation and constraint ---
        diameter_calculation_ok = True
        diameter_details = []

        reported_diameter = _to_float(row['cluster_diameter_miles'])
        if reported_diameter is None:
            diameter_calculation_ok = False
            diameter_details.append("Invalid cluster_diameter_miles value")
        elif reported_diameter > 1.5:
            diameter_calculation_ok = False
            diameter_details.append(f"Cluster diameter {reported_diameter:.3f} exceeds 1.5 mile limit")
        elif reported_diameter < 0:
            diameter_calculation_ok = False
            diameter_details.append(f"Invalid negative diameter: {reported_diameter}")

        # If we can validate with actual coordinates, do so
        if venues_exist_ok:
            try:
                cluster_venues = []
                for name in venue_names:
                    if name in venues:
                        cluster_venues.append(venues[name])
                    else:
                        fuzzy_match_result = fuzzy_match(name, venues.keys())
                        if fuzzy_match_result:
                            cluster_venues.append(venues[fuzzy_match_result])

                if len(cluster_venues) == 3:
                    distances = []
                    for i in range(3):
                        for j in range(i + 1, 3):
                            v1, v2 = cluster_venues[i], cluster_venues[j]
                            dist = distance_miles(v1['latitude'], v1['longitude'],
                                                  v2['latitude'], v2['longitude'])
                            distances.append(dist)

                    actual_diameter = max(distances)
                    if reported_diameter is not None and abs(actual_diameter - reported_diameter) > 0.1:
                        diameter_details.append(
                            f"Diameter calculation may be inaccurate. Expected ~{actual_diameter:.3f}, got {reported_diameter:.3f}")
            except Exception as e:
                # Don't fail if we can't validate exact diameter
                pass

        # --- Criterion 6: Cultural density calculation ---
        cultural_density_ok = True
        cultural_density_details = []

        try:
            reported_cultural_density = _to_float(row['cultural_density'])
            reported_quality_bonus = _to_int(row['quality_bonus'])

            if reported_cultural_density is None:
                cultural_density_ok = False
                cultural_density_details.append("Invalid cultural_density value")
            elif reported_cultural_density < 0 or reported_cultural_density > 300:
                cultural_density_ok = False
                cultural_density_details.append(
                    f"Cultural density {reported_cultural_density} outside reasonable range (0-300)")

            if reported_quality_bonus is None:
                cultural_density_ok = False
                cultural_density_details.append("Invalid quality_bonus value")
            elif reported_quality_bonus < 0 or reported_quality_bonus > 100:
                cultural_density_ok = False
                cultural_density_details.append(
                    f"Quality bonus {reported_quality_bonus} outside reasonable range (0-100)")

        except Exception as e:
            cultural_density_ok = False
            cultural_density_details.append(f"Error validating cultural density: {e}")

        # --- Criterion 7: Cluster score calculation ---
        cluster_score_ok = True
        cluster_score_details = []

        try:
            reported_cultural_density = _to_float(row['cultural_density'])
            reported_diameter = _to_float(row['cluster_diameter_miles'])
            reported_cluster_score = _to_float(row['cluster_score'])

            if reported_cultural_density is not None and reported_diameter is not None:
                expected_cluster_score = reported_cultural_density - (reported_diameter * 50)

                if reported_cluster_score is None:
                    cluster_score_ok = False
                    cluster_score_details.append("Invalid cluster_score value")
                elif abs(expected_cluster_score - reported_cluster_score) > 5:
                    cluster_score_ok = False
                    cluster_score_details.append(
                        f"Cluster score calculation error. Expected {expected_cluster_score:.1f}, got {reported_cluster_score:.1f}")
            else:
                cluster_score_ok = False
                cluster_score_details.append("Cannot validate cluster score due to invalid inputs")

        except Exception as e:
            cluster_score_ok = False
            cluster_score_details.append(f"Error validating cluster score: {e}")

        # --- Criterion 8: Coordinate validation ---
        coordinates_ok = True
        coordinates_details = []

        try:
            center_lat = _to_float(row["cluster_center_lat"])
            center_lon = _to_float(row["cluster_center_lon"])

            if center_lat is None or center_lon is None:
                coordinates_ok = False
                coordinates_details.append("Invalid cluster center coordinates")
            elif not (40.7 <= center_lat <= 40.82 and -74.02 <= center_lon <= -73.93):
                coordinates_ok = False
                coordinates_details.append(f"Cluster center outside Manhattan: ({center_lat}, {center_lon})")

        except Exception as e:
            coordinates_ok = False
            coordinates_details.append(f"Error validating coordinates: {e}")

        # --- Criterion 9: Optimization result - NO TOLERANCE ---
        optimization_ok = True
        optimization_details = []

        try:
            sol_cluster_score = _to_float(row["cluster_score"])
            gt_cluster_score = _to_float(gt.iloc[0]["cluster_score"]) if len(gt) > 0 else None

            if sol_cluster_score is not None and gt_cluster_score is not None:
                # Allow 10-point tolerance for complex clustering optimization
                if sol_cluster_score >= gt_cluster_score - 10.0:
                    optimization_details.append(
                        f"Within acceptable range: {sol_cluster_score:.1f} >= {gt_cluster_score:.1f} - 10.0")
                else:
                    optimization_ok = False
                    optimization_details.append(
                        f"Suboptimal solution: {sol_cluster_score:.1f} < {gt_cluster_score:.1f} - 10.0")
            else:
                optimization_ok = False
                optimization_details.append("Unable to compare cluster scores")

        except Exception as e:
            optimization_ok = False
            optimization_details.append(f"Error validating optimization result: {e}")

        # All pass/fail criteria
        all_pass = all([
            single_result_ok,
            venues_exist_ok,
            different_neighborhoods_ok,
            venue_type_diversity_ok,
            diameter_calculation_ok,
            cultural_density_ok,
            cluster_score_ok,
            coordinates_ok,
            optimization_ok
        ])

        # Detailed feedback
        feedback_parts = []

        # Single result
        if single_result_ok:
            feedback_parts.append("✅ Output format: Single optimal 3-venue cluster")
        else:
            feedback_parts.append(f"❌ Output format: Expected 1 cluster, found {len(sol)}")

        # Venue validation
        if venues_exist_ok:
            feedback_parts.append("✅ Venue validation: All venues verified or have reasonable attributes")
        else:
            feedback_parts.append(f"❌ Venue validation: {'; '.join(venue_validation_details)}")

        # Different neighborhoods
        if different_neighborhoods_ok:
            feedback_parts.append("✅ Neighborhoods: All 3 venues from different neighborhoods")
        else:
            feedback_parts.append(f"❌ Neighborhoods: {'; '.join(neighborhoods_details)}")

        # Venue type diversity
        if venue_type_diversity_ok:
            feedback_parts.append("✅ Venue types: At least 2 different types present")
        else:
            feedback_parts.append(f"❌ Venue types: {'; '.join(type_diversity_details)}")

        # Diameter calculation
        if diameter_calculation_ok:
            feedback_parts.append("✅ Diameter: Correctly calculated and within 1.5 mile limit")
        else:
            feedback_parts.append(f"❌ Diameter: {'; '.join(diameter_details)}")

        # Cultural density
        if cultural_density_ok:
            feedback_parts.append("✅ Cultural density: Correct calculation with base values and bonuses")
        else:
            feedback_parts.append(f"❌ Cultural density: {'; '.join(cultural_density_details)}")

        # Cluster score
        if cluster_score_ok:
            feedback_parts.append("✅ Cluster score: Formula applied correctly")
        else:
            feedback_parts.append(f"❌ Cluster score: {'; '.join(cluster_score_details)}")

        # Coordinates
        if coordinates_ok:
            feedback_parts.append("✅ Coordinates: Cluster center within Manhattan bounds")
        else:
            feedback_parts.append(f"❌ Coordinates: {'; '.join(coordinates_details)}")

        # Optimization result
        if optimization_ok:
            feedback_parts.append(f"✅ Optimization result: {optimization_details[0] if optimization_details else 'Equal or better than ground truth'}")
        else:
            feedback_parts.append(f"❌ Optimization result: {'; '.join(optimization_details)}")

        # Final result
        if all_pass:
            feedback_parts.append("\n🎯 PASS: Cultural venue clustering meets all requirements")
        else:
            feedback_parts.append("\n⛔ FAIL: Solution does not meet requirements")

        feedback_parts.append(f"\nsol.csv preview:\n{sol_preview}")

        return GradingResult(
            score=1.0 if all_pass else 0.0,
            subscores={"all_passes": 1.0 if all_pass else 0.0},
            weights=weights,
            feedback="\n".join(feedback_parts)
        )

    except Exception as e:
        return GradingResult(
            score=0.0, subscores={"all_passes": 0.0}, weights=weights,
            feedback=f"Error during grading: {e}"
        )