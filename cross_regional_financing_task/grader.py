from pathlib import Path
import re
import pandas as pd
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
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


def load_risk_assessment_data():
    """Load cars, dealers and financing data for risk assessment analysis"""
    try:
        with open('/workdir/data/cars.json', 'r') as f:
            cars_data = json.load(f)
        with open('/workdir/data/dealers.json', 'r') as f:
            dealers_data = json.load(f)
        with open('/workdir/data/financing_incentives.json', 'r') as f:
            financing_data = json.load(f)
        return cars_data['cars'], dealers_data['dealers'], financing_data['financing_offers']
    except Exception:
        return [], [], []


def find_vehicle_in_data(make, model, year, cars):
    """Find vehicle in cars data"""
    return next((c for c in cars if c['make'] == make and c['model'] == model and c['year'] == year), None)


def count_dealers_with_vehicle(car_id, dealers):
    """Count dealers that have this vehicle in inventory with customer_satisfaction >= 0.75"""
    count = 0
    for dealer in dealers:
        if dealer.get('customer_satisfaction', 0) >= 0.75:
            if 'inventory' in dealer:
                for item in dealer['inventory']:
                    if item['car_id'] == car_id:
                        total_inventory = item['qty_in_stock'] + item.get('qty_in_transit', 0)
                        if total_inventory > 0:
                            count += 1
                            break
    return count


def validate_vehicle_constraints(make, model, year, cars, dealers):
    """Validate vehicle meets all task constraints"""
    car = find_vehicle_in_data(make, model, year, cars)
    if not car:
        return False, f"Vehicle {make} {model} {year} not found in dataset"

    # Price constraint: >= $40,000
    if car['price'] < 40000:
        return False, f"Price ${car['price']} below $40,000 minimum"

    # Safety rating >= 4
    if car['safety_rating'] < 4:
        return False, f"Safety rating {car['safety_rating']} below minimum of 4"

    # Combined MPG >= 25
    if car['mpg_combined'] < 25:
        return False, f"Combined MPG {car['mpg_combined']} below minimum of 25"

    # Must be available at >= 3 dealers
    dealer_count = count_dealers_with_vehicle(car['id'], dealers)
    if dealer_count < 3:
        return False, f"Available at only {dealer_count} dealers (minimum 3 required)"

    return True, ""


def validate_risk_components(row):
    """Validate risk components are reasonable values"""
    issues = []

    depreciation_risk = _to_float(row.get("depreciation_risk", ""))
    financing_exposure = _to_float(row.get("financing_exposure", ""))
    market_volatility = _to_float(row.get("market_volatility", ""))
    inventory_liquidity_risk = _to_float(row.get("inventory_liquidity_risk", ""))
    total_risk_score = _to_float(row.get("total_risk_score", ""))

    # Basic reasonableness checks
    if depreciation_risk is None or not (0 <= depreciation_risk <= 2.0):
        issues.append(f"depreciation_risk {depreciation_risk} unreasonable (should be 0-2.0)")

    if financing_exposure is None or not (0 <= financing_exposure <= 10.0):
        issues.append(f"financing_exposure {financing_exposure} unreasonable (should be 0-10.0)")

    if market_volatility is None or not (0 <= market_volatility <= 10.0):
        issues.append(f"market_volatility {market_volatility} unreasonable (should be 0-10.0)")

    if inventory_liquidity_risk is None or not (0 <= inventory_liquidity_risk <= 5.0):
        issues.append(f"inventory_liquidity_risk {inventory_liquidity_risk} unreasonable (should be 0-5.0)")

    # Check avg_price matches vehicle data if possible
    avg_price = _to_float(row.get("avg_price", ""))
    if avg_price is None or avg_price < 40000:
        issues.append(f"avg_price {avg_price} should be >= $40,000")

    return issues


def grade(transcript: str) -> GradingResult:
    """
    Grade the Cross-Regional Financing Risk Assessment solution.

    PASS iff:
      1) Required columns present and valid data types
      2) Exactly 5 vehicle models in output
      3) All constraint filtering applied (price >= $40K, safety >= 4, MPG >= 25, >= 3 dealers)
      4) Risk score components are reasonable values
      5) Results sorted by total_risk_score (ascending - lowest risk first)
      6) Ranks assigned correctly (1-5)
      7) Solution quality competitive with ground truth
    """
    gt_path = Path("/tests/answers.csv")  # reference solution
    pred_path = Path("/workdir/sol.csv")  # agent output

    weights = {"all_passes": 1.0}

    # File existence checks
    if not gt_path.exists():
        return GradingResult(
            score=0.0, subscores={}, weights=weights,
            feedback=f"Missing ground truth file: {gt_path}"
        )
    if not pred_path.exists():
        return GradingResult(
            score=0.0, subscores={}, weights=weights,
            feedback=f"Missing solution file: {pred_path}. Agent must create sol.csv in /workdir/"
        )

    try:
        # Load data
        gt = pd.read_csv(gt_path)
        sol = pd.read_csv(pred_path)
        cars, dealers, financing_offers = load_risk_assessment_data()

        # Preview for debugging
        try:
            sol_preview = sol.head(10).to_csv(index=False)
        except Exception:
            sol_preview = "<unable to render preview>"

        # Required columns check
        required_cols = {
            "make", "model", "year", "avg_price", "depreciation_risk",
            "financing_exposure", "market_volatility", "inventory_liquidity_risk",
            "total_risk_score", "rank"
        }
        missing = required_cols - set(sol.columns)
        if missing:
            return GradingResult(
                score=0.0,
                subscores={"all_passes": 0.0},
                weights=weights,
                feedback=(
                    f"Missing required columns: {sorted(missing)}\n"
                    f"Required: {sorted(required_cols)}\n"
                    f"Found: {sorted(sol.columns)}\n\n"
                    f"sol.csv preview:\n{sol_preview}"
                )
            )

        # Criterion 1: Exactly 5 vehicle models
        correct_count = len(sol) == 5
        if not correct_count:
            return GradingResult(
                score=0.0, subscores={"all_passes": 0.0}, weights=weights,
                feedback=f"Task requires exactly 5 vehicle models. Found {len(sol)} rows.\n\nsol.csv preview:\n{sol_preview}"
            )

        # Criterion 2: Data type validation
        data_validation_ok = True
        validation_details = []

        for i, row in sol.iterrows():
            make = row.get("make", "").strip() if pd.notna(row.get("make", "")) else ""
            model = row.get("model", "").strip() if pd.notna(row.get("model", "")) else ""
            year = _to_int(row.get("year", ""))
            avg_price = _to_float(row.get("avg_price", ""))
            total_risk_score = _to_float(row.get("total_risk_score", ""))
            rank = _to_int(row.get("rank", ""))

            # Validate required fields
            if not make or not model:
                data_validation_ok = False
                validation_details.append(f"Row {i + 1}: Missing make/model")

            if year is None or year < 2020 or year > 2025:
                data_validation_ok = False
                validation_details.append(f"Row {i + 1}: Invalid year")

            if avg_price is None or avg_price <= 0:
                data_validation_ok = False
                validation_details.append(f"Row {i + 1}: Invalid avg_price")

            if total_risk_score is None or total_risk_score <= 0:
                data_validation_ok = False
                validation_details.append(f"Row {i + 1}: Invalid total_risk_score")

            if rank is None or rank not in range(1, 6):
                data_validation_ok = False
                validation_details.append(f"Row {i + 1}: rank must be 1-5")

        # Criterion 3: Constraint validation
        constraints_ok = True
        constraint_details = []

        for i, row in sol.iterrows():
            make = row.get("make", "").strip()
            model = row.get("model", "").strip()
            year = _to_int(row.get("year", ""))

            if make and model and year:
                # Vehicle constraints
                is_valid, error_msg = validate_vehicle_constraints(make, model, year, cars, dealers)
                if not is_valid:
                    constraints_ok = False
                    constraint_details.append(f"Row {i + 1}: {error_msg}")

        # Criterion 4: Risk component validation (reasonableness checks)
        risk_validation_ok = True
        risk_details = []

        for i, row in sol.iterrows():
            issues = validate_risk_components(row)
            if issues:
                risk_validation_ok = False
                risk_details.extend([f"Row {i + 1}: {issue}" for issue in issues])

        # Criterion 5: Score ordering (ascending - lowest risk first)
        score_order_ok = True
        score_order_details = []

        scores = [_to_float(row.get("total_risk_score", "")) for _, row in sol.iterrows()]
        if all(s is not None for s in scores):
            for i in range(len(scores) - 1):
                if scores[i] > scores[i + 1]:
                    score_order_ok = False
                    score_order_details.append(
                        f"Risk scores not in ascending order: position {i + 1} ({scores[i]:.4f}) > position {i + 2} ({scores[i + 1]:.4f})")
                    break
        else:
            score_order_ok = False
            score_order_details.append("Cannot validate ordering due to invalid risk score data")

        # Criterion 6: Ranking consistency
        ranking_ok = True
        ranking_details = []

        ranks = [_to_int(row.get("rank", "")) for _, row in sol.iterrows()]
        expected_ranks = list(range(1, 6))

        if ranks != expected_ranks:
            ranking_ok = False
            ranking_details.append(f"Ranks should be [1, 2, 3, 4, 5]. Got: {ranks}")

        # Criterion 7: Solution quality compared to ground truth
        solution_quality_ok = True
        quality_details = []

        try:
            if len(gt) > 0:
                # Compare average risk scores (lower is better)
                gt_scores = [_to_float(row["total_risk_score"]) for _, row in gt.iterrows()
                             if _to_float(row["total_risk_score"]) is not None]
                sol_scores = [_to_float(row["total_risk_score"]) for _, row in sol.iterrows()
                              if _to_float(row["total_risk_score"]) is not None]

                if gt_scores and sol_scores:
                    gt_avg_risk = sum(gt_scores) / len(gt_scores)
                    sol_avg_risk = sum(sol_scores) / len(sol_scores)

                    # Only accept solutions equal to or better than ground truth (stricter evaluation)
                    if sol_avg_risk <= gt_avg_risk:
                        quality_details.append(
                            f"Solution quality: Equal or better than GT (avg risk {sol_avg_risk:.4f} vs GT {gt_avg_risk:.4f})")
                    else:
                        solution_quality_ok = False
                        quality_details.append(
                            f"Solution quality: Worse than GT (avg risk {sol_avg_risk:.4f} > GT {gt_avg_risk:.4f})")
                else:
                    quality_details.append("Cannot compare solution quality due to invalid data")
        except Exception as e:
            solution_quality_ok = False
            quality_details.append(f"Error comparing with ground truth: {e}")

        # Overall pass/fail
        all_pass = all([
            correct_count,
            data_validation_ok,
            constraints_ok,
            risk_validation_ok,
            score_order_ok,
            ranking_ok,
            solution_quality_ok
        ])

        # Construct feedback
        feedback_parts = []

        if correct_count:
            feedback_parts.append("Format: Exactly 5 vehicle models")
        else:
            feedback_parts.append(f"Format: Expected 5 models, found {len(sol)}")

        if data_validation_ok:
            feedback_parts.append("Data validation: All fields valid")
        else:
            feedback_parts.append(f"Data validation: {'; '.join(validation_details[:3])}")

        if constraints_ok:
            feedback_parts.append("Constraints: All requirements satisfied")
        else:
            feedback_parts.append(f"Constraints: {'; '.join(constraint_details[:3])}")

        if risk_validation_ok:
            feedback_parts.append("Risk calculation: Components are reasonable")
        else:
            feedback_parts.append(f"Risk calculation: {'; '.join(risk_details[:3])}")

        if score_order_ok:
            feedback_parts.append("Ordering: Risk scores in ascending order")
        else:
            feedback_parts.append(f"Ordering: {'; '.join(score_order_details[:1])}")

        if ranking_ok:
            feedback_parts.append("Ranking: Ranks assigned correctly (1-5)")
        else:
            feedback_parts.append(f"Ranking: {'; '.join(ranking_details[:1])}")

        if solution_quality_ok:
            feedback_parts.append(
                f"Solution quality: {quality_details[0] if quality_details else 'Meets expectations'}")
        else:
            feedback_parts.append(f"Solution quality: {'; '.join(quality_details[:1])}")

        # Final result
        if all_pass:
            feedback_parts.append(
                "\nPASS: Cross-Regional Financing Risk Assessment solution meets all requirements")
        else:
            feedback_parts.append("\nFAIL: Solution does not meet task requirements")

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