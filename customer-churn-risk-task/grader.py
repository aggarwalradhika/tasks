"""
Customer Churn Risk Segmentation - Grader

Validates /workdir/sol.csv against ground truth computed from data files.
Uses shared logic module to ensure consistency with solution.

Checks:
1. Exactly 10 data rows
2. Correct header
3. All segment calculations correct
4. Correct sorting and ranking
5. Proper decimal formatting
6. Segment size >= 2 filter applied

Scoring: Binary (1.0 if all checks pass, 0.0 otherwise)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any
import json
import csv
import sys

# Add current directory to path to import shared module
sys.path.insert(0, str(Path(__file__).parent))

from churn_logic import (
    is_eligible, calculate_individual_churn_risk, index_data_by_customer,
    compute_segment_metrics, get_top_10_segments, get_cohort_quarter
)

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

HEADER = [
    "segment_name", "customer_tier", "region", "cohort_quarter", "segment_size",
    "at_risk_count", "at_risk_percentage", "weighted_mean_risk",
    "cohort_risk_multiplier", "segment_churn_risk", "rank"
]

def load_data():
    """Load all JSON data files."""
    customers = json.loads((DATA_DIR / "customers.json").read_text())
    transactions = json.loads((DATA_DIR / "transactions.json").read_text())
    support_tickets = json.loads((DATA_DIR / "support_tickets.json").read_text())
    usage_metrics = json.loads((DATA_DIR / "usage_metrics.json").read_text())
    cohort_benchmarks_list = json.loads((DATA_DIR / "cohort_benchmarks.json").read_text())
    
    cohort_benchmarks = {cb["cohort_quarter"]: cb for cb in cohort_benchmarks_list}
    
    return customers, transactions, support_tickets, usage_metrics, cohort_benchmarks

def compute_ground_truth():
    """Compute ground truth top 10 segments using shared logic."""
    customers, transactions, support_tickets, usage_metrics, cohort_benchmarks = load_data()
    
    # Index data by customer_id
    txn_by_cust, tickets_by_cust, usage_by_cust = index_data_by_customer(
        transactions, support_tickets, usage_metrics
    )
    
    # Calculate individual scores
    customer_scores = []
    
    for customer in customers:
        cid = customer["customer_id"]
        
        if not is_eligible(customer, txn_by_cust):
            continue
        
        tier = customer["customer_tier"].lower()
        region = customer["region"].lower()
        cohort_quarter = get_cohort_quarter(customer["account_created_date"])
        lifetime_value = float(customer.get("lifetime_value", 0))
        
        churn_risk = calculate_individual_churn_risk(
            customer,
            txn_by_cust[cid],
            tickets_by_cust[cid],
            usage_by_cust[cid],
            cohort_benchmarks
        )
        
        segment = f"{tier}_{region}_{cohort_quarter}"
        
        customer_scores.append({
            "customer_id": cid,
            "segment": segment,
            "tier": tier,
            "region": region,
            "cohort_quarter": cohort_quarter,
            "churn_risk": churn_risk,
            "lifetime_value": lifetime_value
        })
    
    # Aggregate by segment
    segment_data = compute_segment_metrics(customer_scores, cohort_benchmarks)
    
    # Get top 10
    top10 = get_top_10_segments(segment_data)
    
    return top10

def read_submission():
    """Read and validate submission CSV structure."""
    if not SUBMISSION_CSV.exists():
        raise FileNotFoundError("Missing /workdir/sol.csv")
    
    with open(SUBMISSION_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    
    if len(rows) == 0:
        raise ValueError("CSV is empty")
    
    return rows

def grade(transcript: str = None) -> GradingResult:
    """Grade the submission."""
    subscores = {"exact_match": 0.0}
    weights = {"exact_match": 1.0}
    details = {}
    
    try:
        # Compute ground truth
        gt = compute_ground_truth()
        details["expected_segments"] = [s["segment_name"] for s in gt]
        
        # Read submission
        try:
            sub_rows = read_submission()
        except Exception as e:
            return GradingResult(
                score=0.0,
                feedback=f"Error reading CSV: {e}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # Check header
        if sub_rows[0] != HEADER:
            details["found_header"] = sub_rows[0]
            details["expected_header"] = HEADER
            return GradingResult(
                score=0.0,
                feedback=f"Incorrect header. Expected: {HEADER}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # Check row count
        sub_data = sub_rows[1:]
        if len(sub_data) != 10:
            details["found_rows"] = len(sub_data)
            return GradingResult(
                score=0.0,
                feedback=f"Expected exactly 10 data rows, found {len(sub_data)}",
                subscores=subscores,
                weights=weights,
                details=details
            )
        
        # Check each row
        for i, (exp, got) in enumerate(zip(gt, sub_data), 1):
            if len(got) != len(HEADER):
                details["row_error"] = f"Row {i} has {len(got)} columns, expected {len(HEADER)}"
                return GradingResult(
                    score=0.0,
                    feedback=details["row_error"],
                    subscores=subscores,
                    weights=weights,
                    details=details
                )
            
            # Check each field
            exp_row = [
                exp["segment_name"],
                exp["customer_tier"],
                exp["region"],
                exp["cohort_quarter"],
                str(exp["segment_size"]),
                str(exp["at_risk_count"]),
                f"{exp['at_risk_percentage']:.2f}",
                f"{exp['weighted_mean_risk']:.3f}",
                f"{exp['cohort_risk_multiplier']:.3f}",
                f"{exp['segment_churn_risk']:.3f}",
                str(exp["rank"])
            ]
            
            for j, (field_name, exp_val, got_val) in enumerate(zip(HEADER, exp_row, got)):
                if exp_val != got_val.strip():
                    details["mismatch"] = {
                        "row": i,
                        "column": field_name,
                        "expected": exp_val,
                        "got": got_val.strip()
                    }
                    return GradingResult(
                        score=0.0,
                        feedback=f"Row {i}, column '{field_name}': expected '{exp_val}', got '{got_val.strip()}'",
                        subscores=subscores,
                        weights=weights,
                        details=details
                    )
        
        # All checks passed
        subscores["exact_match"] = 1.0
        return GradingResult(
            score=1.0,
            feedback="Correct",
            subscores=subscores,
            weights=weights,
            details={"rows_checked": 10}
        )
    
    except Exception as e:
        return GradingResult(
            score=0.0,
            feedback=f"Error during grading: {e}",
            subscores=subscores,
            weights=weights,
            details={"error": str(e), "type": type(e).__name__}
        )