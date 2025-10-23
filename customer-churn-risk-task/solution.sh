#!/usr/bin/env bash
set -euo pipefail

# Customer Churn Risk Segmentation - Reference Solution
# Uses shared logic module to avoid duplication with grader

python3 << 'PYTHON_SCRIPT'
import json
import csv
import sys
from pathlib import Path

# Add parent directory to path to import shared module
sys.path.insert(0, str(Path(__file__).parent))

from churn_logic import (
    is_eligible, calculate_individual_churn_risk, index_data_by_customer,
    compute_segment_metrics, get_top_10_segments, get_cohort_quarter
)

DATA_DIR = Path("/workdir/data")
OUT_CSV = Path("/workdir/sol.csv")

def load_data():
    """Load all JSON data files."""
    customers = json.loads((DATA_DIR / "customers.json").read_text())
    transactions = json.loads((DATA_DIR / "transactions.json").read_text())
    support_tickets = json.loads((DATA_DIR / "support_tickets.json").read_text())
    usage_metrics = json.loads((DATA_DIR / "usage_metrics.json").read_text())
    cohort_benchmarks_list = json.loads((DATA_DIR / "cohort_benchmarks.json").read_text())
    
    # Convert benchmarks list to dict
    cohort_benchmarks = {cb["cohort_quarter"]: cb for cb in cohort_benchmarks_list}
    
    return customers, transactions, support_tickets, usage_metrics, cohort_benchmarks

def main():
    """Main processing function."""
    customers, transactions, support_tickets, usage_metrics, cohort_benchmarks = load_data()
    
    # Index data by customer_id
    txn_by_cust, tickets_by_cust, usage_by_cust = index_data_by_customer(
        transactions, support_tickets, usage_metrics
    )
    
    # Calculate individual customer scores
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
    
    # Write CSV
    header = [
        "segment_name", "customer_tier", "region", "cohort_quarter", "segment_size",
        "at_risk_count", "at_risk_percentage", "weighted_mean_risk", 
        "cohort_risk_multiplier", "segment_churn_risk", "rank"
    ]
    
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for seg in top10:
            writer.writerow([
                seg["segment_name"],
                seg["customer_tier"],
                seg["region"],
                seg["cohort_quarter"],
                seg["segment_size"],
                seg["at_risk_count"],
                f"{seg['at_risk_percentage']:.2f}",
                f"{seg['weighted_mean_risk']:.3f}",
                f"{seg['cohort_risk_multiplier']:.3f}",
                f"{seg['segment_churn_risk']:.3f}",
                seg["rank"]
            ])
    
    print(f"Written top 10 segments to {OUT_CSV}")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT