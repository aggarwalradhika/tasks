"""
Shared logic for customer churn risk calculation.
Used by both solution.sh and grader.py to ensure consistency.
"""

import math
from datetime import datetime
from collections import defaultdict

ANALYSIS_DATE = datetime(2025, 9, 1)
TIER_EXPECTED_FEATURES = {"bronze": 2, "silver": 3, "gold": 4, "platinum": 5}

def parse_date(date_str):
    """Parse YYYY-MM-DD format date string."""
    return datetime.strptime(date_str, "%Y-%m-%d")

def days_between(date1, date2):
    """Calculate days between two dates."""
    return abs((date2 - date1).days)

def mean(values):
    """Calculate mean, return 0.0 for empty list."""
    return sum(values) / len(values) if values else 0.0

def pop_stddev(values):
    """Calculate population standard deviation."""
    if not values or len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def get_cohort_quarter(date_str):
    """Convert date to cohort quarter format Q{1-4}_{year}."""
    dt = parse_date(date_str)
    quarter = (dt.month - 1) // 3 + 1
    return f"Q{quarter}_{dt.year}"

def is_eligible(customer, transactions_by_customer):
    """Check if customer meets eligibility criteria."""
    if customer.get("account_status", "").lower() != "active":
        return False
    
    created_date = parse_date(customer["account_created_date"])
    account_age = days_between(created_date, ANALYSIS_DATE)
    if account_age < 180:
        return False
    
    customer_id = customer["customer_id"]
    if customer_id not in transactions_by_customer or len(transactions_by_customer[customer_id]) == 0:
        return False
    
    tier = customer.get("customer_tier", "").lower()
    if tier not in ["bronze", "silver", "gold", "platinum"]:
        return False
    
    if float(customer.get("lifetime_value", 0)) < 100.0:
        return False
    
    return True

def calculate_tds(transactions, cohort_quarter, cohort_benchmarks):
    """Calculate Transaction Decline Score with cohort normalization."""
    recent = []
    prior = []
    
    for txn in transactions:
        txn_date = parse_date(txn["transaction_date"])
        days_ago = days_between(txn_date, ANALYSIS_DATE)
        
        if 0 <= days_ago <= 30:
            recent.append(float(txn["amount"]))
        elif 31 <= days_ago <= 90:
            prior.append(float(txn["amount"]))
    
    recent_avg = mean(recent)
    prior_avg = mean(prior)
    
    if prior_avg == 0:
        decline_ratio = 0.0
    else:
        decline_ratio = max(0.0, (prior_avg - recent_avg) / prior_avg)
    
    # Cohort normalization
    expected_decline = cohort_benchmarks.get(cohort_quarter, {}).get("expected_transaction_decline", 0.0)
    excess_decline = max(0.0, decline_ratio - expected_decline)
    
    return (decline_ratio * 60) + (excess_decline * 40)

def calculate_eds(usage):
    """Calculate Engagement Drop Score with volatility."""
    if not usage:
        return 0.0
    
    recent_sessions = 0
    prior_sessions = 0
    weekly_sessions = []
    
    # Calculate weekly sessions (13 weeks in 90 days)
    for week_start in range(0, 91, 7):
        week_total = 0
        for entry in usage:
            entry_date = parse_date(entry["date"])
            days_ago = days_between(entry_date, ANALYSIS_DATE)
            if week_start <= days_ago < week_start + 7:
                week_total += int(entry.get("daily_sessions", 0))
        if week_start < 90:
            weekly_sessions.append(week_total)
    
    # Recent vs prior
    for entry in usage:
        entry_date = parse_date(entry["date"])
        days_ago = days_between(entry_date, ANALYSIS_DATE)
        sessions = int(entry.get("daily_sessions", 0))
        
        if 0 <= days_ago <= 30:
            recent_sessions += sessions
        elif 31 <= days_ago <= 90:
            prior_sessions += sessions
    
    if prior_sessions == 0:
        drop_ratio = 0.0
    else:
        drop_ratio = max(0.0, (prior_sessions - recent_sessions) / prior_sessions)
    
    # Session volatility
    session_volatility = pop_stddev(weekly_sessions) / (mean(weekly_sessions) + 1)
    
    return (drop_ratio * 70) + (session_volatility * 30)

def calculate_sbs(tickets):
    """Calculate Support Burden Score with critical tickets."""
    if not tickets:
        return 0.0
    
    total_tickets = len(tickets)
    unresolved = 0
    critical = 0
    resolution_days = []
    
    for ticket in tickets:
        if ticket.get("status", "").lower() != "resolved":
            unresolved += 1
        else:
            res_days = float(ticket.get("resolution_days", 0))
            resolution_days.append(res_days)
        
        if ticket.get("priority", "").lower() == "critical":
            critical += 1
    
    avg_resolution = mean(resolution_days)
    
    ticket_score = (total_tickets * 1.5) + (unresolved * 4.0) + (critical * 6.0)
    resolution_penalty = min(avg_resolution * 2.0, 30.0)
    
    return ticket_score + resolution_penalty

def calculate_fag(customer, usage, tier):
    """Calculate Feature Adoption Gap with tier expectations."""
    features_available = len(customer.get("feature_access", []))
    
    if features_available == 0:
        return 0.0
    
    features_used = set()
    for entry in usage:
        entry_date = parse_date(entry["date"])
        if days_between(entry_date, ANALYSIS_DATE) <= 90:
            for feature in entry.get("feature_usage", []):
                features_used.add(feature)
    
    adoption_rate = len(features_used) / features_available
    
    expected_for_tier = TIER_EXPECTED_FEATURES.get(tier, 2)
    tier_gap = max(0, expected_for_tier - len(features_used))
    
    return ((1.0 - adoption_rate) * 70) + (tier_gap * 10)

def calculate_phs(transactions):
    """Calculate Payment Health Score."""
    if not transactions:
        return 0.0
    
    late_payments = 0
    failed_payments = 0
    total_payments = len(transactions)
    recent_count = 0
    prior_count = 0
    
    for txn in transactions:
        status = txn.get("payment_status", "").lower()
        if status == "late":
            late_payments += 1
        elif status == "failed":
            failed_payments += 1
        
        txn_date = parse_date(txn["transaction_date"])
        days_ago = days_between(txn_date, ANALYSIS_DATE)
        
        if 0 <= days_ago <= 30:
            recent_count += 1
        elif 31 <= days_ago <= 90:
            prior_count += 1
    
    if total_payments == 0:
        payment_issue_rate = 0.0
    else:
        payment_issue_rate = (late_payments + failed_payments * 2) / total_payments
    
    if prior_count == 0:
        frequency_decline = 0.0
    else:
        frequency_decline = max(0.0, (prior_count - recent_count) / prior_count)
    
    return (payment_issue_rate * 50) + (frequency_decline * 50)

def calculate_individual_churn_risk(customer, transactions, tickets, usage, cohort_benchmarks):
    """Calculate complete individual churn risk score."""
    cid = customer["customer_id"]
    tier = customer["customer_tier"].lower()
    cohort_quarter = get_cohort_quarter(customer["account_created_date"])
    
    tds = calculate_tds(transactions, cohort_quarter, cohort_benchmarks)
    eds = calculate_eds(usage)
    sbs = calculate_sbs(tickets)
    fag = calculate_fag(customer, usage, tier)
    phs = calculate_phs(transactions)
    
    return (tds * 0.25) + (eds * 0.20) + (sbs * 0.25) + (fag * 0.15) + (phs * 0.15)

def index_data_by_customer(transactions, support_tickets, usage_metrics):
    """Index all data sources by customer_id with time filtering."""
    transactions_by_customer = defaultdict(list)
    for txn in transactions:
        if days_between(parse_date(txn["transaction_date"]), ANALYSIS_DATE) <= 90:
            transactions_by_customer[txn["customer_id"]].append(txn)
    
    tickets_by_customer = defaultdict(list)
    for ticket in support_tickets:
        if days_between(parse_date(ticket["created_date"]), ANALYSIS_DATE) <= 90:
            tickets_by_customer[ticket["customer_id"]].append(ticket)
    
    usage_by_customer = defaultdict(list)
    for usage in usage_metrics:
        if days_between(parse_date(usage["date"]), ANALYSIS_DATE) <= 90:
            usage_by_customer[usage["customer_id"]].append(usage)
    
    return transactions_by_customer, tickets_by_customer, usage_by_customer

def compute_segment_metrics(customer_scores, cohort_benchmarks):
    """Aggregate customer scores into segment metrics."""
    segments = defaultdict(list)
    for score in customer_scores:
        segments[score["segment"]].append(score)
    
    segment_data = []
    for segment_name, scores_list in segments.items():
        segment_size = len(scores_list)
        
        # Filter: segment_size >= 2
        if segment_size < 2:
            continue
        
        at_risk_count = sum(1 for s in scores_list if s["churn_risk"] > 50.0)
        at_risk_pct = (at_risk_count / segment_size * 100) if segment_size > 0 else 0.0
        
        # Weighted mean risk
        total_weighted_risk = sum(s["churn_risk"] * s["lifetime_value"] for s in scores_list)
        total_weight = sum(s["lifetime_value"] for s in scores_list)
        weighted_mean_risk = total_weighted_risk / total_weight if total_weight > 0 else 0.0
        
        # Extract tier, region, cohort
        parts = segment_name.split("_")
        tier = parts[0]
        region = parts[1]
        cohort_quarter = "_".join(parts[2:])
        
        # Get cohort risk multiplier
        cohort_risk_multiplier = cohort_benchmarks.get(cohort_quarter, {}).get("cohort_risk_factor", 1.0)
        
        # Final segment risk
        segment_churn_risk = weighted_mean_risk * cohort_risk_multiplier
        
        segment_data.append({
            "segment_name": segment_name,
            "customer_tier": tier,
            "region": region,
            "cohort_quarter": cohort_quarter,
            "segment_size": segment_size,
            "at_risk_count": at_risk_count,
            "at_risk_percentage": at_risk_pct,
            "weighted_mean_risk": weighted_mean_risk,
            "cohort_risk_multiplier": cohort_risk_multiplier,
            "segment_churn_risk": segment_churn_risk
        })
    
    return segment_data

def get_top_10_segments(segment_data):
    """Sort segments and return top 10 with ranks."""
    segment_data.sort(
        key=lambda x: (-x["segment_churn_risk"], -x["at_risk_percentage"], 
                      -x["segment_size"], x["segment_name"].lower())
    )
    
    top10 = segment_data[:10]
    
    for rank, seg in enumerate(top10, 1):
        seg["rank"] = rank
    
    return top10