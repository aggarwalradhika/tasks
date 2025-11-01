"""
Binary grader for 'Supply Chain Route Anomaly Detection (Top 8 by anomaly_score)'.

What this grader does
- Recomputes ground truth (GT) ONLY from /workdir/data JSON files.
- Validates /workdir/sol.csv for:
  * Exact header & column order.
  * Exactly 8 data rows (no index/extra cols).
  * Exact string equality vs GT with fixed decimals:
      - delay_volatility_score, cost_inflation_score, weather_impact_correlation,
        capacity_utilization_penalty, anomaly_score: 3dp
      - rank: "1".."8"
  * Ordering: anomaly_score desc; ties by route_id.

Scoring
- Binary: 1.0 on a perfect match; otherwise 0.0 with helpful feedback.
"""

import csv, json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import hashlib
from collections import defaultdict
from datetime import datetime
import statistics

# ---- GradingResult (pydantic if available) ----
try:
    from pydantic import BaseModel, Field
    _USE_PYD = True
except Exception:
    _USE_PYD = False

    class BaseModel:
        def __init__(self, **kw): self.__dict__.update(kw)
        def model_dump(self): return self.__dict__

    def Field(default=None, **kw):
        return default

class GradingResult(BaseModel):
    """Platform-compatible result object."""
    score: float = Field(..., description="Final score in [0,1]")
    feedback: str = Field(default="")
    subscores: Dict[str, float] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)

DATA = Path("/workdir/data")
SOL  = Path("/workdir/sol.csv")

HEADER = [
    "route_id","route_name","origin_city","destination_city",
    "delay_volatility_score","cost_inflation_score","weather_impact_correlation",
    "capacity_utilization_penalty","anomaly_score","rank"
]

def _load_json(name: str):
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)

def _fmt3(x: float) -> str: return f"{x:.3f}"

# --- helpers for diagnostics ---
def _csv_to_string(rows: List[List[str]]) -> str:
    from io import StringIO
    s = StringIO()
    w = csv.writer(s)
    for r in rows:
        w.writerow(r)
    return s.getvalue()

def _try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _row_diff(exp_row: List[str], got_row: List[str]) -> Dict[str, Any]:
    diffs = []
    for idx, col in enumerate(HEADER):
        exp = exp_row[idx]
        got = got_row[idx] if idx < len(got_row) else ""
        if exp != got:
            e_num = _try_float(exp)
            g_num = _try_float(got)
            diffs.append({
                "column": col,
                "expected": exp,
                "found": got,
                "numeric_delta": (None if (e_num is None or g_num is None) else (g_num - e_num))
            })
    return {
        "mismatch_columns": [d["column"] for d in diffs],
        "mismatches": diffs
    }

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _recompute_ground_truth():
    """Deterministically recompute expected top-8 rows using task rules."""
    routes = _load_json("routes.json")
    shipments = _load_json("shipments.json")
    weather_events = _load_json("weather_events.json")
    warehouse_capacity = _load_json("warehouse_capacity.json")

    route_by_id = {r["route_id"]: r for r in routes}
    warehouse_by_id = {w["warehouse_id"]: w for w in warehouse_capacity}

    # Group by route and filter to last 60 days
    shipments_by_route = defaultdict(list)
    weather_by_route = defaultdict(list)
    
    ref_date = datetime.strptime("2025-10-30", "%Y-%m-%d")
    
    for s in shipments:
        ship_date = datetime.strptime(s["date"], "%Y-%m-%d")
        if (ref_date - ship_date).days <= 60:
            shipments_by_route[s["route_id"]].append(s)
    
    for w in weather_events:
        weather_date = datetime.strptime(w["date"], "%Y-%m-%d")
        if (ref_date - weather_date).days <= 60:
            weather_by_route[w["route_id"]].append(w)

    # Eligibility
    eligible = []
    for r in routes:
        if r.get("status") != "active":
            continue
        if int(r.get("monthly_volume", 0)) < 100:
            continue
        if len(shipments_by_route[r["route_id"]]) < 20:
            continue
        eligible.append(r["route_id"])

    # Features
    rows = []
    for rid in eligible:
        r = route_by_id[rid]
        route_shipments = shipments_by_route[rid]
        route_weather = weather_by_route[rid]
        
        # 1) delay_volatility_score
        delays = []
        for s in route_shipments:
            exp_time = datetime.strptime(s["expected_delivery_time"], "%Y-%m-%dT%H:%M:%SZ")
            act_time = datetime.strptime(s["actual_delivery_time"], "%Y-%m-%dT%H:%M:%SZ")
            delay_hours = (act_time - exp_time).total_seconds() / 3600
            delays.append(max(0, delay_hours))
        
        if len(delays) >= 2:
            std_dev = statistics.stdev(delays)
        else:
            std_dev = 0.0
        delay_volatility = min(1.0, std_dev / 24.0)
        
        # 2) cost_inflation_score
        cost_per_km_list = []
        for s in route_shipments:
            cost_per_km = float(s["cost"]) / float(r["distance_km"])
            cost_per_km_list.append(cost_per_km)
        
        if len(cost_per_km_list) > 0:
            median_cpk = statistics.median(cost_per_km_list)
            sorted_cpk = sorted(cost_per_km_list)
            p75_index = int(len(sorted_cpk) * 0.75)
            if p75_index >= len(sorted_cpk):
                p75_index = len(sorted_cpk) - 1
            percentile_75 = sorted_cpk[p75_index]
            
            if median_cpk > 0:
                cost_inflation_ratio = (percentile_75 - median_cpk) / median_cpk
            else:
                cost_inflation_ratio = 0.0
            cost_inflation = min(1.0, cost_inflation_ratio * 2.0)
        else:
            cost_inflation = 0.0
        
        # 3) weather_impact_correlation
        severe_weather_dates = set()
        for w in route_weather:
            if w.get("severity") in ["severe", "extreme"]:
                severe_weather_dates.add(w["date"])
        
        delayed_dates = set()
        for s in route_shipments:
            exp_time = datetime.strptime(s["expected_delivery_time"], "%Y-%m-%dT%H:%M:%SZ")
            act_time = datetime.strptime(s["actual_delivery_time"], "%Y-%m-%dT%H:%M:%SZ")
            delay_hours = (act_time - exp_time).total_seconds() / 3600
            if delay_hours > 6:
                delayed_dates.add(s["date"])
        
        severe_weather_days = len(severe_weather_dates)
        delayed_shipment_days = len(delayed_dates)
        
        if severe_weather_days > 0:
            expected_delayed_days = severe_weather_days * 0.7
            weather_correlation = max(0, (delayed_shipment_days - expected_delayed_days) / severe_weather_days)
            weather_correlation = min(1.0, weather_correlation)
        else:
            weather_correlation = 0.0
        
        # 4) capacity_utilization_penalty
        overutilization_count = 0
        total_shipments = len(route_shipments)
        
        for s in route_shipments:
            wh_id = s.get("destination_warehouse_id")
            if wh_id in warehouse_by_id:
                max_capacity = float(warehouse_by_id[wh_id]["max_weight_kg"])
                utilization = float(s["weight_kg"]) / max_capacity
                if utilization > 0.85:
                    overutilization_count += 1
        
        if total_shipments > 0:
            capacity_penalty = min(1.0, overutilization_count / total_shipments)
        else:
            capacity_penalty = 0.0
        
        anomaly_score = 3.0*delay_volatility + 2.5*cost_inflation + 4.0*weather_correlation + 3.5*capacity_penalty

        rows.append({
            "route_id": r["route_id"],
            "route_name": r["route_name"],
            "origin_city": r["origin_city"],
            "destination_city": r["destination_city"],
            "delay_volatility_score": _fmt3(delay_volatility),
            "cost_inflation_score": _fmt3(cost_inflation),
            "weather_impact_correlation": _fmt3(weather_correlation),
            "capacity_utilization_penalty": _fmt3(capacity_penalty),
            "anomaly_score": _fmt3(anomaly_score),
        })

    # Sort & take top-8
    rows.sort(key=lambda r: (-float(r["anomaly_score"]), r["route_id"]))
    rows = rows[:8]
    for i, r in enumerate(rows, 1):
        r["rank"] = str(i)

    gt = [HEADER]
    for r in rows:
        gt.append([r[k] for k in HEADER])
    return gt

def grade(transcript: str | None = None) -> GradingResult:
    """Compare /workdir/sol.csv against recomputed GT and return a binary score."""
    subscores: Dict[str, float] = {"exact_match": 0.0}
    weights:   Dict[str, float] = {"exact_match": 1.0}
    details:   Dict[str, Any]   = {}

    gt = _recompute_ground_truth()
    details["expected_csv"] = _csv_to_string(gt)
    details["expected_csv_sha256"] = _sha256_text(details["expected_csv"])

    if not SOL.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing /workdir/sol.csv",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # Read as table
    try:
        raw_df = pd.read_csv(SOL, dtype=str, keep_default_na=False, skip_blank_lines=True)
    except Exception as e:
        details["read_error"] = f"{type(e).__name__}: {e}"
        return GradingResult(
            score=0.0,
            feedback="Failed to read sol.csv as a table",
            subscores=subscores,
            weights=weights,
            details=details
        )

    def _strip_df_strings(df: pd.DataFrame) -> pd.DataFrame:
        return df.applymap(lambda x: x.strip() if isinstance(x, str) else x).astype(str)

    def _nonempty_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.replace("", pd.NA).dropna(how="all")

    df = _strip_df_strings(raw_df)
    df_ne = _nonempty_df(df).reset_index(drop=True)

    # Build GT DataFrame
    gt_header = gt[0]
    gt_rows = gt[1:]
    gt_df = pd.DataFrame(gt_rows, columns=gt_header, dtype=str)
    gt_df = _strip_df_strings(gt_df).reset_index(drop=True)

    # Header check
    found_header = df_ne.columns.tolist()
    if found_header != gt_header:
        details["found_header"] = found_header
        details["expected_header"] = gt_header
        extra_cols = [c for c in found_header if c not in gt_header]
        missing_cols = [c for c in gt_header if c not in found_header]
        if extra_cols: details["extra_columns"] = extra_cols
        if missing_cols: details["missing_columns"] = missing_cols
        return GradingResult(
            score=0.0,
            feedback="Incorrect header.",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # Row count check
    if len(df_ne) != 8:
        details["found_row_count"] = len(df_ne)
        details["expected_row_count"] = 8
        return GradingResult(
            score=0.0,
            feedback=f"Expected exactly 8 data rows, found {len(df_ne)}",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # Content check
    left  = df_ne.reset_index(drop=True)
    right = gt_df.reset_index(drop=True)

    if left.shape != right.shape:
        details["found_shape"] = left.shape
        details["expected_shape"] = right.shape
        return GradingResult(
            score=0.0,
            feedback=f"Shape mismatch: expected {right.shape}, found {left.shape}",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # Boolean-matrix mismatch count
    try:
        df_cmp = (left == right)
        import numpy as _np
        cmp_mat = df_cmp.to_numpy()
        mismatch_count = int(_np.count_nonzero(~cmp_mat))
    except Exception as e:
        details["comparison_error"] = f"{type(e).__name__}: {e}"
        mismatch_count = 1

    # Hard string-equality gate
    gt_rows_list   = [ [right.iloc[r, c] for c in range(right.shape[1])] for r in range(right.shape[0]) ]
    cand_rows_list = [ [left.iloc[r, c]  for c in range(left.shape[1])]  for r in range(left.shape[0])  ]
    hard_mismatch  = (cand_rows_list != gt_rows_list)

    if (mismatch_count > 0) or hard_mismatch:
        first_row_idx = None
        for r in range(len(gt_rows_list)):
            if r >= len(cand_rows_list) or cand_rows_list[r] != gt_rows_list[r]:
                first_row_idx = r + 1
                break
        if first_row_idx is None:
            try:
                bad = _np.argwhere(~cmp_mat)
                first_row_idx = int(bad[0, 0]) + 1 if bad.size else 1
            except:
                first_row_idx = 1

        details["first_mismatch_row"] = first_row_idx

        row_diffs = []
        reported = 0
        for r in range(len(gt_rows_list)):
            if r >= len(cand_rows_list):
                exp_row = gt_rows_list[r]
                got_row = []
            else:
                exp_row = gt_rows_list[r]
                got_row = cand_rows_list[r]
            if exp_row != got_row:
                rd = _row_diff(exp_row, got_row)
                rd["row_index"] = r + 1
                row_diffs.append(rd)
                reported += 1
                if reported >= 3:
                    break

        details["row_diffs"] = row_diffs
        details["diff_summary"] = "; ".join(
            [f"row {d['row_index']}: " + ", ".join(d["mismatch_columns"]) for d in row_diffs]
        )

        first = row_diffs[0]
        cols_list = ", ".join(first.get("mismatch_columns", [])) or "unknown columns"
        pairs = []
        for m in first.get("mismatches", []):
            pairs.append(f"{m['column']} (exp={m['expected']}, got={m['found']})")
        preview = "; ".join(pairs[:5]) if pairs else "no per-column details"
        more_note = "" if len(pairs) <= 5 else f" (+{len(pairs)-5} more)"

        return GradingResult(
            score=0.0,
            feedback=(
                f"Content mismatch at data row {first_row_idx}. "
                f"Columns differing: {cols_list}. "
                f"First-row diffs: {preview}{more_note}. "
                f"See details.row_diffs for full breakdown."
            ),
            subscores=subscores,
            weights=weights,
            details=details
        )

    subscores["exact_match"] = 1.0
    return GradingResult(
        score=1.0,
        feedback="Correct",
        subscores=subscores,
        weights=weights,
        details={"rows_checked": 8}
    )