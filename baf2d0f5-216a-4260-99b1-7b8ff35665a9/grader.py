"""Apex Arena grader for 'Supply-Chain Stockout Vulnerability'.

This grader recomputes the ground-truth top-5 SKUs directly from the read-only
JSON inputs in /workdir/data and validates a contestant's /workdir/sol.csv.

Checks performed:
- Exact header match and exactly 5 data rows.
- Rank must be 1..5 in order, and rows must be sorted by:
  vulnerability_score (desc), then category (asc), then sku_id (asc).
- Numeric columns must have exact fixed decimals (per spec).
- Values must exactly match recomputed ground truth (string-wise).

Scoring:
- score is BINARY (1.0 only if everything matches exactly; else 0.0).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import csv
import math
import json
import pandas as pd
import hashlib

# ---- GradingResult -----------------
try:
    from apex_arena._types import GradingResult  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class GradingResult:
        """Fallback grading result when Apex's type is unavailable."""
        score: float
        feedback: str
        subscores: Dict[str, float] = field(default_factory=dict)
        details: Dict[str, object] = field(default_factory=dict)
        weights: Dict[str, float] = field(default_factory=dict)

# ---- Paths & Spec ----------------------------------------------------
DATA_DIR = Path("/workdir/data")
SUBMISSION_CSV = Path("/workdir/sol.csv")

COLS = [
    "sku_id","sku_name","category","mean_daily_demand",
    "safety_stock_ratio","in_transit_risk","warehouse_concentration",
    "supplier_reliability_penalty","vulnerability_score","rank"
]

# Required decimals (used for rounding/equality); "up to" in task spec.
DECIMALS = {
    "mean_daily_demand": 2,
    "safety_stock_ratio": 3,
    "in_transit_risk": 3,
    "warehouse_concentration": 3,
    "supplier_reliability_penalty": 3,
    "vulnerability_score": 3,
}
NUMERIC_COLS = list(DECIMALS.keys())
TEXT_COLS = ["sku_id", "sku_name", "category"]

# ---- Utils -----------------------------------------------------------
def _round_to_str(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _pop_stdev(xs: List[float]) -> float:
    if not xs:
        return 0.0
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / 10.0))

def _try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _csv_to_string(rows: List[List[str]]) -> str:
    """Convert list of rows to CSV string."""
    from io import StringIO
    s = StringIO()
    w = csv.writer(s)
    for r in rows:
        w.writerow(r)
    return s.getvalue()

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---- Load data -------------------------------------------------------
def _load_json(name: str):
    return json.loads((DATA_DIR / name).read_text())

# ---- Compute ground truth (strings with required rounding) -----------
def _compute_ground_truth_rows() -> List[List[str]]:
    """Returns [header_row, data_row1, data_row2, ...]"""
    skus = _load_json("skus.json")
    warehouses = _load_json("warehouses.json")
    shipments = _load_json("shipments.json")
    demand = _load_json("demand_forecast.json")

    # Demand horizon (30d) per sku_id
    demand30 = {d["sku_id"]: list(d.get("daily_demand", []))[:30] for d in demand}

    # Stock per sku per warehouse (only >0)
    stock_by_sku_wh: Dict[str, Dict[str, int]] = {}
    for wh in warehouses:
        wid = wh["warehouse_id"]
        for it in wh.get("inventory", []):
            sid = it.get("sku_id"); q = int(it.get("current_stock", 0) or 0)
            if sid and q > 0:
                stock_by_sku_wh.setdefault(sid, {}).setdefault(wid, 0)
                stock_by_sku_wh[sid][wid] += q

    # On-way qty for allowed statuses
    onway = {}
    for sh in shipments:
        if sh.get("status") in {"in_transit","scheduled"}:
            sid = sh.get("sku_id")
            if sid:
                onway[sid] = onway.get(sid, 0.0) + float(sh.get("qty", 0) or 0.0)

    rows = []
    for sku in skus:
        if not sku.get("active", False):
            continue
        sid = sku["sku_id"]
        lt = float(sku.get("lead_time_days", 0.0))
        dd = demand30.get(sid, [])
        if len(dd) < 30:
            continue
        mu = _mean(dd)
        if lt < 3.0 or mu < 5.0:
            continue
        whs = stock_by_sku_wh.get(sid, {})
        if sum(1 for v in whs.values() if v > 0) < 2:
            continue

        sd = _pop_stdev(dd)
        if sd == 0.0:
            sd = 0.5

        reorder_point = mu * lt + 1.65 * (lt * (sd ** 2)) ** 0.5
        ssr = (reorder_point - mu * lt) / mu

        ow = float(onway.get(sid, 0.0))
        itr = 1.0 - _sigmoid(ow / (mu * lt + 1.0))

        stocks_pos = [float(v) for v in whs.values() if v > 0]
        if len(stocks_pos) <= 1:
            whc = 1.0
        else:
            total = sum(stocks_pos)
            fracs = [v / total for v in stocks_pos]
            whc = sum(f * f for f in fracs)

        sups = sku.get("suppliers", [])
        supplier_score = _mean([float(s["historical_on_time_rate"]) for s in sups]) if sups else 0.0
        penalty = max(0.0, 1.0 - supplier_score)

        score = 2.5*ssr + 1.8*itr + 2.0*whc + 3.0*penalty

        rows.append({
            "sku_id": sid,
            "sku_name": sku["sku_name"],
            "category": sku["category"],
            "mean_daily_demand": _round_to_str(mu, DECIMALS["mean_daily_demand"]),
            "safety_stock_ratio": _round_to_str(ssr, DECIMALS["safety_stock_ratio"]),
            "in_transit_risk": _round_to_str(itr, DECIMALS["in_transit_risk"]),
            "warehouse_concentration": _round_to_str(whc, DECIMALS["warehouse_concentration"]),
            "supplier_reliability_penalty": _round_to_str(penalty, DECIMALS["supplier_reliability_penalty"]),
            "vulnerability_score": _round_to_str(score, DECIMALS["vulnerability_score"]),
        })

    # Sort & take top-5; tie-break by category then sku_id
    rows.sort(key=lambda r: (-float(r["vulnerability_score"]), r["category"], r["sku_id"]))
    rows = rows[:5]
    for i, r in enumerate(rows, 1):
        r["rank"] = str(i)

    # Convert to list of lists format
    result = [COLS]  # Header
    for r in rows:
        result.append([r[col] for col in COLS])
    
    return result

# ---- Detailed row comparison ----------------------------------------

def _row_diff(exp_row: List[str], got_row: List[str], row_idx: int) -> Dict[str, Any]:
    """Compare two rows with numeric tolerance on decimals."""
    diffs = []
    for idx, col in enumerate(COLS):
        exp = exp_row[idx]
        got = got_row[idx] if idx < len(got_row) else ""

        # Try to interpret both as floats
        e_num = _try_float(exp)
        g_num = _try_float(got)

        if e_num is not None and g_num is not None and col in DECIMALS:
            # Get number of decimals to check for this column
            nd = DECIMALS.get(col, 3)
            # Compare numerically, not as strings
            if round(e_num, nd) != round(g_num, nd):
                diffs.append({
                    "column": col,
                    "expected": exp,
                    "found": got,
                    "numeric_delta": g_num - e_num
                })
        else:
            # Non-numeric comparison (string columns)
            if exp.strip() != got.strip():
                diffs.append({
                    "column": col,
                    "expected": exp,
                    "found": got,
                    "numeric_delta": None
                })

    return {
        "row_index": row_idx,
        "mismatch_columns": [d["column"] for d in diffs],
        "mismatches": diffs
    }


# ---- Grade -----------------------------------------------------------
def grade(_submission_dir: str | None = None) -> GradingResult:
    subscores = {"exact_match": 0.0}
    details: Dict[str, object] = {}
    weights = {"exact_match": 1.0}

    # Compute ground truth
    try:
        gt_rows = _compute_ground_truth_rows()
        if len(gt_rows) != 6:  # header + 5 data rows
            details["gt_row_count"] = len(gt_rows) - 1
            return GradingResult(
                score=0.0,
                feedback=f"Internal error: ground truth produced {len(gt_rows)-1} rows (expected 5).",
                subscores=subscores, details=details, weights=weights
            )
    except Exception as e:
        details["gt_error"] = str(e)
        return GradingResult(
            score=0.0, 
            feedback=f"Internal error computing ground truth: {e}",
            subscores=subscores, details=details, weights=weights
        )

    # Store expected output for debugging
    gt_csv_str = _csv_to_string(gt_rows)
    details["expected_csv"] = gt_csv_str
    details["expected_csv_sha256"] = _sha256_text(gt_csv_str)

    # Check file exists
    if not SUBMISSION_CSV.exists():
        return GradingResult(
            score=0.0, 
            feedback="Missing /workdir/sol.csv",
            subscores=subscores, details=details, weights=weights
        )

    # Read submission as raw rows (strings only)
    try:
        with open(SUBMISSION_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            sub_rows = []
            for row in reader:
                # Strip whitespace from each cell
                sub_rows.append([cell.strip() for cell in row])
    except Exception as e:
        details["read_error"] = str(e)
        return GradingResult(
            score=0.0,
            feedback=f"Failed to read sol.csv: {e}",
            subscores=subscores, details=details, weights=weights
        )

    # Remove completely empty rows
    sub_rows = [row for row in sub_rows if any(cell for cell in row)]
    
    if len(sub_rows) == 0:
        return GradingResult(
            score=0.0,
            feedback="sol.csv is empty",
            subscores=subscores, details=details, weights=weights
        )

    # Store submitted CSV for debugging
    details["submitted_csv"] = _csv_to_string(sub_rows)
    details["submitted_csv_sha256"] = _sha256_text(details["submitted_csv"])

    # Check header
    sub_header = sub_rows[0]
    exp_header = gt_rows[0]
    
    if sub_header != exp_header:
        details["found_header"] = sub_header
        details["expected_header"] = exp_header
        extra_cols = [c for c in sub_header if c not in exp_header]
        missing_cols = [c for c in exp_header if c not in sub_header]
        if extra_cols:
            details["extra_columns"] = extra_cols
        if missing_cols:
            details["missing_columns"] = missing_cols
        
        return GradingResult(
            score=0.0,
            feedback=f"Incorrect header. Expected: {exp_header}, Got: {sub_header}",
            subscores=subscores, details=details, weights=weights
        )

    # Check row count (should be exactly 5 data rows)
    sub_data_rows = sub_rows[1:]
    if len(sub_data_rows) != 5:
        details["found_row_count"] = len(sub_data_rows)
        details["expected_row_count"] = 5
        return GradingResult(
            score=0.0,
            feedback=f"Expected exactly 5 data rows, found {len(sub_data_rows)}",
            subscores=subscores, details=details, weights=weights
        )

    # Check each data row has correct number of columns
    for i, row in enumerate(sub_data_rows, 1):
        if len(row) != len(COLS):
            details["row_with_wrong_col_count"] = i
            details["expected_col_count"] = len(COLS)
            details["found_col_count"] = len(row)
            return GradingResult(
                score=0.0,
                feedback=f"Row {i} has {len(row)} columns, expected {len(COLS)}",
                subscores=subscores, details=details, weights=weights
            )

    # Detailed content comparison (cell-by-cell, exact string match)
    gt_data_rows = gt_rows[1:]
    all_diffs = []
    
    for i in range(5):
        exp_row = gt_data_rows[i]
        got_row = sub_data_rows[i]
        
        # Check if rows match exactly
        if exp_row != got_row:
            diff_info = _row_diff(exp_row, got_row, i + 1)
            all_diffs.append(diff_info)

    # If there are any differences, fail with detailed feedback
    if all_diffs:
        details["total_rows_with_mismatches"] = len(all_diffs)
        details["row_diffs"] = all_diffs[:5]  # Show first 5 rows with issues
        
        # Build detailed feedback message
        first_diff = all_diffs[0]
        first_row_idx = first_diff["row_index"]
        cols_list = ", ".join(first_diff["mismatch_columns"])
        
        # Show detailed mismatches for first row
        lines = [f"Content mismatch at data row {first_row_idx}."]
        lines.append(f"Columns differing: {cols_list}")
        lines.append("")
        lines.append("Details for first mismatched row:")
        
        for mismatch in first_diff["mismatches"][:5]:  # Show first 5 column diffs
            col = mismatch["column"]
            exp = mismatch["expected"]
            got = mismatch["found"]
            
            if mismatch["numeric_delta"] is not None:
                delta = mismatch["numeric_delta"]
                lines.append(f"  • {col}: expected={exp}, got={got}, delta={delta:.6f}")
            else:
                lines.append(f"  • {col}: expected={exp!r}, got={got!r}")
        
        if len(first_diff["mismatches"]) > 5:
            lines.append(f"  ... and {len(first_diff['mismatches']) - 5} more column(s) in this row")
        
        if len(all_diffs) > 1:
            lines.append("")
            lines.append(f"Total: {len(all_diffs)} row(s) with mismatches. See details.row_diffs for full breakdown.")
        
        feedback = "\n".join(lines)
        
        return GradingResult(
            score=0.0,
            feedback=feedback,
            subscores=subscores,
            details=details,
            weights=weights
        )

    # All checks passed!
    subscores["exact_match"] = 1.0
    details["rows_checked"] = 5
    return GradingResult(
        score=1.0,
        feedback="Correct",
        subscores=subscores,
        details=details,
        weights=weights
    )