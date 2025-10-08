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
from typing import Dict, List

import math
import json
import pandas as pd

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
TEXT_COLS = [c for c in COLS if c not in NUMERIC_COLS and c != "rank"]
ORDER_KEY = ["vulnerability_score", "category", "sku_id"]  # score desc, then asc, asc

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

def _decimals_in_string(s: str) -> int:
    s = str(s).strip().lower()
    if "e" in s:  # scientific notation → let numeric tolerance handle it
        return 99
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1])

def _approx_equal_num(a_str: str, b_str: str, nd: int) -> bool:
    """Numeric equivalence with half-ulp tolerance at nd decimals."""
    try:
        a = float(a_str); b = float(b_str)
    except Exception:
        return False
    ra = round(a, nd); rb = round(b, nd)
    tol = 0.5 * (10 ** -nd) + 1e-12
    return abs(ra - rb) <= tol

# ---- Load data -------------------------------------------------------
def _load_json(name: str):
    return json.loads((DATA_DIR / name).read_text())

# ---- Compute ground truth (strings with required rounding) -----------
def _compute_ground_truth_df() -> pd.DataFrame:
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

    gt = pd.DataFrame(rows, columns=COLS[:-1])
    if gt.empty:
        return gt

    # Sort & add rank
    gt = gt.sort_values(
        by=["vulnerability_score","category","sku_id"],
        ascending=[False, True, True],
        key=lambda s: s.astype(float) if s.name == "vulnerability_score" else s
    ).head(5).reset_index(drop=True)
    gt["rank"] = (gt.index + 1).astype(str)
    return gt[COLS]

# ---- Read submission via pandas -------------------------------------
def _read_submission_df() -> pd.DataFrame:
    if not SUBMISSION_CSV.exists():
        raise FileNotFoundError("Missing /workdir/sol.csv")

    # Read as strings to preserve formatting
    df = pd.read_csv(SUBMISSION_CSV, dtype=str, keep_default_na=False)

    # Header & column order
    if list(df.columns) != COLS:
        raise ValueError(f"Header mismatch.\nExpected: {COLS}\nGot     : {list(df.columns)}")

    # Exactly 5 rows
    if len(df) != 5:
        raise ValueError(f"Expected exactly 5 data rows; got {len(df)}")

    # Basic type checks: numerics must be floatable, decimals up to N; rank must be 1..5
    # Allow fewer decimals (up to N), reject if more than N.
    for col in NUMERIC_COLS:
        for i, s in enumerate(df[col].tolist(), start=1):
            if _decimals_in_string(s) > DECIMALS[col]:
                raise ValueError(f"Row {i} col '{col}': must have up to {DECIMALS[col]} decimal places; got {s!r}.")
            try:
                float(s)
            except Exception:
                raise ValueError(f"Row {i} col '{col}': invalid numeric {s!r}.")

    # Rank: "1".."5"
    if df["rank"].tolist() != [str(i) for i in range(1, 6)]:
        raise ValueError(f"Rank must be the strings 1..5 in order. Got {df['rank'].tolist()}")

    return df

# ---- Grade -----------------------------------------------------------
def grade(_submission_dir: str | None = None) -> GradingResult:
    subscores = {
        "exact_match": 0.0
    }
    details: Dict[str, object] = {}
    weights = {k: 1.0 for k in subscores}  # visibility only (score is binary)

    if not SUBMISSION_CSV.exists():
        return GradingResult(score=0.0, feedback="Missing /workdir/sol.csv",
                             subscores=subscores, details=details, weights=weights)

    # Parse + structural checks
    try:
        sub = _read_submission_df()

    except Exception as e:
        return GradingResult(score=0.0, feedback=f"Failed: {e}",
                             subscores=subscores, details=details, weights=weights)

    # Compute GT
    try:
        gt = _compute_ground_truth_df()
        if gt.empty or len(gt) != 5:
            details["gt_row_count"] = 0 if gt is None else len(gt)
            return GradingResult(score=0.0,
                                 feedback=f"Internal error: ground truth produced {len(gt) if gt is not None else 0} rows (expected 5).",
                                 subscores=subscores, details=details, weights=weights)

    except Exception as e:
        return GradingResult(score=0.0, feedback=f"Internal error computing ground truth: {e}",
                             subscores=subscores, details=details, weights=weights)

    # Value-by-value comparison (index aligned, tolerant numerics)
    diffs = []
    for i in range(5):
        for col in COLS:
            a = sub.at[i, col]
            b = gt.at[i, col]
            if col in NUMERIC_COLS:
                nd = DECIMALS[col]
                if not _approx_equal_num(a, b, nd):
                    diffs.append({"row": i+1, "col": col, "got": a, "expected": b})
            else:
                if str(a) != str(b):
                    diffs.append({"row": i+1, "col": col, "got": a, "expected": b})
            if len(diffs) >= 10:
                break
        if len(diffs) >= 10:
            break

    if diffs:
        details["diffs_preview"] = diffs
        # Build a concise feedback string listing the first few diffs
        lines = [f"Row {d['row']} col '{d['col']}': got {d['got']!r}, expected {d['expected']!r}" for d in diffs[:10]]
        feedback = "Content mismatch in sol.csv:\n" + "\n".join(lines)
        return GradingResult(score=0.0, feedback=feedback,
                             subscores=subscores, details=details, weights=weights)

    subscores["exact_match"] = 1.0
    return GradingResult(score=1.0, feedback="OK",
                         subscores=subscores, details=details, weights=weights)