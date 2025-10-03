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
import csv, json, math
from typing import List, Dict, Tuple

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

HEADER = [
    "sku_id","sku_name","category","mean_daily_demand",
    "safety_stock_ratio","in_transit_risk","warehouse_concentration",
    "supplier_reliability_penalty","vulnerability_score","rank"
]

# Rounding spec per column
NDEC = {
    "mean_daily_demand": 2,
    "safety_stock_ratio": 3,
    "in_transit_risk": 3,
    "warehouse_concentration": 3,
    "supplier_reliability_penalty": 3,
    "vulnerability_score": 3,
}

# ---- Helpers (pure Python; no numpy) --------------------------------
def _round_to_str(x: float, nd: int) -> str:
    """Return a decimal-fixed string representation of x with nd places."""
    return f"{x:.{nd}f}"

def _mean(xs: List[float]) -> float:
    """Population mean; returns 0.0 for empty input."""
    return sum(xs) / len(xs) if xs else 0.0

def _pop_stdev(xs: List[float]) -> float:
    """Population standard deviation (denominator = N). Returns 0.0 if empty."""
    if not xs:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))

def _sigmoid(x: float) -> float:
    """Sigmoid used by spec: 1/(1+exp(-x/10)). Note the division by 10."""
    return 1.0 / (1.0 + math.exp(-x / 10.0))

# ---- Loaders ---------------------------------------------------------
def _load_data():
    """Load all required JSON inputs from /workdir/data."""
    skus = json.loads((DATA_DIR / "skus.json").read_text())
    warehouses = json.loads((DATA_DIR / "warehouses.json").read_text())
    shipments = json.loads((DATA_DIR / "shipments.json").read_text())
    demand = json.loads((DATA_DIR / "demand_forecast.json").read_text())
    return skus, warehouses, shipments, demand

# ---- Metric Computation per Spec ------------------------------------
def _compute_ground_truth():
    """Compute the ground-truth top-5 rows according to the task specification.

    Implements:
      - Eligibility filters (lead time, demand mean≥5, active, ≥2 stocked WHs).
      - Metric formulas (safety_stock_ratio, in_transit_risk, Herfindahl, supplier penalty).
      - Total vulnerability score and sorting with tie-breakers.
      - Fixed-decimal string formatting before comparison.

    Returns:
        List[Dict[str, str]]: Exactly five dictionaries with all output columns,
        each value as string (pre-rounded), and rank as "1".."5".
    """
    skus, warehouses, shipments, demand = _load_data()

    # Index: 30-day demand per sku_id
    demand30: Dict[str, List[float]] = {}
    for rec in demand:
        arr = list(rec.get("daily_demand", []))[:30]
        demand30[rec["sku_id"]] = arr

    # Aggregate current stock per SKU per warehouse (only >0 counts)
    stock_by_sku_wh: Dict[str, Dict[str, int]] = {}
    for wh in warehouses:
        wid = wh["warehouse_id"]
        for it in wh.get("inventory", []):
            sku_id = it.get("sku_id")
            qty = int(it.get("current_stock", 0) or 0)
            if qty > 0 and sku_id:
                stock_by_sku_wh.setdefault(sku_id, {}).setdefault(wid, 0)
                stock_by_sku_wh[sku_id][wid] += qty

    # On-way quantities: statuses limited to ["in_transit","scheduled"]
    ONWAY_OK = {"in_transit", "scheduled"}
    on_way_qty: Dict[str, float] = {}
    for sh in shipments:
        if sh.get("status") in ONWAY_OK:
            sku_id = sh.get("sku_id")
            if sku_id:
                on_way_qty[sku_id] = on_way_qty.get(sku_id, 0.0) + float(sh.get("qty", 0) or 0.0)

    rows = []
    for sku in skus:
        if not sku.get("active", False):
            continue
        sku_id = sku["sku_id"]
        lt = float(sku.get("lead_time_days", 0.0))
        dd = demand30.get(sku_id, [])
        if len(dd) < 30:
            # Require full 30-day horizon
            continue
        mu = _mean(dd)
        if lt < 3.0:
            continue
        if mu < 5.0:
            continue

        # Warehouses with stock > 0
        whs = stock_by_sku_wh.get(sku_id, {})
        wh_count = sum(1 for v in whs.values() if v > 0)
        if wh_count < 2:
            continue

        # demand stddev with special rule
        sd = _pop_stdev(dd)
        if sd == 0.0:
            sd = 0.5

        # (1) safety_stock_ratio
        reorder_point = mu * lt + 1.65 * math.sqrt(lt * (sd ** 2))
        safety_stock_ratio = (reorder_point - mu * lt) / mu

        # (2) in_transit_risk
        ow = float(on_way_qty.get(sku_id, 0.0))
        denom = mu * lt + 1.0
        in_transit_risk = 1.0 - _sigmoid(ow / denom)

        # (3) warehouse_concentration (Herfindahl over >0 stocks)
        stocks_pos = [float(v) for v in whs.values() if v > 0]
        if len(stocks_pos) <= 1:
            warehouse_concentration = 1.0
        else:
            total = sum(stocks_pos)
            fracs = [v / total for v in stocks_pos]
            warehouse_concentration = sum(f * f for f in fracs)

        # (4) supplier_reliability_penalty
        suppliers = sku.get("suppliers", [])
        if suppliers:
            supplier_score = _mean([float(s["historical_on_time_rate"]) for s in suppliers])
        else:
            supplier_score = 0.0
        supplier_reliability_penalty = max(0.0, 1.0 - supplier_score)

        # Total score
        vulnerability_score = (
            2.5 * safety_stock_ratio +
            1.8 * in_transit_risk +
            2.0 * warehouse_concentration +
            3.0 * supplier_reliability_penalty
        )

        # Collect (with string rounding exactly as the output spec)
        rows.append({
            "sku_id": sku_id,
            "sku_name": sku["sku_name"],
            "category": sku["category"],
            "mean_daily_demand": _round_to_str(mu, NDEC["mean_daily_demand"]),
            "safety_stock_ratio": _round_to_str(safety_stock_ratio, NDEC["safety_stock_ratio"]),
            "in_transit_risk": _round_to_str(in_transit_risk, NDEC["in_transit_risk"]),
            "warehouse_concentration": _round_to_str(warehouse_concentration, NDEC["warehouse_concentration"]),
            "supplier_reliability_penalty": _round_to_str(supplier_reliability_penalty, NDEC["supplier_reliability_penalty"]),
            "vulnerability_score": _round_to_str(vulnerability_score, NDEC["vulnerability_score"]),
        })

    # Sort: score desc; tie-breaker by category asc, then sku_id asc
    def _sort_key(r):
        """Key function for final ordering: (-score, category, sku_id)."""
        return (-float(r["vulnerability_score"]), r["category"], r["sku_id"])

    rows.sort(key=_sort_key)
    top5 = rows[:5]
    for i, r in enumerate(top5, 1):
        r["rank"] = str(i)  # compare as strings for CSV parity

    return top5

# ---- Submission Reader & Checks -------------------------------------
def _read_submission() -> List[Dict[str, str]]:
    """Read and validate /workdir/sol.csv structure and formatting.

    Validates:
      - Header exact match and row count == 5.
      - Rank sequence 1..5 and sorted order per spec.
      - Decimal precision for numeric columns.

    Returns:
        A list of row dicts (strings preserved) for subsequent value checks.
    """
    if not SUBMISSION_CSV.exists():
        raise FileNotFoundError("Missing /workdir/sol.csv")
    with SUBMISSION_CSV.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("sol.csv is empty")

    header = rows[0]
    if header != HEADER:
        raise ValueError(f"Header mismatch.\nExpected: {HEADER}\nGot     : {header}")

    data = rows[1:]
    if len(data) != 5:
        raise ValueError(f"Expected exactly 5 data rows; got {len(data)}")

    # Convert to list of dicts (strings preserved)
    dict_rows: List[Dict[str, str]] = []
    for i, r in enumerate(data, 1):
        if len(r) != len(HEADER):
            raise ValueError(f"Row {i} has {len(r)} columns; expected {len(HEADER)}")
        rec = {HEADER[j]: r[j].strip() for j in range(len(HEADER))}
        dict_rows.append(rec)

    # Validate rank sequence 1..5 and ordering (desc by vulnerability_score, tiebreakers)
    for i, rec in enumerate(dict_rows, 1):
        if rec["rank"] != str(i):
            raise ValueError(f"Rank mismatch at row {i}: got {rec['rank']}, expected {i}")

    # Check the rows are in the correct sorted order per spec
    def _order_key(r):
        """Ordering key for verifying submission: (-score, category, sku_id)."""
        return (-float(r["vulnerability_score"]), r["category"], r["sku_id"])

    sorted_copy = sorted(dict_rows, key=_order_key)
    if sorted_copy != dict_rows:
        raise ValueError("Rows are not sorted by vulnerability_score desc with specified tie-breakers.")

    # Check numeric formatting (exact decimals)
    def _check_dp(name: str, s: str, nd: int):
        """Ensure s parses as float and re-renders with exactly nd decimals."""
        try:
            val = float(s)
        except Exception:
            raise ValueError(f"Invalid numeric value for {name}: {s!r}")
        back = _round_to_str(val, nd)
        if back != s:
            raise ValueError(f"{name} must have exactly {nd} decimals. Got {s!r}, expected {back!r}.")

    for rec in dict_rows:
        _check_dp("mean_daily_demand", rec["mean_daily_demand"], NDEC["mean_daily_demand"])
        _check_dp("safety_stock_ratio", rec["safety_stock_ratio"], NDEC["safety_stock_ratio"])
        _check_dp("in_transit_risk", rec["in_transit_risk"], NDEC["in_transit_risk"])
        _check_dp("warehouse_concentration", rec["warehouse_concentration"], NDEC["warehouse_concentration"])
        _check_dp("supplier_reliability_penalty", rec["supplier_reliability_penalty"], NDEC["supplier_reliability_penalty"])
        _check_dp("vulnerability_score", rec["vulnerability_score"], NDEC["vulnerability_score"])

    return dict_rows

# ---- Grade -----------------------------------------------------------
def grade(_submission_dir: str | None = None) -> GradingResult:
    """Main entry for Apex Arena.

    Returns:
        GradingResult: score=1.0 if submission exactly matches recomputed ground
        truth (including formatting and ordering), else score=0.0 with feedback.
    """
    subscores = {"exact_match": 0.0}
    weights = {"exact_match": 1.0}
    details: Dict[str, object] = {}
    try:
        gt = _compute_ground_truth()
        sub = _read_submission()

        diffs = []
        for i, (a, b) in enumerate(zip(sub, gt), 1):
            for col in HEADER:
                if a[col] != b[col]:
                    diffs.append({"row": i, "col": col, "got": a[col], "expected": b[col]})
                    if len(diffs) >= 5:
                        break
            if diffs:
                break

        if diffs:
            details["diffs_preview"] = diffs
            return GradingResult(
                score=0.0,
                feedback="Content mismatch in sol.csv.",
                subscores=subscores,
                details=details,
                weights=weights,
            )

        subscores["exact_match"] = 1.0
        return GradingResult(score=1.0, feedback="OK",subscores=subscores, details=details, weights=weights)
    except Exception as e:
        return GradingResult(score=0.0, feedback=f"Failed: {e}",subscores=subscores, details=details, weights=weights)
