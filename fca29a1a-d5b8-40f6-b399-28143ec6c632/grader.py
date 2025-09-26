from dataclasses import dataclass

try:
    # Use Apex's type if present
    from apex_arena._types import GradingResult  # type: ignore
except Exception:
    # Fallback for images where apex_arena isn't importable at runtime
    @dataclass
    class GradingResult:
        score: float
        feedback: str


from pathlib import Path
import csv, json, math
from typing import List, Dict, Tuple
from apex_arena._types import GradingResult

DATA_DIR = Path("/workdir/data")
SUBMISSION_CSV = Path("/workdir/sol.csv")

# ---- Spec constants (must match task.yaml exactly) ----
SAT_THRESHOLD = 0.75
MIN_PRICE = 40000
MIN_SAFETY = 4
MIN_MPG = 25
MIN_DEALERS = 3
FED_BASELINE = 3.5
DEFAULT_MAX_APR = 8.5  # when no matching offer

SEG_MULT = {
    "luxury": 1.5, "ultra_luxury": 1.5, "premium": 1.3, "mainstream": 1.0,
    "economy": 0.8, "specialty": 1.2, "value": 0.9, "performance": 1.4, "family": 0.95
}

COLS = [
    "make","model","year","avg_price","depreciation_risk",
    "financing_exposure","market_volatility","inventory_liquidity_risk",
    "total_risk_score","rank"
]

# ---------- Helpers mirroring the prompt ----------

def _load():
    cars = json.loads((DATA_DIR / "cars.json").read_text())["cars"]
    dealers = json.loads((DATA_DIR / "dealers.json").read_text())["dealers"]
    offers = json.loads((DATA_DIR / "financing_incentives.json").read_text())["financing_offers"]
    return cars, dealers, offers

def _eligible_dealers_for_car(car_id: str, dealers: List[dict]) -> List[dict]:
    out = []
    for d in dealers:
        if float(d.get("customer_satisfaction", 0.0)) < SAT_THRESHOLD:
            continue
        for it in d.get("inventory", []):
            if it.get("car_id") == car_id:
                qty = float(it.get("qty_in_stock", 0)) + float(it.get("qty_in_transit", 0))
                if qty > 0:
                    out.append({"dealer": d, "item": it, "qty": qty})
                break
    return out

def _pop_variance(vals: List[float]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    mean = sum(vals) / n
    return sum((v - mean)**2 for v in vals) / n

def _depreciation_risk(depr_rate: float, segment: str) -> float:
    mult = SEG_MULT.get((segment or "").lower(), 1.0)
    return float(depr_rate) * mult

def _max_apr(make: str, model: str, offers: List[dict]) -> float:
    mx = None
    for off in offers:
        if make in off.get("applicable_makes", []):
            models = off.get("applicable_models", [])
            if (model in models) or ("all_luxury_models" in models):
                tiers = off.get("apr_tiers", [])
                for t in tiers:
                    apr = t.get("apr")
                    if apr is not None:
                        mx = float(apr) if mx is None else max(mx, float(apr))
    return mx if mx is not None else DEFAULT_MAX_APR

def _financing_exposure(max_apr: float) -> float:
    return max(0.0, float(max_apr) - FED_BASELINE)

def _market_volatility(inv_counts: List[float]) -> float:
    if len(inv_counts) <= 1:
        return 2.0
    mean = sum(inv_counts)/len(inv_counts)
    if mean == 0:
        return 2.5
    return _pop_variance(inv_counts) / mean

def _inventory_liq(turnover_days_list: List[float]) -> float:
    if not turnover_days_list:
        return 2.0
    cleaned = [30.0 if d == 0 else float(d) for d in turnover_days_list]
    return (sum(cleaned)/len(cleaned)) / 45.0

def _round(v: float, ndigits: int) -> float:
    return float(f"{v:.{ndigits}f}")

def _compute_for_model(model_key: Tuple[str,str,int], cars, dealers, offers):
    make, model, year = model_key
    # gather car rows for this model_key
    rows = [c for c in cars if c["make"]==make and c["model"]==model and int(c["year"])==year]
    prices = [float(c["price"]) for c in rows]
    if not prices:
        return None
    avg_price = sum(prices)/len(prices)

    # base attributes (same across rows of same model_key)
    c0 = rows[0]
    safety = float(c0["safety_rating"])
    mpg = float(c0["mpg_combined"])

    # dealers eligibility & metrics
    inv_counts, tdays = [], []
    # use car_id basis for dealer availability
    car_ids = {c["id"] for c in rows}
    for d in dealers:
        if float(d.get("customer_satisfaction", 0.0)) < SAT_THRESHOLD:
            continue
        seen = False
        for it in d.get("inventory", []):
            if it.get("car_id") in car_ids:
                qty = float(it.get("qty_in_stock", 0)) + float(it.get("qty_in_transit", 0))
                if qty > 0:
                    inv_counts.append(qty)
                    tdays.append(float(d.get("inventory_turnover_days", 0)))
                    seen = True
                    break
        # (no else – we only count qualifying dealers)

    # Eligibility
    if not (avg_price >= MIN_PRICE and safety >= MIN_SAFETY and mpg >= MIN_MPG and len(inv_counts) >= MIN_DEALERS):
        return None

    dep = _depreciation_risk(float(c0["depreciation_rate"]), c0.get("market_segment"))
    mx = _max_apr(make, model, offers)
    fin = _financing_exposure(mx)
    vol = _market_volatility(inv_counts)
    liq = _inventory_liq(tdays)
    total = 2.0*dep + 0.8*fin + 1.5*vol + liq

    # rounding per spec
    return {
        "make": make,
        "model": model,
        "year": year,
        "avg_price": _round(avg_price, 2),
        "depreciation_risk": _round(dep, 3),
        "financing_exposure": _round(fin, 2),
        "market_volatility": _round(vol, 2),
        "inventory_liquidity_risk": _round(liq, 2),
        "total_risk_score": _round(total, 3),
    }

def _ground_truth(cars, dealers, offers):
    # build model keys
    keys = {(c["make"], c["model"], int(c["year"])) for c in cars}
    rows = []
    for key in keys:
        rec = _compute_for_model(key, cars, dealers, offers)
        if rec: rows.append(rec)
    # deterministic sort: total risk, then make/model/year
    rows.sort(key=lambda r: (r["total_risk_score"], r["make"], r["model"], r["year"]))
    top5 = rows[:5]
    for i, r in enumerate(top5, 1):
        r["rank"] = i
    return top5

def _read_submission() -> List[Dict]:
    if not SUBMISSION_CSV.exists():
        raise FileNotFoundError("sol.csv not found at /workdir/sol.csv")
    with SUBMISSION_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLS:
            raise ValueError(f"CSV header mismatch. Expected: {COLS}, got: {reader.fieldnames}")
        rows = [r for r in reader]
    if len(rows) != 5:
        raise ValueError(f"Expected exactly 5 rows; got {len(rows)}")
    # parse & normalize types (as strings with given rounding are OK; we compare as strings)
    for r in rows:
        r["year"] = int(r["year"])
        for k in ("avg_price","depreciation_risk","financing_exposure","market_volatility",
                  "inventory_liquidity_risk","total_risk_score"):
            # round parsed floats back to spec decimals so compare is robust
            nd = {"avg_price":2,"depreciation_risk":3,"financing_exposure":2,
                  "market_volatility":2,"inventory_liquidity_risk":2,
                  "total_risk_score":3}[k]
            r[k] = float(f"{float(r[k]):.{nd}f}")
        r["rank"] = int(r["rank"])
    # ordering + ranks
    for i, r in enumerate(rows, 1):
        if r["rank"] != i:
            raise ValueError(f"Rank mismatch at row {i}: got {r['rank']}, expected {i}")
    # ensure ascending by total risk
    risks = [r["total_risk_score"] for r in rows]
    if risks != sorted(risks):
        raise ValueError("Rows not sorted by total_risk_score ascending.")
    return rows

def grade(submission_dir: str | None = None) -> GradingResult:
    try:
        cars, dealers, offers = _load()
        gt = _ground_truth(cars, dealers, offers)
        sub = _read_submission()

        # strict set + order equality
        for i, (a, b) in enumerate(zip(sub, gt), 1):
            for k in COLS:
                if k == "rank":  # already enforced ranks
                    continue
                if a[k] != b[k]:
                    return GradingResult(
                        score=0.0,
                        feedback=f"Row {i} column '{k}' mismatch. Got {a[k]!r}, expected {b[k]!r}."
                    )
        return GradingResult(score=1.0, feedback="Correct.")
    except Exception as e:
        return GradingResult(score=0.0, feedback=f"Failed: {e}")
