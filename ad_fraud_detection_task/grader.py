"""
Binary grader for 'Cross-Channel Ad Fraud Detection (Top 5 by fraud_score)'.

What this grader does
- Recomputes ground truth (GT) ONLY from /workdir/data JSON files.
- Validates /workdir/sol.csv for:
  * Exact header & column order.
  * Exactly 5 data rows (no index/extra cols).
  * Exact string equality vs GT with fixed decimals:
      - spend_last_30d: 2dp
      - abnormal_click_rate, ip_aggregation_score, rapid_fire_clicks,
        conversion_quality_penalty, fraud_score: 3dp
      - rank: "1".."5"
  * Ordering: fraud_score desc; ties by account_id, then campaign_id.

Scoring
- Binary: 1.0 on a perfect match; otherwise 0.0 with helpful feedback.

Notes
- Platform compatibility: include subscores, weights, details as dicts (never None).
"""

import csv, json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# ---- GradingResult (pydantic if available) ----
try:
    from pydantic import BaseModel, Field
    _USE_PYD = True
except Exception:
    _USE_PYD = False

    class BaseModel:  # very small shim
        def __init__(self, **kw): self.__dict__.update(kw)
        def model_dump(self): return self.__dict__

    def Field(default=None, **kw):  # type: ignore
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
    "account_id","campaign_id","campaign_name","spend_last_30d",
    "abnormal_click_rate","ip_aggregation_score","rapid_fire_clicks",
    "conversion_quality_penalty","fraud_score","rank"
]

def _load_json(name: str):
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)

def _minute_bucket(ts: str) -> str:
    from datetime import datetime
    # Truncate ISO timestamp like 'YYYY-MM-DDTHH:MM:SSZ' to 'YYYY-MM-DDTHH:MM'
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%dT%H:%M")

def _fmt2(x: float) -> str: return f"{x:.2f}"
def _fmt3(x: float) -> str: return f"{x:.3f}"

# --- helpers added for better diagnostics (do not change existing comments above) ---
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
# --- end helpers ---

def _recompute_ground_truth():
    """
    Deterministically recompute expected top-5 rows using task rules:
      - Eligibility gates.
      - Feature computations (CTR, IP top-3, minute bursts, conversion penalty).
      - Sorting & ranking.
      - Fixed-decimal formatting for CSV comparison.
    """
    campaigns    = _load_json("campaigns.json")
    impressions  = _load_json("impressions.json")
    clicks       = _load_json("clicks.json")
    conversions  = _load_json("conversions.json")
    _            = _load_json("ip_metadata.json")  # not used (red herring)

    from collections import defaultdict, Counter

    camp_by_id = {c["campaign_id"]: c for c in campaigns}

    # Aggregates
    impr_by = defaultdict(int)                     # (campaign, channel) -> impressions
    for r in impressions:
        impr_by[(r["campaign_id"], r["channel"])] += int(r["impressions"])

    clk_by  = defaultdict(int)                     # (campaign, channel) -> clicks
    clk_min = defaultdict(lambda: defaultdict(int))# campaign -> minute -> clicks
    clk_ip  = defaultdict(Counter)                 # campaign -> ip counter

    for r in clicks:
        cid, ch, ip, ts = r["campaign_id"], r["channel"], r["ip"], r["ts"]
        clk_by[(cid, ch)] += 1
        clk_min[cid][_minute_bucket(ts)] += 1
        clk_ip[cid][ip] += 1

    conv_by = defaultdict(int)                     # campaign -> conversions
    for r in conversions:
        conv_by[r["campaign_id"]] += 1

    # Eligibility
    eligible = []
    for c in campaigns:
        if not c.get("active", False):
            continue
        if float(c.get("spend_last_30d", 0)) < 500:
            continue
        total_impr = sum(impr_by[(c["campaign_id"], ch)] for ch in c.get("channels", []))
        if total_impr < 2000:
            continue
        eligible.append(c["campaign_id"])

    # Features
    rows = []
    for cid in eligible:
        c = camp_by_id[cid]
        chans = c.get("channels", [])

        # 1) abnormal_click_rate
        ctrs = []
        for ch in chans:
            impr = impr_by[(cid, ch)]
            clk  = clk_by[(cid, ch)]
            ctrs.append((clk / impr) if impr > 0 else 0.0)
        channel_ctr_max = max(ctrs) if ctrs else 0.0

        expected_ctr = 0.02
        avg_map = c.get("average_ctr_by_channel")
        if isinstance(avg_map, dict) and avg_map:
            expected_ctr = sum(avg_map.values()) / len(avg_map)
        abnormal_click_rate = max(0.0, channel_ctr_max - expected_ctr)

        # 2) ip_aggregation_score
        ip_counter   = clk_ip[cid]
        total_clicks = sum(ip_counter.values())
        if total_clicks == 0:
            ip_aggregation_score = 0.0
        else:
            top3 = sum(v for _, v in ip_counter.most_common(3))
            ip_aggregation_score = min(1.0, top3 / total_clicks)

        # 3) rapid_fire_clicks
        mins = clk_min[cid]
        if mins:
            burst = sum(1 for _, cnt in mins.items() if cnt > 5)
            rapid_fire_clicks = max(0.0, min(1.0, burst / len(mins)))
        else:
            rapid_fire_clicks = 0.0

        # 4) conversion_quality_penalty
        convs = conv_by[cid]
        cr  = (convs / total_clicks) if total_clicks > 0 else 0.0
        ecr = float(c.get("expected_conversion_rate", 0.05))
        if ecr <= 0:
            ecr = 0.05
        penalty = ((ecr - cr) / ecr) if cr < ecr else 0.0
        penalty = max(0.0, min(1.0, penalty))

        fraud_score = 4.0 * abnormal_click_rate + 3.0 * ip_aggregation_score \
                      + 2.0 * rapid_fire_clicks + 3.5 * penalty

        rows.append({
            "account_id": c["account_id"],
            "campaign_id": c["campaign_id"],
            "campaign_name": c["campaign_name"],
            "spend_last_30d": _fmt2(float(c["spend_last_30d"])),
            "abnormal_click_rate": _fmt3(abnormal_click_rate),
            "ip_aggregation_score": _fmt3(ip_aggregation_score),
            "rapid_fire_clicks": _fmt3(rapid_fire_clicks),
            "conversion_quality_penalty": _fmt3(penalty),
            "fraud_score": _fmt3(fraud_score),
        })

    # Sort & take top-5; tie-break by account_id then campaign_id
    rows.sort(key=lambda r: (-float(r["fraud_score"]), r["account_id"], r["campaign_id"]))
    rows = rows[:5]
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

    if not SOL.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing /workdir/sol.csv",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # --- NEW: read as a table using pandas (all as strings, no NaNs) ---
    try:
        df = pd.read_csv(SOL, dtype=str, keep_default_na=False)
    except Exception as e:
        details["read_error"] = f"{type(e).__name__}: {e}"
        return GradingResult(
            score=0.0,
            feedback="Failed to read sol.csv as a table",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # Build GT as a DataFrame (strings only) for comparison
    gt_header = gt[0]
    gt_rows = gt[1:]  # list of lists
    gt_df = pd.DataFrame(gt_rows, columns=gt_header, dtype=str)

    # ---- Header check via df.columns (order matters) ----
    found_header = df.columns.tolist()
    if found_header != gt_header:
        details["found_header"] = found_header
        details["expected_header"] = gt_header
        return GradingResult(
            score=0.0,
            feedback="Incorrect header.",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # ---- Row-count check via df.shape ----
    if df.shape[0] != 5:
        details["found_row_count_excluding_header"] = int(df.shape[0])
        details["expected_row_count"] = 5
        return GradingResult(
            score=0.0,
            feedback="Expected exactly 5 data rows",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # Normalize row order (they must match GT order exactly)
    # Compare cell-by-cell equality as a boolean table
    df_cmp = (df.reset_index(drop=True) == gt_df.reset_index(drop=True))
    # Find all mismatching (row, col) coordinates
    import numpy as _np
    mism = _np.argwhere(~df_cmp.to_numpy())  # array of [row_idx, col_idx]

    if mism.size > 0:
        # first mismatching data row index (1..5 when counted like CSV data rows)
        first_row_idx = int(mism[0, 0]) + 1
        details["first_mismatch_row"] = first_row_idx

        # Build up to 3 detailed row diffs using your existing helper
        row_diffs = []
        seen_rows = set()
        for r, c in mism:
            if r in seen_rows:
                continue
            seen_rows.add(r)
            exp_row = gt_rows[r]
            got_row = df.iloc[r].tolist()
            rd = _row_diff(exp_row, got_row)
            rd["row_index"] = r + 1  # CSV-style data-row index (1..5)
            row_diffs.append(rd)
            if len(row_diffs) >= 3:
                break

        details["row_diffs"] = row_diffs
        details["diff_summary"] = "; ".join(
            [f"row {d['row_index']}: " + ", ".join(d["mismatch_columns"]) for d in row_diffs]
        )

        # Also put a concise, human-readable preview directly in feedback
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
        details={"rows_checked": 5}
    )