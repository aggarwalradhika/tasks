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
import hashlib

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

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
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
    details["expected_csv_sha256"] = _sha256_text(details["expected_csv"])

    if not SOL.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing /workdir/sol.csv",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # --- Read as a table; normalize whitespace; drop fully-empty rows ---
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

    # helpers kept separate so comments/docstrings above remain unchanged
    def _strip_df_strings(df: pd.DataFrame) -> pd.DataFrame:
        return df.applymap(lambda x: x.strip() if isinstance(x, str) else x).astype(str)

    def _nonempty_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.replace("", pd.NA).dropna(how="all")

    df = _strip_df_strings(raw_df)
    df_ne = _nonempty_df(df).reset_index(drop=True)

    # Build GT DataFrame (strings only)
    gt_header = gt[0]
    gt_rows = gt[1:]
    gt_df = pd.DataFrame(gt_rows, columns=gt_header, dtype=str)
    gt_df = _strip_df_strings(gt_df).reset_index(drop=True)

    # ---- Header check via df.columns (order matters); also detect extra/missing columns ----
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

    # ---- Row count check ----
    if len(df_ne) != 5:
        details["found_row_count"] = len(df_ne)
        details["expected_row_count"] = 5
        return GradingResult(
            score=0.0,
            feedback=f"Expected exactly 5 data rows, found {len(df_ne)}",
            subscores=subscores,
            weights=weights,
            details=details
        )

    # ---- Content check (cell-by-cell) on normalized frames ----
    left  = df_ne.reset_index(drop=True)
    right = gt_df.reset_index(drop=True)

    # Check shapes match (they should after header & row count checks, but be safe)
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

    # 1) Boolean-matrix mismatch count (fast path)
    try:
        df_cmp = (left == right)
        import numpy as _np
        cmp_mat = df_cmp.to_numpy()
        mismatch_count = int(_np.count_nonzero(~cmp_mat))
    except Exception as e:
        # If comparison fails for any reason, fall back to row-by-row
        details["comparison_error"] = f"{type(e).__name__}: {e}"
        mismatch_count = 1  # Force detailed check

    # 2) Hard string-equality gate (authoritative)
    #    Build row-wise lists and compare to GT rows exactly.
    gt_rows_list   = [ [right.iloc[r, c] for c in range(right.shape[1])] for r in range(right.shape[0]) ]
    cand_rows_list = [ [left.iloc[r, c]  for c in range(left.shape[1])]  for r in range(left.shape[0])  ]
    hard_mismatch  = (cand_rows_list != gt_rows_list)

    if (mismatch_count > 0) or hard_mismatch:
        # First mismatching row index (1..5) using robust scan on hard lists
        first_row_idx = None
        for r in range(len(gt_rows_list)):
            if r >= len(cand_rows_list) or cand_rows_list[r] != gt_rows_list[r]:
                first_row_idx = r + 1
                break
        if first_row_idx is None:
            # fallback to boolean matrix detection (shouldn't happen)
            try:
                bad = _np.argwhere(~cmp_mat)
                first_row_idx = int(bad[0, 0]) + 1 if bad.size else 1
            except:
                first_row_idx = 1

        details["first_mismatch_row"] = first_row_idx

        # Build up to 3 detailed diffs
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

        # concise inline feedback
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

    # If we reach here, both checks agree there's no difference.
    subscores["exact_match"] = 1.0
    return GradingResult(
        score=1.0,
        feedback="Correct",
        subscores=subscores,
        weights=weights,
        details={"rows_checked": 5}
    )