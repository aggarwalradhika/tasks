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
from typing import Dict, Any

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

def _load_json(name: str):
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)

def _minute_bucket(ts: str) -> str:
    from datetime import datetime
    # Truncate ISO timestamp like 'YYYY-MM-DDTHH:MM:SSZ' to 'YYYY-MM-DDTHH:MM'
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%dT%H:%M")

def _fmt2(x: float) -> str: return f"{x:.2f}"
def _fmt3(x: float) -> str: return f"{x:.3f}"

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

    header = [
        "account_id","campaign_id","campaign_name","spend_last_30d",
        "abnormal_click_rate","ip_aggregation_score","rapid_fire_clicks",
        "conversion_quality_penalty","fraud_score","rank"
    ]
    gt = [header]
    for r in rows:
        gt.append([r[k] for k in header])
    return gt

def grade(transcript: str | None = None) -> GradingResult:
    """Compare /workdir/sol.csv against recomputed GT and return a binary score."""
    subscores: Dict[str, float] = {"exact_match": 0.0}
    weights:   Dict[str, float] = {"exact_match": 1.0}
    details:   Dict[str, Any]   = {}

    if not SOL.exists():
        return GradingResult(
            score=0.0,
            feedback="Missing /workdir/sol.csv",
            subscores=subscores,
            weights=weights,
            details=details
        )

    gt = _recompute_ground_truth()

    with open(SOL, "r", encoding="utf-8") as f:
        lines = [list(row) for row in csv.reader(f)]

    if not lines:
        return GradingResult(
            score=0.0,
            feedback="sol.csv empty",
            subscores=subscores,
            weights=weights,
            details=details
        )

    if lines[0] != gt[0]:
        return GradingResult(
            score=0.0,
            feedback=f"Incorrect header. Expected: {gt[0]}",
            subscores=subscores,
            weights=weights,
            details=details
        )

    if len(lines) != 6:
        return GradingResult(
            score=0.0,
            feedback="Expected exactly 5 data rows",
            subscores=subscores,
            weights=weights,
            details=details
        )

    for i in range(1, 6):
        cand = lines[i]
        exp  = gt[i]
        if cand != exp:
            details.update({"first_mismatch_row": i, "expected": exp, "found": cand})
            return GradingResult(
                score=0.0,
                feedback=f"Row {i} mismatch.",
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
