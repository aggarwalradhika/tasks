"""
Supply-Chain Bottleneck Scoring — Grader

Validates /workdir/sol.csv against ground truth derived from:
  /workdir/data/shipments.json
  /workdir/data/capacities.json

Contract:
  Header: node,delay_contribution_score,weighted_impact,normalized_impact
  Sorted by NI desc, WI desc, node asc (case-insensitive)
  All numeric fields exactly 4 decimals.
"""

from dataclasses import dataclass
from pathlib import Path
import json, csv, io, re
from statistics import median
from typing import Any

# Prefer arena's GradingResult if provided; otherwise use fallback
try:
    from apex_arena._types import GradingResult  # type: ignore
    _HAS_EXTERNAL_GR = True
except Exception:
    _HAS_EXTERNAL_GR = False
    @dataclass
    class GradingResult:  # type: ignore
        score: float
        feedback: str

DATA_DIR = Path("/workdir/data")
OUT_CSV  = Path("/workdir/sol.csv")
COLS = ["node","delay_contribution_score","weighted_impact","normalized_impact"]
DEC4 = re.compile(r"^-?\d+\.\d{4}$")

def _load() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load JSON inputs (shipments, capacities) from /workdir/data."""
    shipments = json.loads((DATA_DIR / "shipments.json").read_text(encoding="utf-8"))
    capacities = json.loads((DATA_DIR / "capacities.json").read_text(encoding="utf-8"))
    return shipments, capacities

def _canon_name(name: str, caps_idx: dict[str, dict]) -> str:
    """
    Canonicalize node casing using capacities index.
    If not found, Title Case the trimmed name.
    """
    n = (name or "").strip()
    key = n.casefold()
    if key in caps_idx:
        return caps_idx[key]["node"]
    return n.title()

def _default_capacity(cap_list: list[dict]) -> float:
    """
    Default capacity:
      - median of POSITIVE daily_capacity values
      - rounded to integer (bankers vs half-up doesn't matter for ints)
      - fallback 1.0
    """
    vals = [c.get("daily_capacity") for c in cap_list]
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and float(v) > 0]
    if not vals:
        return 1.0
    return float(int(round(median(vals))))

def _fmt4(x: float) -> str:
    """Format as fixed 4-decimal string."""
    return f"{float(x):.4f}"

def _int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def _ground_truth(shipments: list[dict], capacities: list[dict]) -> list[dict]:
    """
    Compute node metrics:
      1) delayed shipments (actual>expected), delay=actual-expected
      2) nodes per shipment = unique (case-insensitive) hops + destination
      3) per node: sum(delay), sum(quantity>=0)
      4) DCS = delay_sum / cap; WI = DCS * qty_sum; NI = WI/max(WI)
      5) format to 4dp; sort NI↓, WI↓, node↑(casefold)
    """
    # capacity index (case-insensitive)
    caps_idx: dict[str, dict] = {}
    for c in capacities:
        node = str(c.get("node") or "").strip()
        if not node:
            continue
        caps_idx[node.casefold()] = {"node": node, "cap": float(c.get("daily_capacity") or 0)}

    default_cap = _default_capacity(capacities)

    delayed = []
    for s in shipments:
        try:
            exp = _int(s.get("expected_days"))
            act = _int(s.get("actual_days"))
        except Exception:
            continue
        if act > exp:
            delayed.append({**s, "delay": act - exp})

    # expand nodes, unique within shipment (case-insensitive)
    total_delay: dict[str, float] = {}
    total_qty: dict[str, float] = {}

    for s in delayed:
        hops = [str(h) for h in (s.get("hops") or [])]
        dest = str(s.get("destination") or "")
        expanded = [*hops, dest]

        seen = set()
        uniq = []
        for n in expanded:
            k = (n or "").strip().casefold()
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(n)

        qty = max(0.0, float(_int(s.get("quantity"))))
        for n in uniq:
            canon = _canon_name(n, caps_idx)
            total_delay[canon] = total_delay.get(canon, 0.0) + float(s["delay"])
            total_qty[canon]   = total_qty.get(canon,   0.0) + qty

    rows = []
    for node in total_delay:
        cap = caps_idx.get(node.casefold(), {}).get("cap", default_cap)
        if cap <= 0:
            cap = default_cap
        dcs = total_delay[node] / cap
        wi  = dcs * total_qty.get(node, 0.0)
        rows.append({"node": node, "delay_contribution_score": dcs, "weighted_impact": wi})

    wi_max = max((r["weighted_impact"] for r in rows), default=0.0)
    for r in rows:
        r["normalized_impact"] = 0.0 if wi_max == 0 else (r["weighted_impact"] / wi_max)

    rows.sort(key=lambda r: (-r["normalized_impact"], -r["weighted_impact"], r["node"].casefold()))

    # print-as-strings with exactly 4 dp
    for r in rows:
        r["delay_contribution_score"] = _fmt4(r["delay_contribution_score"])
        r["weighted_impact"]          = _fmt4(r["weighted_impact"])
        r["normalized_impact"]        = _fmt4(r["normalized_impact"])
    return rows

def _read_submission() -> list[dict]:
    """
    Strictly read /workdir/sol.csv:
      - Header must match COLS exactly
      - Each numeric column must be DEC4
      - No extra/blank rows
    """
    if not OUT_CSV.exists():
        raise FileNotFoundError("Output CSV /workdir/sol.csv not found.")

    raw = OUT_CSV.read_text(encoding="utf-8-sig")
    rdr = csv.reader(io.StringIO(raw, newline=""))
    try:
        header = next(rdr)
    except StopIteration:
        raise ValueError("CSV is empty (no header).")

    if header != COLS:
        raise ValueError(f"CSV header mismatch. Expected {COLS}, got {header}")

    rows = []
    for line_no, row in enumerate(rdr, start=2):
        if not row:
            continue
        if len(row) != len(COLS):
            raise ValueError(f"Row {line_no}: expected {len(COLS)} columns, got {len(row)}")
        node, dcs_s, wi_s, ni_s = row
        for colname, val in (("delay_contribution_score", dcs_s),
                             ("weighted_impact", wi_s),
                             ("normalized_impact", ni_s)):
            if not DEC4.match(val):
                raise ValueError(f"Row {line_no} col '{colname}' must be a decimal with exactly 4 places: {val!r}")
        rows.append({"node": node, "delay_contribution_score": dcs_s, "weighted_impact": wi_s, "normalized_impact": ni_s})

    if not rows:
        raise ValueError("CSV has a header but no data rows.")
    return rows

def grade(_: str | None = None) -> GradingResult:
    """
    Compare contestant CSV to ground truth (row count, order, and values).
    Emits subscores/weights when host supports dynamic attributes.
    """
    try:
        shipments, capacities = _load()
        gt = _ground_truth(shipments, capacities)
        sub = _read_submission()

        subs = {"has_file": 1.0, "row_count": 0.0, "exact_match": 0.0}
        wts  = {"has_file": 0.05, "row_count": 0.05, "exact_match": 0.9}

        if len(sub) != len(gt):
            score = sum(subs[k]*wts[k] for k in subs)
            res = GradingResult(score=score, feedback=f"Row count mismatch. Expected {len(gt)}, got {len(sub)}.")
            for k,v in (("subscores",subs),("weights",wts)): 
                try: setattr(res, k, v)
                except Exception: pass
            return res

        subs["row_count"] = 1.0

        for i, (a, b) in enumerate(zip(sub, gt), 1):
            for k in COLS:
                if a[k] != b[k]:
                    score = sum(subs[x]*wts[x] for x in subs)
                    res = GradingResult(score=score, feedback=f"Row {i} col '{k}' mismatch: got={a[k]} expected={b[k]}")
                    for kk,vv in (("subscores",subs),("weights",wts)): 
                        try: setattr(res, kk, vv)
                        except Exception: pass
                    return res

        subs["exact_match"] = 1.0
        score = sum(subs[k]*wts[k] for k in subs)
        res = GradingResult(score=float(score), feedback="Correct.")
        for k,v in (("subscores",subs),("weights",wts)): 
            try: setattr(res, k, v)
            except Exception: pass
        return res

    except Exception as e:
        return GradingResult(score=0.0, feedback=f"Failed: {e}")
