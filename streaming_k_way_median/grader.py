# tests/grader.py
"""
Grader for Streaming K-Way Percentiles (Sliding Window, Weights, Retractions).

This grader:
  - Loads /workdir/data/stream.jsonl with the same tolerant normalization rules
    used by the reference solution.
  - Recomputes the canonical expected CSV rows.
  - Validates the contestant's /workdir/sol.csv for:
      * exact schema (column names & order),
      * matching row count,
      * exact cell-by-cell equality.
  - Returns a structured GradingResult with subscores and optional diff details.

No dependency on answers.csv. No internet access.
"""
import csv, json, math
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# --- Pydantic result shape (shim if pydantic not installed) ---
try:
    from pydantic import BaseModel
except Exception:
    class BaseModel:
        def __init__(self, **kw): self.__dict__.update(kw)
        def model_dump(self): return self.__dict__

class GradingResult(BaseModel):
    score: float
    feedback: str | None = None
    subscores: dict = {}
    details: dict | None = None
    weights: dict | None = None

# ---- Task constants (MUST match task.yaml & solution.sh) ----
PCTS: List[int] = [10, 50, 90]
W: int = 6
COLS: List[str] = ["ingest_index","window_start_ingest","window_end_ingest","scope","key"] + [f"p{p}" for p in PCTS]

WORKDIR = Path("/workdir")
STREAM  = WORKDIR / "data" / "stream.jsonl"
SOL     = WORKDIR / "sol.csv"

# ----------------- Normalization (identical to solution) -----------------
def _normalize_line(obj, next_batch_id: int) -> dict | None:
    """
    Normalize a parsed JSON line into canonical object form or return None to skip.

    - Array line → data for key 'A', weight=1 per value, auto batch_id.
    - Object line with type in {'data','retract'}:
        * Ensure batch_id (auto-assign if missing).
        * For 'data': coerce record weights (<=0 or non-int) to 1 and drop malformed records.
    """
    if isinstance(obj, list):
        recs = [{"key": "A", "value": int(v), "weight": 1} for v in obj]
        return {"batch_id": next_batch_id, "type": "data", "records": recs}
    if isinstance(obj, dict):
        t = obj.get("type")
        if t in ("data", "retract"):
            if "batch_id" not in obj or not isinstance(obj["batch_id"], int):
                obj = {**obj, "batch_id": next_batch_id}
            if t == "data":
                recs = obj.get("records", [])
                if not isinstance(recs, list):
                    recs = []
                fixed = []
                for r in recs:
                    if not isinstance(r, dict): 
                        continue
                    k = r.get("key"); v = r.get("value"); w = r.get("weight", 1)
                    if not isinstance(k, str) or not isinstance(v, int):
                        continue
                    try: w = int(w)
                    except Exception: w = 1
                    if w <= 0: w = 1
                    fixed.append({"key": k, "value": int(v), "weight": int(w)})
                obj = {**obj, "records": fixed}
            return obj
    return None

def _load_stream(stream_path: Path) -> list[dict]:
    """
    Load/normalize the input stream with the tolerance rules described in task.yaml.
    - Ignores blank lines, '//' comment lines, and malformed JSON lines.
    - Auto-assigns batch_id in ingest order when missing.
    """
    objs: List[dict] = []
    if not stream_path.exists():
        return objs
    next_bid = 1
    for raw in stream_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        try:
            parsed = json.loads(s)
        except Exception:
            continue
        v = _normalize_line(parsed, next_bid)
        if v is None:
            continue
        objs.append(v)
        next_bid = max(next_bid, v.get("batch_id", next_bid)) + 1
    return objs

# ----------------- Canonical computation (identical to solution) -----------------
def _nearest_rank(counter: Dict[int, int]) -> List[str]:
    """Return weighted nearest-rank percentiles as strings over an int->count mapping."""
    N = sum(counter.values())
    if N == 0:
        return [""] * len(PCTS)
    items = sorted(counter.items())
    out = []
    for p in PCTS:
        r = math.ceil(p/100 * N)
        cum = 0
        for v, c in items:
            cum += c
            if cum >= r:
                out.append(str(v))
                break
    return out

def _expected_df(stream_objs: list[dict]) -> pd.DataFrame:
    """
    Compute canonical expected rows:
      - Sliding window of W ingests (by ingest order)
      - Retraction removes the target data-batch contributions if the target is in-window
      - GLOBAL row then per-key rows (present keys only), per ingest
    """
    from collections import defaultdict, Counter
    contrib: Dict[int, dict[str, Counter]] = {}
    typ: Dict[int, str] = {}
    target: Dict[int, int] = {}

    for i, obj in enumerate(stream_objs, start=1):
        t = obj["type"]
        typ[i] = t
        if t == "data":
            m: dict[str, Counter] = defaultdict(Counter)
            for r in obj.get("records", []):
                m[r["key"]][r["value"]] += int(r.get("weight", 1))
            contrib[i] = m
        else:
            target[i] = obj.get("target_batch_id")

    rows: List[List[str]] = []
    for i in range(1, len(stream_objs)+1):
        ws, we = max(1, i - W + 1), i
        idxs = range(ws, we + 1)

        by_key: dict[str, Counter] = defaultdict(Counter)
        # add data in window
        for j in idxs:
            if typ[j] == "data":
                for k, cnt in contrib.get(j, {}).items():
                    by_key[k].update(cnt)
        # apply retractions in window
        for j in idxs:
            if typ[j] == "retract":
                t = target.get(j)
                if t is not None and ws <= t <= we and typ.get(t) == "data":
                    for k, cnt in contrib.get(t, {}).items():
                        for v, c in cnt.items():
                            by_key[k][v] -= c
                            if by_key[k][v] <= 0:
                                del by_key[k][v]

        # GLOBAL
        g = Counter()
        for c in by_key.values(): g.update(c)
        rows.append([str(i), str(ws), str(we), "GLOBAL", ""] + _nearest_rank(g))

        # per-key rows
        for k in sorted(k for k, c in by_key.items() if sum(c.values()) > 0):
            rows.append([str(i), str(ws), str(we), "KEY", k] + _nearest_rank(by_key[k]))

    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS: df[c] = df[c].astype(str)
    return df

def _read_solution(sol_path: Path) -> Optional[pd.DataFrame]:
    """Load /workdir/sol.csv with strict columns/order; treat all cells as strings."""
    if not sol_path.exists():
        return None
    try:
        sdf = pd.read_csv(sol_path, dtype=str, keep_default_na=False)
        if list(sdf.columns) != COLS:
            return None
        for c in COLS: sdf[c] = sdf[c].astype(str)
        return sdf
    except Exception:
        return None

def grade(transcript: str | None = None) -> GradingResult:
    """
    Grade by recomputing expected rows from stream.jsonl (same logic as solution)
    and comparing to /workdir/sol.csv (schema, row count, exact cell values).
    """
    subs = {"stream_loaded": 0.0, "schema": 0.0, "row_count": 0.0, "exact_values": 0.0}
    weights = {k: 1 for k in subs}

    # Load & normalize stream with tolerance rules
    stream_objs = _load_stream(STREAM)
    if not stream_objs:
        return GradingResult(score=0.0, feedback="No usable lines in data/stream.jsonl", subscores=subs, weights=weights)
    subs["stream_loaded"] = 1.0

    # Compute expected
    exp = _expected_df(stream_objs)

    # Read contestant solution
    sol = _read_solution(SOL)
    if sol is None:
        score = sum(subs[k]*weights[k] for k in subs)
        return GradingResult(score=score, feedback="Missing or unreadable /workdir/sol.csv", subscores=subs, weights=weights)

    # Schema
    subs["schema"] = 1.0

    # Row count
    if len(sol) != len(exp):
        score = sum(subs[k]*weights[k] for k in subs)
        return GradingResult(score=score, feedback=f"Row count mismatch. Expected {len(exp)}, got {len(sol)}.", subscores=subs, weights=weights)
    subs["row_count"] = 1.0

    # Exact values (cell-by-cell)
    mismatches = []
    for i in range(len(exp)):
        e = exp.iloc[i].to_dict()
        s = sol.iloc[i].to_dict()
        if e != s:
            mismatches.append({"row": i+1, "expected": e, "got": s})
            if len(mismatches) >= 25:
                break
    if mismatches:
        score = sum(subs[k]*weights[k] for k in subs)
        return GradingResult(score=score, feedback="Values do not match expected.", subscores=subs, weights=weights, details={"diff": mismatches})

    # Success
    subs["exact_values"] = 1.0
    score = sum(subs[k]*weights[k] for k in subs)
    return GradingResult(score=score, feedback="Correct!", subscores=subs, weights=weights)

if __name__ == "__main__":
    g = grade(None)
    print(g.model_dump() if hasattr(g, "model_dump") else g.__dict__)
