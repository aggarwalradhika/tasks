# tests/grader.py
import csv
import json
import bz2
import gzip
from pathlib import Path
from typing import Dict, List, Tuple
from pydantic import BaseModel
import math
import pandas as pd
from datetime import datetime

START = datetime.fromisoformat("2023-03-26T01:00:00+00:00")
END   = datetime.fromisoformat("2023-03-26T02:30:00+00:00")

class GradingResult(BaseModel):
    """Structured result returned by the grader."""
    score: float
    feedback: str | None = None
    subscores: dict = {}
    details: dict | None = None
    weights: dict | None = None

def _iter_logs(root: Path):
    """
    Yield open file handles for each log file under data/logs.

    Supports compressed and uncompressed formats:
      - *.jsonl and *.jsonl.gz
      - *.csv and *.csv.bz2

    Returns tuples: (filename, kind, file_handle)
      kind ∈ {"jsonl", "csv"}
    """
    logs = (root / "data" / "logs")
    if not logs.exists():
        return
    for p in logs.iterdir():
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                yield p.name, "jsonl", f
        elif p.suffix == ".bz2":
            with bz2.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                yield p.name, "csv", f
        elif p.suffix == ".jsonl":
            yield p.name, "jsonl", open(p, "rt", encoding="utf-8", errors="ignore")
        elif p.suffix == ".csv":
            yield p.name, "csv", open(p, "rt", encoding="utf-8", errors="ignore")

def _parse_jsonl(f):
    """
    Parse a JSONL log file into Python dicts.

    Skips blank lines and lines starting with '//' (treated as comments).
    Malformed lines are ignored.
    """
    for line in f:
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
            yield obj
        except Exception:
            continue

def _parse_csv(f):
    """
    Parse a CSV log file into dicts.

    Strips inline '#' comments from cell values.
    Malformed rows are ignored.
    """
    try:
        reader = csv.DictReader(f)
    except Exception:
        return
    for row in reader:
        clean = {}
        for k, v in row.items():
            if isinstance(v, str):
                v = v.split("#")[0].strip()
            clean[k] = v
        yield clean

def _within_window(ts_iso: str) -> bool:
    """
    Return True if timestamp (ISO string) falls within the
    inclusive UTC window [START, END].
    """
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except Exception:
        return False
    return (START <= dt <= END)

def _to_record(d: dict) -> dict | None:
    """
    Normalize a raw log dict into a structured record.

    Expected fields: ts, request_id, user_id, path, status, latency_ms.
    Records are dropped if:
      - path != '/api/v2/order'
      - timestamp outside the UTC window
      - required fields missing/invalid
    """
    try:
        ts = d["ts"]
        rid = d["request_id"]
        uid = d["user_id"]
        path = d["path"]
        status = int(d["status"])
        lat = int(d["latency_ms"])
    except Exception:
        return None
    if path != "/api/v2/order":
        return None
    if not _within_window(ts):
        return None
    return {"ts": ts, "request_id": rid, "user_id": uid, "status": status, "latency_ms": lat}

def _expected_df(workdir: Path) -> pd.DataFrame:
    """
    Compute the expected DataFrame solution from ground-truth logs.

    - Deduplicates requests by request_id across all files
    - Filters by path and time window
    - Groups by user_id
    - Computes:
        * p95_ms (95th percentile, half-up rounding)
        * error_rate (fraction of status>=500, rounded 4 dp half-up)
    """
    seen = set()
    rows = []
    for name, kind, f in _iter_logs(workdir) or []:
        with f:
            it = _parse_jsonl(f) if kind == "jsonl" else _parse_csv(f)
            for obj in it:
                rec = _to_record(obj)
                if not rec:
                    continue
                if rec["request_id"] in seen:
                    continue
                seen.add(rec["request_id"])
                rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["user_id","p95_ms","error_rate"])

    df = pd.DataFrame(rows)

    out = []
    for uid, g in df.groupby("user_id"):
        lat = g["latency_ms"].to_numpy()
        p = float(pd.Series(lat).quantile(0.95, interpolation="linear"))
        p95 = int(math.floor(p + 0.5))
        total = len(g)
        errs = int((g["status"] >= 500).sum())
        er = errs / total
        er_rounded = f"{(math.floor(er*10000 + 0.5)/10000):.4f}"
        out.append({"user_id": uid, "p95_ms": p95, "error_rate": er_rounded})
    out_df = pd.DataFrame(out).sort_values("user_id").reset_index(drop=True)
    out_df["user_id"] = out_df["user_id"].astype(str)
    out_df["p95_ms"] = out_df["p95_ms"].astype(int)
    out_df["error_rate"] = out_df["error_rate"].astype(str)
    return out_df

def _read_solution(path: Path) -> pd.DataFrame | None:
    """
    Load the contestant's /workdir/sol.csv into a normalized DataFrame.

    Ensures:
      - Columns = user_id,p95_ms,error_rate
      - error_rate always 4 decimals
      - Sorted by user_id
    """
    if not path.exists():
        return None
    try:
        sdf = pd.read_csv(path, dtype={"user_id": str, "p95_ms": int, "error_rate": str})
        sdf["error_rate"] = sdf["error_rate"].astype(float).map(lambda x: f"{x:.4f}")
        sdf = sdf.sort_values("user_id").reset_index(drop=True)
        return sdf[["user_id","p95_ms","error_rate"]]
    except Exception:
        return None

def grade(transcript: str | None = None) -> GradingResult:
    """
    Main grading entrypoint.

    Compares contestant solution (sol.csv) with expected output.
    Returns GradingResult with overall score, subscores, and feedback.
    """
    workdir = Path("/workdir")
    exp = _expected_df(workdir)
    sol = _read_solution(workdir / "sol.csv")

    subs = {
        "exact_values": 0.0,
    }
    weights = {"exact_values": 1}

    if sol is None:
        feedback = "Missing or unreadable /workdir/sol.csv"
        return GradingResult(score=0.0, feedback=feedback, subscores=subs, weights=weights)

    if list(sol.columns) != ["user_id","p95_ms","error_rate"]:
        feedback = "Incorrect columns. Expected 'user_id,p95_ms,error_rate'."
        score = sum(subs[k]*weights[k] for k in subs)
        return GradingResult(score=score, feedback=feedback, subscores=subs, weights=weights)

    if len(sol) != len(exp):
        feedback = f"Row count mismatch. Expected {len(exp)}, got {len(sol)}."
        score = sum(subs[k]*weights[k] for k in subs)
        return GradingResult(score=score, feedback=feedback, subscores=subs, weights=weights)

    mismatches = []
    for i in range(len(exp)):
        e = exp.iloc[i].to_dict()
        s = sol.iloc[i].to_dict()
        if e != s:
            mismatches.append({"row": i, "expected": e, "got": s})

    if mismatches:
        feedback = "Values do not match expected."
        score = sum(subs[k]*weights[k] for k in subs)
        return GradingResult(score=score, feedback=feedback, subscores=subs, weights=weights, details={"diff": mismatches})

    subs["exact_values"] = 1.0
    score = sum(subs[k]*weights[k] for k in subs)
    return GradingResult(score=score, feedback="Correct!", subscores=subs, weights=weights)
