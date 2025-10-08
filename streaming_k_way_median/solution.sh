#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PY'
"""
Reference solution (tolerant, matches grader exactly):

- Accepts both "object lines" and "array lines" as defined in task.yaml.
- Auto-assigns batch_id when missing.
- Coerces invalid weights (<=0 or non-int) to 1.
- Ignores blank lines, '//' comment lines, and malformed JSON lines.
- Sliding window W=6; weighted nearest-rank percentiles [10,50,90].
- Writes /workdir/sol.csv with strict schema and ordering.
"""
import json, math, csv
from pathlib import Path
from collections import defaultdict, Counter

# constants 
PCTS = [10, 50, 90]
W = 6

STREAM = Path("/workdir/data/stream.jsonl")
OUT = Path("/workdir/sol.csv")

def normalize_line(obj, next_batch_id: int):
    """Normalize a parsed JSON into canonical object form or return None to skip."""
    # Array line → treat as data batch for key "A"
    if isinstance(obj, list):
        recs = [{"key":"A","value":int(v),"weight":1} for v in obj]
        return {"batch_id": next_batch_id, "type":"data", "records": recs}
    # Object line → ensure fields, coerce weights
    if isinstance(obj, dict):
        t = obj.get("type")
        if t in ("data","retract"):
            if not isinstance(obj.get("batch_id"), int):
                obj = {**obj, "batch_id": next_batch_id}
            if t == "data":
                fixed=[]
                for r in obj.get("records", []) or []:
                    if not isinstance(r, dict):
                        continue
                    k=r.get("key"); v=r.get("value"); w=r.get("weight",1)
                    if not isinstance(k,str) or not isinstance(v,int):
                        continue
                    try: w=int(w)
                    except: w=1
                    if w<=0: w=1
                    fixed.append({"key":k,"value":int(v),"weight":int(w)})
                obj = {**obj, "records": fixed}
            return obj
    return None

# Load & normalize stream
objs=[]
next_bid=1
if STREAM.exists():
    for raw in STREAM.read_text().splitlines():
        s=raw.strip()
        if not s or s.startswith("//"):  # ignore blanks and comment lines
            continue
        try:
            parsed = json.loads(s)
        except Exception:
            continue
        v = normalize_line(parsed, next_bid)
        if v is None:
            continue
        objs.append(v)
        next_bid = max(next_bid, v.get("batch_id", next_bid)) + 1

# Pre-index contributions per ingest
contrib, typ, target = {}, {}, {}
for i, obj in enumerate(objs, start=1):
    t=obj["type"]; typ[i]=t
    if t=="data":
        m=defaultdict(Counter)
        for r in obj.get("records", []):
            m[r["key"]][r["value"]] += int(r.get("weight",1))
        contrib[i]=m
    else:
        target[i]=obj.get("target_batch_id")

def pvals(counter):
    """Weighted nearest-rank percentiles over a Counter[int->int]."""
    N=sum(counter.values())
    if N==0: return [""]*len(PCTS)
    items=sorted(counter.items())
    out=[]
    for p in PCTS:
        r=math.ceil(p/100 * N)
        cum=0
        for v,c in items:
            cum+=c
            if cum>=r:
                out.append(str(v)); break
    return out

# Build rows per ingest
rows=[]
for i in range(1, len(objs)+1):
    ws, we = max(1, i-W+1), i
    idxs = range(ws, we+1)
    by_key=defaultdict(Counter)

    # add data in window
    for j in idxs:
        if typ[j]=="data":
            for k,cnt in contrib.get(j, {}).items():
                by_key[k].update(cnt)
    # apply retractions (only when target is inside window and is a data batch)
    for j in idxs:
        if typ[j]=="retract":
            t=target.get(j)
            if t is not None and ws<=t<=we and typ.get(t)=="data":
                for k,cnt in contrib.get(t, {}).items():
                    for v,c in cnt.items():
                        by_key[k][v]-=c
                        if by_key[k][v]<=0: del by_key[k][v]

    # GLOBAL row
    g=Counter()
    for c in by_key.values(): g.update(c)
    rows.append([str(i), str(ws), str(we), "GLOBAL", ""] + pvals(g))
    # per-key rows
    for k in sorted(k for k,c in by_key.items() if sum(c.values())>0):
        rows.append([str(i), str(ws), str(we), "KEY", k] + pvals(by_key[k]))

# Write CSV
with OUT.open("w", newline="") as f:
    w=csv.writer(f)
    w.writerow(["ingest_index","window_start_ingest","window_end_ingest","scope","key"] + [f"p{p}" for p in PCTS])
    w.writerows(rows)
PY
