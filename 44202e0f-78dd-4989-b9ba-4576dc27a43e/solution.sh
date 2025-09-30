#!/usr/bin/env bash
set -euo pipefail

# Parse mixed compressed logs, dedupe by request_id, filter UTC window, compute metrics, write /workdir/sol.csv
python3 - << 'PY'
import csv, json, gzip, bz2, math
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

START = datetime.fromisoformat("2023-03-26T01:00:00+00:00")
END   = datetime.fromisoformat("2023-03-26T02:30:00+00:00")

def iter_logs(root: Path):
    for p in root.iterdir():
        if p.suffix == ".gz":
            f = gzip.open(p, "rt", encoding="utf-8", errors="ignore")
            kind = "jsonl"
        elif p.suffix == ".bz2":
            f = bz2.open(p, "rt", encoding="utf-8", errors="ignore")
            kind = "csv"
        elif p.suffix == ".jsonl":
            f = open(p, "rt", encoding="utf-8", errors="ignore")
            kind = "jsonl"
        elif p.suffix == ".csv":
            f = open(p, "rt", encoding="utf-8", errors="ignore")
            kind = "csv"
        else:
            continue
        yield p, kind, f

def parse_jsonl(f):
    for line in f:
        line=line.strip()
        if not line or line.startswith("//"): continue
        try:
            yield json.loads(line)
        except: pass

def parse_csv(f):
    try:
        reader = csv.DictReader(f)
    except Exception:
        return
    for row in reader:
        clean = {k:(v.split("#")[0].strip() if isinstance(v,str) else v) for k,v in row.items()}
        yield clean

def within(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
    except Exception:
        return False
    return START <= dt <= END

root = Path("/workdir/data/logs")
seen = set()
rows = []
for p, kind, f in iter_logs(root):
    with f:
        it = parse_jsonl(f) if kind=="jsonl" else parse_csv(f)
        for obj in it:
            try:
                ts = obj["ts"]; rid = obj["request_id"]; uid = obj["user_id"]
                path = obj["path"]; status = int(obj["status"]); lat = int(obj["latency_ms"])
            except Exception:
                continue
            if path != "/api/v2/order": continue
            if not within(ts): continue
            if rid in seen: continue
            seen.add(rid)
            rows.append({"user_id": uid, "status": status, "latency_ms": lat})

if not rows:
    with open("/workdir/sol.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["user_id","p95_ms","error_rate"])
    raise SystemExit(0)

import pandas as pd, math
df = pd.DataFrame(rows)
out = []
for uid, g in df.groupby("user_id"):
    lat = g["latency_ms"].to_numpy()
    p = float(pd.Series(lat).quantile(0.95, interpolation="linear"))
    p95 = int(math.floor(p + 0.5))
    total = len(g); errs = int((g["status"]>=500).sum())
    er = errs/total
    er_rounded = f"{(math.floor(er*10000 + 0.5)/10000):.4f}"
    out.append({"user_id": uid, "p95_ms": p95, "error_rate": er_rounded})

out = pd.DataFrame(out).sort_values("user_id")
out.to_csv("/workdir/sol.csv", index=False, columns=["user_id","p95_ms","error_rate"])
PY
