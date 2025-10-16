#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/workdir/out"
mkdir -p "$OUT_DIR"

python3 - << 'PY'
from __future__ import annotations
import csv, json, re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Dict

# --- Sources / format notes ---
# Apache Combined Log Format (CLF + referer + UA):
#   %h %l %u [%t] "%r" %>s %b "%{Referer}i" "%{User-Agent}i"
# Timestamp [%t] uses: %d/%b/%Y:%H:%M:%S %z  e.g. [22/Jan/2019:03:56:14 +0330]
# References: Wikipedia CLF/Combined; Loggly guide; common regex patterns. 
# (Used here purely to justify format & parsing approach.)
# -------------------------------------------------------------------------

root = Path("./")

# -------------------- Policy: load, normalize to UTC, merge overlaps --------------------
def iso_to_utc(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

policy = json.loads((root/"data/policy.json").read_text(encoding="utf-8"))
timeout = timedelta(minutes=int(policy.get("timeout_minutes", 30)))

windows_raw = [(iso_to_utc(w["start"]), iso_to_utc(w["end"])) for w in policy.get("maintenance_windows", [])]
# merge overlapping [start,end) windows in UTC for determinism
windows: List[Tuple[datetime, datetime]] = []
for s,e in sorted(windows_raw, key=lambda x: x[0]):
    if not windows:
        windows.append([s,e])
    else:
        ps,pe = windows[-1]
        if s <= pe:  # overlap/adjacent: keep closed-open semantics exactly
            if e > pe:
                windows[-1][1] = e
        else:
            windows.append([s,e])
windows = [(s,e) for s,e in windows]

def in_windows(ts: datetime) -> bool:
    return any(s <= ts < e for s,e in windows)

# -------------------- Apache Combined parser --------------------
# Pattern: ip ident user [ts] "req" status bytes "ref" "ua" "extra"
# The actual log format has an extra field at the end
LOG_RE = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+"(?P<req>[^"]*)"\s+\d{3}\s+\S+\s+"[^"]*"\s+"(?P<ua>[^"]*)"\s+"[^"]*"'
)
# ts like: 22/Jan/2019:03:56:14 +0330  (optionally with .ffffff before space)
def parse_ts_apache(ts: str) -> datetime:
    # split "date time+offset"
    # allow optional fractional seconds: 22/Jan/2019:03:56:14.123456 +0330
    # Normalize to a python parseable form
    # First, separate the offset (last space +HHMM or -HHMM)
    parts = ts.rsplit(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad ts: {ts}")
    dt_part, off = parts
    # Prepare format based on fractional presence
    if "." in dt_part:
        dt = datetime.strptime(dt_part, "%d/%b/%Y:%H:%M:%S.%f")
    else:
        dt = datetime.strptime(dt_part, "%d/%b/%Y:%H:%M:%S")
    # attach numeric offset like +0330
    if not re.fullmatch(r"[+-]\d{4}", off):
        raise ValueError(f"Bad offset: {off}")
    # Convert +HHMM to hours/minutes
    sign = 1 if off[0] == "+" else -1
    hh = int(off[1:3]); mm = int(off[3:5])
    tz = timezone(sign * timedelta(hours=hh, minutes=mm))
    dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)

def extract_path(req: str) -> str:
    # req typically: METHOD SP PATH SP HTTP/version
    # e.g., GET /product/10214 HTTP/1.1
    # Must have exactly 3 parts to be valid
    toks = req.split()
    if len(toks) >= 3:
        return toks[1]
    else:
        return None  # Invalid request line

@dataclass
class Hit:
    ts: datetime
    ip: str
    ua: str
    path: str
    idx: int  # original line index for stable ordering

hits: List[Hit] = []
with (root/"data/access.log").open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        line = line.rstrip("\n")
        if not line:
            continue
        m = LOG_RE.match(line)
        if not m:
            # Strict task says: drop lines that fail to parse
            continue
        ip = m.group("ip")
        ua = m.group("ua")
        req = m.group("req")
        path = extract_path(req)
        if path is None:
            # drop malformed request lines
            continue
        try:
            ts = parse_ts_apache(m.group("ts"))
        except Exception:
            # drop unparseable timestamps
            continue
        if in_windows(ts):
            continue
        hits.append(Hit(ts, ip, ua, path, i))

# Ordering: ts_utc ASC, then original line index ASC
hits.sort(key=lambda h: (h.ts, h.idx))

# -------------------- Sessionization per (ip, ua) --------------------
from collections import defaultdict
@dataclass
class Session:
    ip: str
    ua: str
    start: datetime
    end: datetime
    hits: int
    first_path: str

key_sessions: Dict[tuple, List[Session]] = defaultdict(list)
last_by_key: Dict[tuple, Session] = {}

for h in hits:
    k = (h.ip, h.ua)
    if k not in last_by_key:
        s = Session(h.ip, h.ua, h.ts, h.ts, 1, h.path)
        key_sessions[k].append(s)
        last_by_key[k] = s
        continue
    last = last_by_key[k]
    # exact boundary: >= timeout starts new session
    if (h.ts - last.end) >= timeout:
        s = Session(h.ip, h.ua, h.ts, h.ts, 1, h.path)
        key_sessions[k].append(s)
        last_by_key[k] = s
    else:
        last.end = h.ts
        last.hits += 1

sessions: List[Session] = [s for group in key_sessions.values() for s in group]

# Output order & tie-breakers:
# Sort by start ASC, then lexicographically by (ip, ua)
sessions.sort(key=lambda s: (s.start, s.ip, s.ua))

def fmt(dt: datetime) -> str:
    # Emit 'Z' and include fractional seconds only when non-zero
    dt = dt.astimezone(timezone.utc)
    if dt.microsecond:
        # Format with fractional seconds, strip trailing zeros
        s = dt.isoformat().replace("+00:00", "Z")
        # Remove trailing zeros from fractional seconds but keep at least 1 digit
        if '.' in s:
            before_z, z = s.rsplit('Z', 1)
            if '.' in before_z:
                integer_part, frac_part = before_z.rsplit('.', 1)
                # Strip trailing zeros but keep at least one decimal place for clarity
                frac_part = frac_part.rstrip('0') or '0'
                s = f"{integer_part}.{frac_part}Z"
        return s
    else:
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

out = root/"tests/expected_sessions.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["session_id","ip","ua","start","end","hits"])
    for i, s in enumerate(sessions, 1):
        sid = f"s{i:06d}"
        w.writerow([sid, s.ip, s.ua, fmt(s.start), fmt(s.end), s.hits])
PY

echo "Wrote /workdir/tests/expected_sessions.csv"