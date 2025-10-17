#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/workdir/out"
mkdir -p "$OUT_DIR"

python3 - << 'PY'
from __future__ import annotations
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

# Apache Combined Log Format with extra field:
#   %h %l %u [%t] "%r" %>s %b "%{Referer}i" "%{User-Agent}i" "%{extra}i"
# Timestamp format: %d/%b/%Y:%H:%M:%S %z (e.g., 22/Jan/2019:03:56:14 +0330)

root = Path(".")

# -------------------- Load policy configuration --------------------
def iso_to_utc(s: str) -> datetime:
    """Convert ISO-8601 string to UTC datetime"""
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


policy = json.loads((root / "data/policy.json").read_text(encoding="utf-8"))
timeout = timedelta(minutes=int(policy.get("timeout_minutes", 30)))

# Parse maintenance windows and convert to UTC
windows = [(iso_to_utc(w["start"]), iso_to_utc(w["end"])) 
           for w in policy.get("maintenance_windows", [])]


def in_maintenance_window(ts: datetime) -> bool:
    """Check if timestamp falls within any maintenance window [start, end)"""
    return any(start <= ts < end for start, end in windows)


# -------------------- Parse Apache Combined Log Format --------------------
# Regex pattern for: ip ident user [ts] "req" status bytes "ref" "ua" "extra"
LOG_RE = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<req>[^"]*)"\s+\d{3}\s+\S+\s+"[^"]*"\s+"(?P<ua>[^"]*)"\s+"[^"]*"'
)


def parse_apache_timestamp(ts: str) -> datetime:
    """
    Parse Apache timestamp format to UTC datetime.
    Format: %d/%b/%Y:%H:%M:%S %z (e.g., 22/Jan/2019:03:56:14 +0330)
    Supports optional fractional seconds: 22/Jan/2019:03:56:14.123456 +0330
    """
    parts = ts.rsplit(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid timestamp format: {ts}")
    
    dt_part, tz_offset = parts
    
    # Parse datetime with or without fractional seconds
    if "." in dt_part:
        dt = datetime.strptime(dt_part, "%d/%b/%Y:%H:%M:%S.%f")
    else:
        dt = datetime.strptime(dt_part, "%d/%b/%Y:%H:%M:%S")
    
    # Parse timezone offset (+HHMM or -HHMM)
    if not re.fullmatch(r"[+-]\d{4}", tz_offset):
        raise ValueError(f"Invalid timezone offset: {tz_offset}")
    
    sign = 1 if tz_offset[0] == "+" else -1
    hours = int(tz_offset[1:3])
    minutes = int(tz_offset[3:5])
    tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    
    # Attach timezone and convert to UTC
    dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


def extract_request_path(request_line: str) -> str:
    """
    Extract path from HTTP request line.
    Request format: METHOD PATH HTTP/VERSION
    Returns None if request line is malformed.
    """
    tokens = request_line.split()
    if len(tokens) >= 3:
        return tokens[1]  # PATH is the second token
    return None


@dataclass
class Hit:
    """Represents a single parsed log entry"""
    ts: datetime
    ip: str
    ua: str
    idx: int  # Original line index for stable sorting


# -------------------- Parse log file --------------------
hits: List[Hit] = []

with (root / "data/access.log").open("r", encoding="utf-8", errors="replace") as f:
    for line_idx, line in enumerate(f):
        line = line.rstrip("\n")
        if not line:
            continue
        
        # Try to match log format
        match = LOG_RE.match(line)
        if not match:
            continue  # Drop lines that don't match format
        
        ip = match.group("ip")
        ua = match.group("ua")
        request = match.group("req")
        
        # Extract and validate path from request
        path = extract_request_path(request)
        if path is None:
            continue  # Drop malformed request lines
        
        # Parse timestamp
        try:
            ts = parse_apache_timestamp(match.group("ts"))
        except Exception:
            continue  # Drop lines with unparseable timestamps
        
        # Filter out hits in maintenance windows
        if in_maintenance_window(ts):
            continue
        
        hits.append(Hit(ts, ip, ua, line_idx))

# Sort hits by UTC timestamp ASC, then by original line index ASC
hits.sort(key=lambda h: (h.ts, h.idx))


# -------------------- Sessionize by (IP, User-Agent) --------------------
@dataclass
class Session:
    """Represents a user session"""
    ip: str
    ua: str
    start: datetime
    end: datetime
    hits: int


# Track sessions per (ip, ua) key
sessions_by_key: Dict[tuple, List[Session]] = defaultdict(list)
last_session_by_key: Dict[tuple, Session] = {}

for hit in hits:
    key = (hit.ip, hit.ua)
    
    # Check if this is first hit for this key or if timeout exceeded
    if key not in last_session_by_key:
        # Start new session
        session = Session(hit.ip, hit.ua, hit.ts, hit.ts, 1)
        sessions_by_key[key].append(session)
        last_session_by_key[key] = session
    else:
        last_session = last_session_by_key[key]
        
        # Check if gap to previous hit >= timeout (starts new session)
        if (hit.ts - last_session.end) >= timeout:
            # Start new session
            session = Session(hit.ip, hit.ua, hit.ts, hit.ts, 1)
            sessions_by_key[key].append(session)
            last_session_by_key[key] = session
        else:
            # Extend existing session
            last_session.end = hit.ts
            last_session.hits += 1

# Collect all sessions
all_sessions: List[Session] = []
for session_list in sessions_by_key.values():
    all_sessions.extend(session_list)

# Sort sessions: by start time ASC, then lexicographically by (ip, ua)
all_sessions.sort(key=lambda s: (s.start, s.ip, s.ua))


# -------------------- Format output --------------------
def format_timestamp(dt: datetime) -> str:
    """
    Format datetime to ISO-8601 UTC with 'Z'.
    Include fractional seconds only when non-zero, with trailing zeros stripped.
    """
    dt = dt.astimezone(timezone.utc)
    
    if dt.microsecond:
        # Format with fractional seconds
        iso_str = dt.isoformat().replace("+00:00", "Z")
        
        # Strip trailing zeros from fractional part
        if '.' in iso_str:
            before_z = iso_str.rstrip('Z')
            if '.' in before_z:
                integer_part, frac_part = before_z.rsplit('.', 1)
                frac_part = frac_part.rstrip('0')
                if frac_part:
                    iso_str = f"{integer_part}.{frac_part}Z"
                else:
                    iso_str = f"{integer_part}Z"
        
        return iso_str
    else:
        # No fractional seconds
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# Write output CSV
output_path = root / "out/sessions.csv"
with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["session_id", "ip", "ua", "start", "end", "hits"])
    
    for idx, session in enumerate(all_sessions, start=1):
        session_id = f"s{idx:06d}"
        writer.writerow([
            session_id,
            session.ip,
            session.ua,
            format_timestamp(session.start),
            format_timestamp(session.end),
            session.hits
        ])

PY

echo "Successfully wrote /workdir/out/sessions.csv"