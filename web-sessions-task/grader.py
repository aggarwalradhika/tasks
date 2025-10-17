from __future__ import annotations
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from apex_arena._types import GradingResult

CANDIDATE = Path("/workdir/sessions.csv")
ACCESS_LOG = Path("/workdir/data/access.log")
POLICY_JSON = Path("/workdir/data/policy.json")


def _csv_rows_from_file(path: Path) -> List[List[str]]:
    """Parse CSV into rows of fields; ignore completely blank rows"""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader if not all((c.strip() == "" for c in row))]


def iso_to_utc(s: str) -> datetime:
    """Convert ISO-8601 string to UTC datetime"""
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_apache_timestamp(ts: str) -> datetime:
    """Parse Apache timestamp format to UTC datetime"""
    parts = ts.rsplit(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid timestamp: {ts}")
    
    dt_part, tz_offset = parts
    
    if "." in dt_part:
        dt = datetime.strptime(dt_part, "%d/%b/%Y:%H:%M:%S.%f")
    else:
        dt = datetime.strptime(dt_part, "%d/%b/%Y:%H:%M:%S")
    
    if not re.fullmatch(r"[+-]\d{4}", tz_offset):
        raise ValueError(f"Invalid timezone offset: {tz_offset}")
    
    sign = 1 if tz_offset[0] == "+" else -1
    hours = int(tz_offset[1:3])
    minutes = int(tz_offset[3:5])
    tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    
    dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


def extract_request_path(request_line: str) -> str:
    """Extract path from HTTP request line"""
    tokens = request_line.split()
    if len(tokens) >= 3:
        return tokens[1]
    return None


def in_maintenance_window(ts: datetime, windows: List[Tuple[datetime, datetime]]) -> bool:
    """Check if timestamp falls in any maintenance window [start, end)"""
    return any(start <= ts < end for start, end in windows)


def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO-8601 UTC with 'Z', strip trailing zeros from fractional seconds"""
    dt = dt.astimezone(timezone.utc)
    
    if dt.microsecond:
        iso_str = dt.isoformat().replace("+00:00", "Z")
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
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Hit:
    """Represents a single parsed log entry"""
    ts: datetime
    ip: str
    ua: str
    idx: int


@dataclass
class Session:
    """Represents a user session"""
    ip: str
    ua: str
    start: datetime
    end: datetime
    hits: int


def generate_expected_sessions() -> List[Session]:
    """
    Generate expected sessions by independently processing the access log.
    This implements the complete sessionization algorithm as specified in task.yaml:
    
    1. Parse Apache Combined Log Format with extra field
    2. Extract and convert timestamps to UTC (preserve fractional seconds)
    3. Filter hits in maintenance windows [start, end)
    4. Sort hits by UTC timestamp ASC, then line index ASC
    5. Sessionize by (IP, UA) with timeout_minutes threshold
    6. Sort sessions by start time ASC, tie-break by (ip, ua) lexicographically
    """
    # Load policy
    policy = json.loads(POLICY_JSON.read_text(encoding="utf-8"))
    timeout = timedelta(minutes=int(policy.get("timeout_minutes", 30)))
    
    # Parse maintenance windows
    windows = [(iso_to_utc(w["start"]), iso_to_utc(w["end"])) 
               for w in policy.get("maintenance_windows", [])]
    
    # Parse log file - Apache Combined Log Format with extra field
    LOG_RE = re.compile(
        r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
        r'"(?P<req>[^"]*)"\s+\d{3}\s+\S+\s+"[^"]*"\s+"(?P<ua>[^"]*)"\s+"[^"]*"'
    )
    
    hits: List[Hit] = []
    
    with ACCESS_LOG.open("r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            
            # Match log format
            match = LOG_RE.match(line)
            if not match:
                continue  # Drop lines that don't match
            
            ip = match.group("ip")
            ua = match.group("ua")
            request = match.group("req")
            
            # Validate request has 3 tokens (METHOD PATH VERSION)
            path = extract_request_path(request)
            if path is None:
                continue  # Drop malformed requests
            
            # Parse and convert timestamp to UTC
            try:
                ts = parse_apache_timestamp(match.group("ts"))
            except Exception:
                continue  # Drop unparseable timestamps
            
            # Filter maintenance windows [start, end)
            if in_maintenance_window(ts, windows):
                continue
            
            hits.append(Hit(ts, ip, ua, line_idx))
    
    # Sort hits by UTC timestamp ASC, then original line index ASC
    hits.sort(key=lambda h: (h.ts, h.idx))
    
    # Sessionize by (IP, User-Agent)
    sessions_by_key: Dict[tuple, List[Session]] = defaultdict(list)
    last_session_by_key: Dict[tuple, Session] = {}
    
    for hit in hits:
        key = (hit.ip, hit.ua)
        
        if key not in last_session_by_key:
            # First hit for this key - start new session
            session = Session(hit.ip, hit.ua, hit.ts, hit.ts, 1)
            sessions_by_key[key].append(session)
            last_session_by_key[key] = session
        else:
            last_session = last_session_by_key[key]
            
            # Check timeout: gap >= timeout_minutes starts new session
            if (hit.ts - last_session.end) >= timeout:
                session = Session(hit.ip, hit.ua, hit.ts, hit.ts, 1)
                sessions_by_key[key].append(session)
                last_session_by_key[key] = session
            else:
                # Extend current session
                last_session.end = hit.ts
                last_session.hits += 1
    
    # Collect all sessions
    all_sessions: List[Session] = []
    for session_list in sessions_by_key.values():
        all_sessions.extend(session_list)
    
    # Sort sessions by start time ASC, tie-break by (ip, ua) lexicographically
    all_sessions.sort(key=lambda s: (s.start, s.ip, s.ua))
    
    return all_sessions


def sessions_to_csv_rows(sessions: List[Session]) -> List[List[str]]:
    """Convert sessions to CSV rows with proper formatting per task.yaml"""
    rows = [["session_id", "ip", "ua", "start", "end", "hits"]]
    
    for idx, session in enumerate(sessions, start=1):
        session_id = f"s{idx:06d}"  # s000001, s000002, etc.
        rows.append([
            session_id,
            session.ip,
            session.ua,
            format_timestamp(session.start),
            format_timestamp(session.end),
            str(session.hits)
        ])
    
    return rows


def grade(transcript: str) -> GradingResult:
    """
    Grade the candidate solution by:
    1. Independently processing the access log to generate expected sessions
    2. Comparing candidate output exactly against expected output
    
    This validates ALL requirements from task.yaml:
    - Apache Combined Log Format parsing (with extra field)
    - Timestamp conversion to UTC with fractional second preservation
    - Maintenance window filtering [start, end) closed-open intervals
    - Hit sorting by UTC timestamp then line index
    - Strict line dropping (invalid format, unparseable timestamps, malformed requests)
    - Sessionization by exact (IP, UA) key
    - Timeout threshold handling (>= timeout starts new session)
    - Session ordering by start time with (ip, ua) tie-breaking
    - Sequential session IDs (s000001, s000002, ...)
    - CSV format with exact header
    - Timestamp formatting with 'Z' suffix and trailing zero stripping
    - Final newline at EOF
    """
    subscores = {"all_passes": 0.0}
    weights = {"all_passes": 1.0}
    
    # Check if candidate output exists
    if not CANDIDATE.exists():
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback=(
                "FAIL: Output file /workdir/sessions.csv not found.\n"
                "Ensure your solution writes to this exact path."
            ),
        )
    
    # Check if input files exist
    if not ACCESS_LOG.exists():
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback="FAIL: Input file /workdir/data/access.log not found (system error).",
        )
    
    if not POLICY_JSON.exists():
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback="FAIL: Input file /workdir/data/policy.json not found (system error).",
        )
    
    # Parse candidate output
    try:
        candidate_rows = _csv_rows_from_file(CANDIDATE)
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback=(
                f"FAIL: Could not parse candidate CSV: {e}\n"
                "Ensure output is valid CSV format with proper escaping."
            ),
        )
    
    # Validate basic CSV structure
    if len(candidate_rows) == 0:
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback="FAIL: CSV file is empty. Must contain at least a header row.",
        )
    
    expected_header = ["session_id", "ip", "ua", "start", "end", "hits"]
    if candidate_rows[0] != expected_header:
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback=(
                f"FAIL: Invalid CSV header.\n"
                f"Expected: {expected_header}\n"
                f"Got: {candidate_rows[0]}"
            ),
        )
    
    # Generate expected sessions by independently processing the log
    try:
        expected_sessions = generate_expected_sessions()
        expected_rows = sessions_to_csv_rows(expected_sessions)
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback=f"FAIL: Error generating expected output (system error): {e}",
        )
    
    # Compare row counts
    if len(candidate_rows) != len(expected_rows):
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback=(
                f"FAIL: Incorrect number of sessions.\n"
                f"Expected: {len(expected_rows) - 1} sessions\n"
                f"Got: {len(candidate_rows) - 1} sessions\n\n"
                f"Common causes:\n"
                f"- Incorrect log parsing (not matching Apache Combined Log Format)\n"
                f"- Wrong maintenance window filtering (should be [start, end) closed-open)\n"
                f"- Incorrect sessionization timeout logic (>= timeout starts new session)\n"
                f"- Not dropping invalid lines (malformed format, bad timestamps, bad requests)"
            ),
        )
    
    # If only headers (no sessions), this is valid if all logs were filtered
    if len(expected_rows) == 1:
        subscores["all_passes"] = 1.0
        return GradingResult(
            score=1.0,
            subscores=subscores,
            weights=weights,
            feedback="PASS: No sessions generated (valid if all logs were filtered out).",
        )
    
    # Compare each row exactly
    for i in range(1, len(expected_rows)):
        exp_row = expected_rows[i]
        cand_row = candidate_rows[i]
        
        if cand_row != exp_row:
            # Build detailed error message
            differences = []
            fields = ["session_id", "ip", "ua", "start", "end", "hits"]
            for j, field in enumerate(fields):
                if exp_row[j] != cand_row[j]:
                    differences.append(f"  {field}: expected '{exp_row[j]}', got '{cand_row[j]}'")
            
            return GradingResult(
                score=0.0,
                subscores=subscores,
                weights=weights,
                feedback=(
                    f"FAIL: Mismatch at row {i+1} (session {exp_row[0]}):\n"
                    + "\n".join(differences) + "\n\n"
                    f"Common causes:\n"
                    f"- Incorrect sessionization logic (timeout boundary handling)\n"
                    f"- Wrong timestamp conversion or timezone handling\n"
                    f"- Incorrect session ordering (should sort by start, then (ip, ua))\n"
                    f"- Wrong timestamp formatting (use 'Z', strip trailing zeros)\n"
                    f"- Incorrect hit counting within sessions\n"
                    f"- Wrong session ID format (should be s000001, s000002, ...)"
                ),
            )
    
    # All rows match - full score
    subscores["all_passes"] = 1.0
    return GradingResult(
        score=1.0,
        subscores=subscores,
        weights=weights,
        feedback=(
            "PASS: All validations passed.\n"
            "✓ Apache Combined Log Format parsing correct\n"
            "✓ Timestamp conversion to UTC correct\n"
            "✓ Maintenance window filtering correct\n"
            "✓ Hit sorting correct\n"
            "✓ Sessionization logic correct\n"
            "✓ Session ordering correct\n"
            "✓ CSV formatting correct\n"
            "✓ All sessions match expected output"
        ),
    )