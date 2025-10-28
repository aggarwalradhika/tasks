#!/usr/bin/env python3
"""
Advanced grader for distributed event log reconciliation task.
Recomputes expected output with all features and validates against submission.

Fixed issues:
1. Added reliability-weighted conflict resolution (conflict_threshold=0.3)
2. Implemented complete unanimous strategy (checks ALL streams, not just 2-way)
3. Implemented complete majority strategy (vote counting across N streams)
4. Added reliability-based conflict resolution at event entry level
5. Clarified snapshot restore behavior (no row generation)
"""
import json, csv, os, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any, Set
from copy import deepcopy

# Pydantic shim
try:
    from pydantic import BaseModel
except:
    class BaseModel:
        def __init__(self, **kw): self.__dict__.update(kw)
        def model_dump(self): return self.__dict__

class GradingResult(BaseModel):
    score: float
    feedback: str | None = None
    subscores: dict = {}
    details: dict | None = None
    weights: dict | None = None

# Constants (must match task specification)
CONSISTENCY_MODES = ["EVENTUAL", "CAUSAL", "LINEARIZABLE"]
CAUSALITY_WINDOW_SIZE = 5
CONFLICT_THRESHOLD = 0.3
MIN_RELIABILITY = 0.1

WORKDIR = Path("/workdir")
EVENTS = WORKDIR / "data" / "events.jsonl"
SOL = WORKDIR / "sol.csv"

COLS = ["event_index", "window_start", "window_end", "stream", "consistency", 
        "total_entries", "unique_ops", "seq_gaps", "conflicts", "causal_violations", 
        "avg_clock_delta", "flags"]

class EventState:
    """Encapsulates complete event processing state for snapshotting"""
    def __init__(self):
        self.streams: Dict[str, Dict[int, Dict]] = defaultdict(lambda: {})  # stream -> seq -> entry_data
        self.stream_aliases: Dict[str, str] = {}  # secondary -> primary mapping
        self.seq_offsets: Dict[str, int] = defaultdict(int)  # stream -> cumulative offset
        self.current_event = 0
        self.max_clocks: Dict[str, int] = defaultdict(lambda: 0)  # stream -> max clock seen
        self.conflict_counts: Dict[str, int] = defaultdict(int)
        self.flags_by_stream: Dict[str, Set[str]] = defaultdict(set)
        
    def clone(self):
        new_state = EventState()
        new_state.streams = deepcopy(self.streams)
        new_state.stream_aliases = deepcopy(self.stream_aliases)
        new_state.seq_offsets = deepcopy(self.seq_offsets)
        new_state.current_event = self.current_event
        new_state.max_clocks = deepcopy(self.max_clocks)
        new_state.conflict_counts = deepcopy(self.conflict_counts)
        new_state.flags_by_stream = deepcopy(self.flags_by_stream)
        return new_state

def parse_csv_line(s: str) -> dict:
    """Parse legacy CSV format"""
    entries = []
    for part in s.split(','):
        parts = part.split(':')
        if len(parts) < 3:
            continue
        stream, seq_str, op = parts[0].strip(), parts[1].strip(), parts[2].strip()
        value = parts[3].strip() if len(parts) > 3 else None
        try:
            # Try to parse value as int, otherwise keep as string
            if value is not None:
                try:
                    value = int(value)
                except ValueError:
                    pass
            entries.append({"stream": stream, "seq": int(seq_str), "op": op, "value": value})
        except (ValueError, IndexError):
            continue
    return {"type": "event", "entries": entries}

def normalize_line(obj, next_event_id: int) -> Optional[dict]:
    """Normalize various line formats to canonical object form"""
    if isinstance(obj, list):
        # Array format - auto-assign to "default" stream
        entries = []
        for idx, item in enumerate(obj):
            if isinstance(item, dict) and "op" in item:
                entries.append({
                    "stream": "default",
                    "seq": idx + 1,
                    "op": item["op"],
                    "value": item.get("value")
                })
        return {"event_id": next_event_id, "type": "event", "entries": entries, "metadata": {}}
    
    if isinstance(obj, dict):
        t = obj.get("type")
        
        if not isinstance(obj.get("event_id"), int):
            obj["event_id"] = next_event_id
            
        if t == "event":
            entries = obj.get("entries", [])
            if not isinstance(entries, list):
                entries = []
            fixed = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                stream = e.get("stream")
                seq = e.get("seq")
                op = e.get("op")
                
                if not isinstance(stream, str) or not isinstance(seq, int) or not isinstance(op, str):
                    continue
                
                entry = {
                    "stream": stream,
                    "seq": seq,
                    "op": op,
                    "value": e.get("value")
                }
                if "clock" in e and isinstance(e["clock"], int):
                    entry["clock"] = e["clock"]
                fixed.append(entry)
                
            obj["entries"] = fixed
            if "metadata" not in obj:
                obj["metadata"] = {}
            return obj
            
        elif t in ("merge", "snapshot", "reorder"):
            return obj
            
    return None

def load_events(events_path: Path) -> List[dict]:
    """Load and normalize all line formats"""
    objs = []
    if not events_path.exists():
        return objs
    
    next_eid = 1
    for raw in events_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
            
        try:
            parsed = json.loads(s)
        except:
            if ':' in s and ',' in s:
                parsed = parse_csv_line(s)
            else:
                continue
                
        v = normalize_line(parsed, next_eid)
        if v is None:
            continue
            
        objs.append(v)
        next_eid = max(next_eid, v.get("event_id", next_eid)) + 1
        
    objs.sort(key=lambda x: x.get("event_id", 0))
    return objs

def resolve_stream(state: EventState, stream: str) -> str:
    """Resolve stream aliases to primary stream"""
    while stream in state.stream_aliases:
        stream = state.stream_aliases[stream]
    return stream

def apply_merge(state: EventState, primary: str, secondaries: List[str], strategy: str):
    """
    Merge secondary streams into primary with conflict resolution.
    
    FIXED: Now implements complete N-way merge logic:
    1. Reliability-weighted conflict resolution (CONFLICT_THRESHOLD check)
    2. Unanimous strategy checks ALL streams, not just 2-way
    3. Majority strategy implements proper vote counting across N streams
    """
    primary = resolve_stream(state, primary)
    
    # Collect all entries from all secondary streams with their metadata
    all_secondary_entries = {}  # seq -> list of (entry, stream_name)
    
    for secondary in secondaries:
        secondary = resolve_stream(state, secondary)
        if secondary == primary:
            continue
            
        # Mark secondary as alias of primary
        state.stream_aliases[secondary] = primary
        
        # Collect entries from this secondary
        sec_entries = state.streams.get(secondary, {})
        for seq, entry in sec_entries.items():
            adjusted_seq = seq + state.seq_offsets[secondary]
            target_seq = adjusted_seq + state.seq_offsets[primary]
            
            if target_seq not in all_secondary_entries:
                all_secondary_entries[target_seq] = []
            all_secondary_entries[target_seq].append((entry, secondary))
    
    # Now process each sequence position
    for target_seq, candidates in all_secondary_entries.items():
        if target_seq in state.streams[primary]:
            # Conflict with existing entry in primary
            existing = state.streams[primary][target_seq]
            all_entries = [(existing, primary)] + candidates
            
            # Check reliability-weighted resolution FIRST (applies to all strategies)
            # If reliability difference > CONFLICT_THRESHOLD, auto-resolve to higher reliability
            reliabilities = []
            for entry, src_stream in all_entries:
                # Try to get reliability from entry metadata if available
                rel = entry.get("_reliability", 1.0)  # Default to 1.0
                reliabilities.append((rel, entry, src_stream))
            
            reliabilities.sort(key=lambda x: x[0], reverse=True)
            max_rel = reliabilities[0][0]
            
            # Check if any entry has significantly higher reliability
            auto_resolved = False
            for rel, entry, src_stream in reliabilities[1:]:
                if max_rel - rel > CONFLICT_THRESHOLD:
                    # Auto-resolve to highest reliability entry
                    state.streams[primary][target_seq] = reliabilities[0][1]
                    state.flags_by_stream[primary].add("MERGE_CONFLICT")
                    state.conflict_counts[primary] += 1
                    auto_resolved = True
                    break
            
            if auto_resolved:
                continue
            
            # Apply merge strategy if no auto-resolution
            if strategy == "latest":
                # Use entry with highest clock, or keep existing if no clocks
                entries_with_clocks = [(e, s) for e, s in all_entries if "clock" in e]
                if entries_with_clocks:
                    entries_with_clocks.sort(key=lambda x: x[0]["clock"], reverse=True)
                    winner = entries_with_clocks[0][0]
                    if winner != existing:
                        state.streams[primary][target_seq] = winner
                        state.flags_by_stream[primary].add("MERGE_CONFLICT")
                # else keep existing
                state.conflict_counts[primary] += 1
                        
            elif strategy == "unanimous":
                # FIXED: Check if ALL streams agree on value (N-way, not just 2-way)
                values = [e.get("value") for e, s in all_entries]
                if len(set(str(v) for v in values)) == 1:
                    # All agree - keep existing (which has the agreed value)
                    pass
                else:
                    # Not unanimous - remove entry
                    del state.streams[primary][target_seq]
                    state.flags_by_stream[primary].add("MERGE_CONFLICT")
                state.conflict_counts[primary] += 1
                        
            elif strategy == "majority":
                # FIXED: Implement proper majority vote counting across N streams
                value_votes = Counter()
                for entry, src_stream in all_entries:
                    value_votes[str(entry.get("value"))] += 1
                
                # Find majority (>50% of streams)
                total_streams = len(all_entries)
                majority_threshold = total_streams / 2.0
                
                most_common = value_votes.most_common(1)
                if most_common and most_common[0][1] > majority_threshold:
                    # Found majority - use entry with this value
                    majority_value = most_common[0][0]
                    for entry, src_stream in all_entries:
                        if str(entry.get("value")) == majority_value:
                            state.streams[primary][target_seq] = entry
                            break
                else:
                    # No majority - keep existing (arbitrary but deterministic)
                    pass
                    
                state.flags_by_stream[primary].add("MERGE_CONFLICT")
                state.conflict_counts[primary] += 1
        else:
            # No conflict with primary - just add first secondary entry
            # (If multiple secondaries have same seq, use first one - deterministic)
            state.streams[primary][target_seq] = candidates[0][0]

def apply_reorder(state: EventState, stream: str, offset: int):
    """Apply sequence number offset to all entries in stream"""
    stream = resolve_stream(state, stream)
    state.seq_offsets[stream] += offset
    
    # Rebuild stream with new sequence numbers
    old_entries = state.streams.get(stream, {})
    new_entries = {}
    
    for seq, entry in old_entries.items():
        new_seq = seq + offset
        if new_seq in new_entries:
            # Collision after reordering - keep earliest
            state.conflict_counts[stream] += 1
            state.flags_by_stream[stream].add("REORDER_COLLISION")
        else:
            new_entries[new_seq] = entry
    
    state.streams[stream] = new_entries

def compute_metrics(state: EventState, stream: str, window_start: int, window_end: int, 
                    consistency: str) -> Tuple[int, int, int, int, int, str, str]:
    """Compute all metrics for a stream in given consistency mode"""
    stream = resolve_stream(state, stream)
    entries = state.streams.get(stream, {})
    
    if not entries:
        return 0, 0, 0, 0, 0, "", ""
    
    # Total entries
    total_entries = len(entries)
    
    # Unique operations
    ops = set(e.get("op") for e in entries.values() if "op" in e)
    unique_ops = len(ops)
    
    # Sequence gaps
    seqs = sorted(entries.keys())
    seq_gaps = 0
    for i in range(len(seqs) - 1):
        if seqs[i+1] - seqs[i] > 1:
            seq_gaps += 1
    
    # Conflicts
    conflicts = state.conflict_counts.get(stream, 0)
    
    # Causal violations (only for CAUSAL and LINEARIZABLE)
    causal_violations = 0
    if consistency in ["CAUSAL", "LINEARIZABLE"]:
        # Check for clock ordering violations
        clocked_entries = [(seq, e) for seq, e in sorted(entries.items()) if "clock" in e]
        for i in range(len(clocked_entries)):
            for j in range(max(0, i - CAUSALITY_WINDOW_SIZE), i):
                if clocked_entries[i][1]["clock"] < clocked_entries[j][1]["clock"]:
                    causal_violations += 1
                    state.flags_by_stream[stream].add("CAUSAL_VIOLATION")
    
    # Average clock delta
    avg_clock_delta = ""
    clocked_entries = sorted([(seq, e["clock"]) for seq, e in entries.items() if "clock" in e])
    if len(clocked_entries) > 1:
        deltas = [clocked_entries[i+1][1] - clocked_entries[i][1] 
                 for i in range(len(clocked_entries) - 1)]
        avg_clock_delta = str(round(sum(deltas) / len(deltas), 2))
    
    # Flags
    flags = ",".join(sorted(state.flags_by_stream.get(stream, set())))
    
    return total_entries, unique_ops, seq_gaps, conflicts, causal_violations, avg_clock_delta, flags

def process_events(objs: List[dict]) -> List[List[str]]:
    """Process complete event stream with all advanced features"""
    state = EventState()
    snapshots: Dict[str, EventState] = {}
    rows = []
    
    i = 0
    while i < len(objs):
        obj = objs[i]
        state.current_event += 1
        event_idx = state.current_event
        
        t = obj["type"]
        
        if t == "event":
            reliability = obj.get("metadata", {}).get("reliability", 1.0)
            
            for entry in obj.get("entries", []):
                stream = resolve_stream(state, entry["stream"])
                seq = entry["seq"] + state.seq_offsets[stream]
                
                # Check for low reliability
                if reliability < MIN_RELIABILITY:
                    state.flags_by_stream[stream].add("LOW_RELIABILITY")
                
                # Track max clock
                if "clock" in entry:
                    state.max_clocks[stream] = max(state.max_clocks[stream], entry["clock"])
                
                # FIXED: Check for conflicts with reliability-based resolution
                if seq in state.streams[stream]:
                    existing = state.streams[stream][seq]
                    existing_reliability = existing.get("_reliability", 1.0)
                    
                    # Check if reliability difference exceeds threshold
                    if reliability - existing_reliability > CONFLICT_THRESHOLD:
                        # New entry has significantly higher reliability - replace
                        entry_copy = entry.copy()
                        entry_copy["_reliability"] = reliability
                        state.streams[stream][seq] = entry_copy
                        state.flags_by_stream[stream].add("SEQ_CONFLICT")
                    elif existing_reliability - reliability > CONFLICT_THRESHOLD:
                        # Existing has significantly higher reliability - keep it
                        state.flags_by_stream[stream].add("SEQ_CONFLICT")
                    else:
                        # No clear winner - keep existing (first-arrival wins)
                        state.flags_by_stream[stream].add("SEQ_CONFLICT")
                    
                    state.conflict_counts[stream] += 1
                    continue
                
                # No conflict - add entry with reliability metadata
                entry_copy = entry.copy()
                entry_copy["_reliability"] = reliability
                state.streams[stream][seq] = entry_copy
                
        elif t == "merge":
            primary = obj.get("primary_stream")
            secondaries = obj.get("secondary_streams", [])
            strategy = obj.get("strategy", "latest")
            apply_merge(state, primary, secondaries, strategy)
            
        elif t == "snapshot":
            snap_id = obj.get("snapshot_id")
            action = obj.get("action")
            
            if action == "save" and snap_id:
                snapshots[snap_id] = state.clone()
            elif action == "restore" and snap_id and snap_id in snapshots:
                # Restore state and continue from next line
                # CLARIFIED: Snapshot restore does NOT generate output rows
                # It simply resets state and processing continues
                state = snapshots[snap_id].clone()
                i += 1
                continue
                
        elif t == "reorder":
            stream = obj.get("stream")
            offset = obj.get("new_offset", 0)
            apply_reorder(state, stream, offset)
        
        # Compute window boundaries
        window_start = max(1, event_idx - CAUSALITY_WINDOW_SIZE + 1)
        window_end = event_idx
        
        # Get all streams (resolve aliases)
        all_streams = set()
        for s in state.streams.keys():
            resolved = resolve_stream(state, s)
            if state.streams[resolved]:  # Only if has entries
                all_streams.add(resolved)
        
        base = [str(event_idx), str(window_start), str(window_end)]
        
        # Generate rows for each consistency mode
        for consistency in CONSISTENCY_MODES:
            # GLOBAL row
            global_total = sum(len(state.streams[s]) for s in all_streams)
            if global_total > 0:
                global_ops = set()
                global_conflicts = 0
                global_violations = 0
                global_deltas = []
                global_flags = set()
                
                for s in all_streams:
                    entries = state.streams[s]
                    global_ops.update(e.get("op") for e in entries.values() if "op" in e)
                    global_conflicts += state.conflict_counts.get(s, 0)
                    global_flags.update(state.flags_by_stream.get(s, set()))
                    
                    if consistency in ["CAUSAL", "LINEARIZABLE"]:
                        clocked = [(seq, e) for seq, e in sorted(entries.items()) if "clock" in e]
                        for idx in range(len(clocked)):
                            for jdx in range(max(0, idx - CAUSALITY_WINDOW_SIZE), idx):
                                if clocked[idx][1]["clock"] < clocked[jdx][1]["clock"]:
                                    global_violations += 1
                    
                    clocked = sorted([e["clock"] for e in entries.values() if "clock" in e])
                    if len(clocked) > 1:
                        global_deltas.extend([clocked[i+1] - clocked[i] for i in range(len(clocked)-1)])
                
                avg_delta = ""
                if global_deltas:
                    avg_delta = str(round(sum(global_deltas) / len(global_deltas), 2))
                
                flags_str = ",".join(sorted(global_flags))
                
                rows.append(base + ["GLOBAL", consistency, str(global_total), str(len(global_ops)), 
                           "0", str(global_conflicts), str(global_violations), avg_delta, flags_str])
            
            # Per-stream rows
            for stream in sorted(all_streams):
                metrics = compute_metrics(state, stream, window_start, window_end, consistency)
                total, unique, gaps, conflicts, violations, avg_delta, flags = metrics
                
                if total > 0:
                    rows.append(base + [stream, consistency, str(total), str(unique), str(gaps),
                               str(conflicts), str(violations), avg_delta, flags])
        
        i += 1
    
    return rows

def expected_df(event_objs: List[dict]):
    """Compute canonical expected rows"""
    rows = process_events(event_objs)
    if not rows:
        return None
    
    # Convert to DataFrame-like structure for comparison
    return rows

def read_solution(sol_path: Path) -> Optional[List[List[str]]]:
    """Load solution CSV"""
    if not sol_path.exists():
        return None
    try:
        with sol_path.open('r') as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != COLS:
                return None
            rows = list(reader)
        return rows
    except Exception:
        return None

def grade(transcript: str | None = None) -> GradingResult:
    """
    Grade by recomputing expected output with all advanced features.
    Returns score=1.0 only if ALL checks pass, otherwise 0.0.
    """
    subscores = {"all_passes": 0.0}
    weights = {"all_passes": 1.0}

    event_objs = load_events(EVENTS)
    if not event_objs:
        return GradingResult(
            score=0.0,
            feedback="No usable lines in data/events.jsonl",
            subscores=subscores,
            weights=weights
        )

    exp = expected_df(event_objs)
    if exp is None:
        return GradingResult(
            score=0.0,
            feedback="Failed to compute expected output",
            subscores=subscores,
            weights=weights
        )

    sol = read_solution(SOL)
    if sol is None:
        return GradingResult(
            score=0.0,
            feedback="Missing or unreadable /workdir/sol.csv, or incorrect schema/columns",
            subscores=subscores,
            weights=weights
        )

    if len(sol) != len(exp):
        return GradingResult(
            score=0.0,
            feedback=f"Row count mismatch. Expected {len(exp)}, got {len(sol)}.",
            subscores=subscores,
            weights=weights
        )

    mismatches = []
    for i in range(len(exp)):
        exp_row = exp[i]
        sol_row = sol[i]
        
        if len(sol_row) != len(COLS):
            mismatches.append({
                "row": i + 1,
                "issue": f"Column count mismatch. Expected {len(COLS)}, got {len(sol_row)}"
            })
            continue
        
        for col_idx, col_name in enumerate(COLS):
            exp_val = str(exp_row[col_idx])
            sol_val = str(sol_row[col_idx])
            if exp_val != sol_val:
                mismatches.append({
                    "row": i + 1,
                    "column": col_name,
                    "expected": exp_val,
                    "got": sol_val
                })
                if len(mismatches) >= 50:
                    break
        if len(mismatches) >= 50:
            break
    
    if mismatches:
        return GradingResult(
            score=0.0,
            feedback=f"Found {len(mismatches)}+ cell mismatches. Output does not match expected.",
            subscores=subscores,
            weights=weights,
            details={"mismatches": mismatches[:25]}
        )

    subscores["all_passes"] = 1.0
    return GradingResult(
        score=1.0,
        feedback="All checks passed! Complex distributed event reconciliation implemented correctly.",
        subscores=subscores,
        weights=weights
    )

if __name__ == "__main__":
    result = grade(None)
    print(result.model_dump() if hasattr(result, "model_dump") else result.__dict__)