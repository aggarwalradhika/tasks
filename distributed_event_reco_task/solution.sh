#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PY'
#!/usr/bin/env python3
"""
Advanced Reference Solution for Distributed Event Log Reconciliation.

Core Features:
- Multiple consistency mode analysis (eventual, causal, linearizable)
- Stream merging with conflict resolution strategies (latest, unanimous, majority)
- State snapshotting: save and restore complete processing state
- Sequence reordering: retroactive adjustment of sequence numbers
- Logical clock-based causal ordering detection
- Reliability-based weighting for conflict resolution
- Multiple input format support (JSON objects, arrays, legacy CSV)
- Causality window analysis (5-event sliding window)

Input Format Support:
1. JSON Object: {"event_id": 1, "type": "event", "entries": [...], "metadata": {...}}
2. JSON Array: [{"op": "SET", "value": 100}, ...] (auto-converts to stream "default")
3. CSV Legacy: A:1:SET:100,B:2:DELETE:foo (colon-separated, comma-delimited)

Output Schema (CSV):
  event_index,window_start,window_end,stream,consistency,total_entries,unique_ops,
  seq_gaps,conflicts,causal_violations,avg_clock_delta,flags

Algorithm Overview:
1. Load and normalize all input lines with format tolerance
2. Sort by event_id
3. Process each event sequentially:
   - Handle command types (event/merge/snapshot/reorder)
   - Resolve stream aliases from merges
   - Apply sequence offsets from reorders
   - Detect and resolve conflicts with reliability weighting
   - Track causal ordering violations
4. For each event, compute metrics across all consistency modes
5. Write results to /workdir/sol.csv

Key Implementation Details:
- State is snapshotted via deep copying all data structures
- Stream aliases redirect all operations to primary stream
- Sequence reordering applies retroactively to existing entries
- Floating-point arithmetic for clock deltas (rounded to 2 decimals)
- Low-reliability entries (< 0.1) are flagged
- Deterministic processing (no randomness, stable sorting)
- Reliability-weighted conflict resolution at both entry and merge levels
"""
import json, csv, os, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any, Set
from copy import deepcopy

# Constants
CONSISTENCY_MODES = ["EVENTUAL", "CAUSAL", "LINEARIZABLE"]
CAUSALITY_WINDOW_SIZE = 5
CONFLICT_THRESHOLD = 0.3
MIN_RELIABILITY = 0.1

EVENTS = Path("/workdir/data/events.jsonl")
OUT = Path("/workdir/sol.csv")

class EventState:
    """
    Complete state container for event processing with snapshotting support.
    
    Attributes:
        streams: Nested dict of stream -> seq -> entry_data
            Stores all entries for each stream, keyed by adjusted sequence number
        stream_aliases: Mapping from secondary stream names to primary
            Created by merge operations to redirect subsequent entries
        seq_offsets: Cumulative sequence number adjustments per stream
            Used to apply reordering retroactively
        current_event: Current event index (1-indexed)
        max_clocks: Highest logical clock seen per stream
            Used for causal ordering violation detection
        conflict_counts: Count of sequence collisions per stream
        flags_by_stream: Set of issue flags per stream
            Examples: "LOW_RELIABILITY", "MERGE_CONFLICT", "CAUSAL_VIOLATION"
    """
    def __init__(self):
        self.streams: Dict[str, Dict[int, Dict]] = defaultdict(lambda: {})
        self.stream_aliases: Dict[str, str] = {}
        self.seq_offsets: Dict[str, int] = defaultdict(int)
        self.current_event = 0
        self.max_clocks: Dict[str, int] = defaultdict(lambda: 0)
        self.conflict_counts: Dict[str, int] = defaultdict(int)
        self.flags_by_stream: Dict[str, Set[str]] = defaultdict(set)
        
    def clone(self):
        """Create a complete deep copy of the state for snapshotting"""
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
    """
    Parse legacy CSV format into normalized event batch structure.
    
    Format: stream1:seq1:op1:val1,stream2:seq2:op2:val2,...
    - Colon-separated stream:seq:op:value tuples
    - Comma-delimited between entries
    - Value is optional, can be numeric or string
    
    Returns:
        dict: Normalized event object with structure:
            {"type": "event", "entries": [{"stream": str, "seq": int, "op": str, "value": any}, ...]}
    """
    entries = []
    for part in s.split(','):
        parts = part.split(':')
        if len(parts) < 3:
            continue
        stream, seq_str, op = parts[0].strip(), parts[1].strip(), parts[2].strip()
        value = parts[3].strip() if len(parts) > 3 else None
        try:
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
    """
    Normalize parsed input to canonical object format with validation.
    
    Handles three input types:
    1. List (JSON array): [{"op": "SET", "value": 100}, ...] → event for stream "default"
    2. Dict (JSON object): Full format with type, event_id, entries, etc.
    3. CSV: Pre-parsed by parse_csv_line() before this function
    
    Normalization rules for "event" batches:
    - Auto-assign event_id if missing or invalid
    - Ensure metadata dict exists (even if empty)
    - Validate each entry has string stream, integer seq, and string op
    - Value can be any type, defaults to None if missing
    - Preserve optional clock field if valid integer
    - Drop malformed entries silently
    """
    if isinstance(obj, list):
        # Array format - auto-assign to "default" stream with sequential seq numbers
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
    """
    Load and normalize the complete input stream with multi-format support.
    
    Processing pipeline:
    1. Read all lines from events.jsonl file
    2. Skip blank lines and comment lines (starting with '#')
    3. Attempt JSON parsing (handles both objects and arrays)
    4. If JSON fails, attempt legacy CSV format parsing
    5. Normalize each successfully parsed line to canonical object structure
    6. Auto-assign event_id in sequential order when missing/invalid
    7. Sort all events by event_id
    """
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
    """
    Resolve stream aliases to primary stream.
    
    After merge operations, secondary streams are aliased to primary.
    This function follows the alias chain to find the ultimate primary stream.
    Handles transitive aliases (A -> B -> C returns C).
    """
    while stream in state.stream_aliases:
        stream = state.stream_aliases[stream]
    return stream

def apply_merge(state: EventState, primary: str, secondaries: List[str], strategy: str):
    """
    Merge secondary streams into primary with conflict resolution.
    
    FIXED: Complete N-way merge implementation:
    1. Collects all entries from all secondary streams
    2. Applies reliability-weighted conflict resolution FIRST
    3. Falls back to merge strategy if no clear reliability winner
    4. Unanimous strategy checks ALL streams (N-way)
    5. Majority strategy implements proper vote counting
    
    Merge process:
    1. Resolve all stream names to their primaries (handles chained merges)
    2. Mark each secondary as alias of primary
    3. Collect all entries from all secondaries
    4. For each sequence position with conflicts:
       a. Check reliability differences (>CONFLICT_THRESHOLD = auto-resolve)
       b. Apply merge strategy if no clear winner
    
    Conflict resolution strategies:
    - "latest": Use entry with highest logical clock (or keep existing if no clocks)
    - "unanimous": Only keep entry if ALL streams have same value (N-way check)
    - "majority": Use value that appears in >50% of streams (proper vote counting)
    
    Side effects:
    - Updates stream_aliases mapping
    - Transfers entries to primary stream
    - Increments conflict_counts for collisions
    - Adds MERGE_CONFLICT flag
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
            
            # FIXED: Check reliability-weighted resolution FIRST (applies to all strategies)
            # If reliability difference > CONFLICT_THRESHOLD, auto-resolve to higher reliability
            reliabilities = []
            for entry, src_stream in all_entries:
                # Get reliability from entry metadata
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
                # Convert to strings for comparison (handles different types)
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
    """
    Apply sequence number offset to all entries in stream retroactively.
    
    Reordering process:
    1. Resolve stream to primary (in case it's been merged)
    2. Add offset to cumulative seq_offsets for this stream
    3. Rebuild stream entries with new sequence numbers
    4. Detect and handle collisions (keep earliest entry)
    
    Use cases:
    - Correcting out-of-order arrivals from distributed sources
    - Aligning sequence numbers when merging distributed logs
    - Negative offsets to shift sequences backward
    
    Side effects:
    - Updates seq_offsets mapping
    - Rebuilds stream entries dict with new keys
    - Increments conflict_counts for collisions
    - Adds REORDER_COLLISION flag if collisions occur
    """
    stream = resolve_stream(state, stream)
    state.seq_offsets[stream] += offset
    
    old_entries = state.streams.get(stream, {})
    new_entries = {}
    
    for seq, entry in old_entries.items():
        new_seq = seq + offset
        if new_seq in new_entries:
            state.conflict_counts[stream] += 1
            state.flags_by_stream[stream].add("REORDER_COLLISION")
        else:
            new_entries[new_seq] = entry
    
    state.streams[stream] = new_entries

def compute_metrics(state: EventState, stream: str, window_start: int, window_end: int, 
                    consistency: str) -> Tuple[int, int, int, int, int, str, str]:
    """
    Compute all metrics for a stream in given consistency mode.
    
    Metrics computed:
    1. total_entries: Count of all entries in stream
    2. unique_ops: Count of distinct operation types
    3. seq_gaps: Count of discontinuities in sequence numbers
    4. conflicts: Count of sequence number collisions
    5. causal_violations: Count of logical clock ordering violations (causal/linearizable only)
    6. avg_clock_delta: Mean difference between successive logical clocks
    7. flags: Comma-separated list of issues detected
    
    Consistency mode differences:
    - EVENTUAL: No causal violation detection (always 0)
    - CAUSAL: Detects ordering violations within causality window
    - LINEARIZABLE: Same as causal (stricter guarantees would require global ordering)
    
    Returns:
        Tuple of (total, unique_ops, gaps, conflicts, violations, avg_delta_str, flags_str)
    """
    stream = resolve_stream(state, stream)
    entries = state.streams.get(stream, {})
    
    if not entries:
        return 0, 0, 0, 0, 0, "", ""
    
    total_entries = len(entries)
    
    ops = set(e.get("op") for e in entries.values() if "op" in e)
    unique_ops = len(ops)
    
    seqs = sorted(entries.keys())
    seq_gaps = 0
    for i in range(len(seqs) - 1):
        if seqs[i+1] - seqs[i] > 1:
            seq_gaps += 1
    
    conflicts = state.conflict_counts.get(stream, 0)
    
    causal_violations = 0
    if consistency in ["CAUSAL", "LINEARIZABLE"]:
        clocked_entries = [(seq, e) for seq, e in sorted(entries.items()) if "clock" in e]
        for i in range(len(clocked_entries)):
            for j in range(max(0, i - CAUSALITY_WINDOW_SIZE), i):
                if clocked_entries[i][1]["clock"] < clocked_entries[j][1]["clock"]:
                    causal_violations += 1
                    state.flags_by_stream[stream].add("CAUSAL_VIOLATION")
    
    avg_clock_delta = ""
    clocked_entries = sorted([(seq, e["clock"]) for seq, e in entries.items() if "clock" in e])
    if len(clocked_entries) > 1:
        deltas = [clocked_entries[i+1][1] - clocked_entries[i][1] 
                 for i in range(len(clocked_entries) - 1)]
        avg_clock_delta = str(round(sum(deltas) / len(deltas), 2))
    
    flags = ",".join(sorted(state.flags_by_stream.get(stream, set())))
    
    return total_entries, unique_ops, seq_gaps, conflicts, causal_violations, avg_clock_delta, flags

def process_events(objs: List[dict]) -> List[List[str]]:
    """
    Process the complete event stream with all advanced features.
    
    Main processing loop:
    1. Process each event by type
    2. For each event, emit metrics for all consistency modes
    3. Each consistency mode gets GLOBAL + per-stream rows
    4. Window boundaries track causality analysis window
    
    Row ordering per event:
    For each consistency mode (EVENTUAL, CAUSAL, LINEARIZABLE):
      1. GLOBAL row
      2. Per-stream rows (sorted alphabetically)
    
    Special handling:
    - Snapshot restore: Skip forward without normal iteration (no row output)
    - Stream aliases: All operations go to primary stream
    - Empty streams: Omitted from output
    - Manual loop control (while i < len) for snapshot jumps
    """
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
                
                if reliability < MIN_RELIABILITY:
                    state.flags_by_stream[stream].add("LOW_RELIABILITY")
                
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
                # CLARIFIED: Restore state and continue from next line
                # Snapshot restore does NOT generate output rows
                # It simply resets state and processing continues
                state = snapshots[snap_id].clone()
                i += 1
                continue
                
        elif t == "reorder":
            stream = obj.get("stream")
            offset = obj.get("new_offset", 0)
            apply_reorder(state, stream, offset)
        
        window_start = max(1, event_idx - CAUSALITY_WINDOW_SIZE + 1)
        window_end = event_idx
        
        all_streams = set()
        for s in state.streams.keys():
            resolved = resolve_stream(state, s)
            if state.streams[resolved]:
                all_streams.add(resolved)
        
        base = [str(event_idx), str(window_start), str(window_end)]
        
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

# Main execution
objs = load_events(EVENTS)
rows = process_events(objs)

# Write CSV
with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    cols = ["event_index", "window_start", "window_end", "stream", "consistency", 
            "total_entries", "unique_ops", "seq_gaps", "conflicts", "causal_violations", 
            "avg_clock_delta", "flags"]
    w.writerow(cols)
    w.writerows(rows)
PY