#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PY'
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Advanced Reference Solution for Streaming K-Way Percentiles with Complex Features.

This solution implements a sophisticated streaming data processing system with:

Core Features:
- Sliding window aggregation with dynamic window sizing (3-20 batches)
- Weighted nearest-rank percentiles (10th, 25th, 50th, 75th, 90th, 95th, 99th)
- Checkpointing: save and restore complete processing state
- Selective retractions: remove specific keys from target batches
- Time-based exponential weight decay using timestamps
- Multiple input format support (JSON objects, arrays, legacy CSV)
- Priority-based batch processing
- Conditional aggregations (GLOBAL, HIGH_WEIGHT, LOW_WEIGHT, POSITIVE, NEGATIVE)
- Optional moving average smoothing via environment variable

Input Format Support:
1. JSON Object: {"batch_id": 1, "type": "data", "records": [...], "metadata": {...}}
2. JSON Array: [10, 20, 30] (auto-converts to key "A" with weight 1)
3. CSV Legacy: A:10:2,B:20,C:30:5 (colon-separated, comma-delimited)

Output Schema (CSV):
  ingest_index,window_start_ingest,window_end_ingest,window_size,scope,key,p10,p25,p50,p75,p90,p95,p99

Algorithm Overview:
1. Load and normalize all input lines with format tolerance
2. Sort by batch_id and priority
3. Process each ingest sequentially:
   - Handle command types (data/retract/checkpoint/adjust_window)
   - Apply time decay to weights based on timestamps
   - Maintain sliding window of configurable size
   - Apply retractions (selective or full)
   - Compute conditional aggregations
   - Calculate percentiles for each scope
4. Write results to /workdir/sol.csv

Key Implementation Details:
- State is checkpointed via deep copying all data structures
- Floating-point arithmetic for time decay (rounded to 2 decimals)
- Weights below MIN_WEIGHT (0.1) threshold are excluded
- Empty aggregations (total weight < 0.1) omitted from output
- Deterministic processing (no randomness, stable sorting)
"""
import json, math, csv, os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

# Constants
PCTS = [10, 25, 50, 75, 90, 95, 99]
INITIAL_W = 6
DECAY_FACTOR = 0.95
MIN_WEIGHT = 0.1

STREAM = Path("/workdir/data/stream.jsonl")
OUT = Path("/workdir/sol.csv")
SMOOTH = os.environ.get("SMOOTH_PERCENTILES", "false").lower() == "true"

class StreamState:
    """
    Complete state container for streaming data processing with checkpointing support.
    
    This class encapsulates all mutable state needed to process the streaming data,
    enabling save/restore checkpoint functionality through deep cloning.
    
    Attributes:
        window_size (int): Current sliding window size (batches), adjustable 3-20, default 6
        contrib (Dict[int, Dict[str, Counter]]): Record contributions by ingest
            Structure: ingest_idx -> key -> Counter(value -> effective_weight)
            Stores weighted value distributions after time decay application
        typ (Dict[int, str]): Operation type for each ingest
            Values: "data", "retract", "checkpoint", "adjust_window"
        target (Dict[int, int]): For retract ingests, which batch_id to retract from
        selective_keys (Dict[int, List[str]]): For selective retracts, which keys to remove
            If present for a retract, only remove these keys from target batch
        timestamps (Dict[int, Dict[str, List[Tuple[int, int]]]]): Timestamp metadata
            Structure: ingest_idx -> key -> [(value, unix_timestamp), ...]
            Used for time decay calculations
        metadata (Dict[int, dict]): Batch metadata (priority, source, etc.)
        current_ingest (int): Current ingest index (1-indexed)
        latest_timestamp (Optional[int]): Most recent timestamp seen (unix epoch seconds)
            Used as reference point for decay calculations
        prev_percentiles (List[Dict[str, List[float]]]): Historical percentile data
            Used for optional moving average smoothing feature
            
    Methods:
        clone(): Deep copy entire state for checkpoint save operation
    """
    def __init__(self):
        self.window_size = INITIAL_W
        self.contrib: Dict[int, Dict[str, Counter]] = {}
        self.typ: Dict[int, str] = {}
        self.target: Dict[int, int] = {}
        self.selective_keys: Dict[int, List[str]] = {}
        self.timestamps: Dict[int, Dict[str, List[Tuple[int, int]]]] = {}
        self.metadata: Dict[int, dict] = {}
        self.current_ingest = 0
        self.latest_timestamp = None
        self.prev_percentiles: List[Dict[str, List[float]]] = []
        
    def clone(self):
        """
        Create a complete deep copy of the state for checkpointing.
        
        This method is essential for implementing checkpoint save functionality.
        It creates an entirely independent copy of all state, including all
        nested data structures (dictionaries, Counters, lists).
        
        Returns:
            StreamState: A new StreamState instance with identical values but
                        independent memory (changes to one don't affect the other)
                        
        Note:
            Uses deepcopy to ensure nested structures are fully cloned.
            This prevents aliasing bugs where restored state shares references
            with current state.
        """
        new_state = StreamState()
        new_state.window_size = self.window_size
        new_state.contrib = deepcopy(self.contrib)
        new_state.typ = deepcopy(self.typ)
        new_state.target = deepcopy(self.target)
        new_state.selective_keys = deepcopy(self.selective_keys)
        new_state.timestamps = deepcopy(self.timestamps)
        new_state.metadata = deepcopy(self.metadata)
        new_state.current_ingest = self.current_ingest
        new_state.latest_timestamp = self.latest_timestamp
        new_state.prev_percentiles = deepcopy(self.prev_percentiles)
        return new_state

def parse_csv_line(s: str) -> dict:
    """
    Parse legacy CSV format into normalized data batch structure.
    
    Format: key1:val1:weight1,key2:val2:weight2,...
    - Colon-separated tuples for each record
    - Comma-delimited between records
    - Weight is optional (defaults to 1)
    - Whitespace is stripped from keys
    
    Args:
        s (str): CSV-formatted line
        
    Returns:
        dict: Normalized batch object with structure:
            {"type": "data", "records": [{"key": str, "value": int, "weight": int}, ...]}
            
    Examples:
        "A:10:2,B:20,C:30:5" → 3 records with keys A, B, C
        "X:100" → single record X with value 100, weight 1
        "A:10:0,B:20" → A dropped (invalid), B kept
        
    Error Handling:
        - Malformed tuples (wrong field count) are silently skipped
        - Non-integer values cause that tuple to be skipped
        - Empty result list if entire line is malformed
    """
    records = []
    for part in s.split(','):
        parts = part.split(':')
        if len(parts) < 2:
            continue
        key, val = parts[0].strip(), parts[1].strip()
        weight = int(parts[2]) if len(parts) > 2 else 1
        try:
            records.append({"key": key, "value": int(val), "weight": weight})
        except ValueError:
            continue
    return {"type": "data", "records": records}

def normalize_line(obj, next_batch_id: int) -> Optional[dict]:
    """
    Normalize parsed input to canonical object format with validation.
    
    Handles three input types:
    1. List (JSON array): [10, 20, 30] → data batch for synthetic key "A"
    2. Dict (JSON object): Full format with type, batch_id, records, etc.
    3. CSV: Pre-parsed by parse_csv_line() before this function
    
    Normalization rules for "data" batches:
    - Auto-assign batch_id if missing or invalid
    - Ensure metadata dict exists (even if empty)
    - Validate each record has string key and integer value
    - Weight defaults to 1 if missing
    - Coerce invalid weights (non-int or ≤0) to 1
    - Preserve optional timestamp field if valid integer
    - Drop malformed records silently
    
    Args:
        obj: Parsed JSON (dict or list) or pre-parsed CSV result
        next_batch_id (int): Next available auto-assigned batch_id
        
    Returns:
        Optional[dict]: Canonical object, or None to skip this line
        
    Return Structure:
        {
            "batch_id": int,
            "type": "data" | "retract" | "checkpoint" | "adjust_window",
            "records": [...],  # for data type
            "metadata": {},    # for data type
            ...                # type-specific fields
        }
        
    Note:
        Modifies obj dict in-place during normalization for efficiency.
    """
    if isinstance(obj, list):
        recs = [{"key": "A", "value": int(v), "weight": 1} for v in obj]
        return {"batch_id": next_batch_id, "type": "data", "records": recs, "metadata": {}}
    
    if isinstance(obj, dict):
        t = obj.get("type")
        
        if not isinstance(obj.get("batch_id"), int):
            obj["batch_id"] = next_batch_id
            
        if t == "data":
            recs = obj.get("records", [])
            if not isinstance(recs, list):
                recs = []
            fixed = []
            for r in recs:
                if not isinstance(r, dict):
                    continue
                k = r.get("key")
                v = r.get("value")
                w = r.get("weight", 1)
                ts = r.get("timestamp")
                
                if not isinstance(k, str) or not isinstance(v, int):
                    continue
                try:
                    w = int(w)
                except:
                    w = 1
                if w <= 0:
                    w = 1
                    
                rec = {"key": k, "value": int(v), "weight": int(w)}
                if ts is not None and isinstance(ts, int):
                    rec["timestamp"] = ts
                fixed.append(rec)
                
            obj["records"] = fixed
            if "metadata" not in obj:
                obj["metadata"] = {}
            return obj
            
        elif t in ("retract", "checkpoint", "adjust_window"):
            return obj
            
    return None

def load_stream(stream_path: Path) -> List[dict]:
    """
    Load and normalize the complete input stream with multi-format support.
    
    Processing pipeline:
    1. Read all lines from stream.jsonl file
    2. Skip blank lines and comment lines (starting with '//')
    3. Attempt JSON parsing (handles both objects and arrays)
    4. If JSON fails, attempt legacy CSV format parsing (contains ':' and ',')
    5. Normalize each successfully parsed line to canonical object structure
    6. Auto-assign batch_id in sequential order when missing/invalid
    7. Sort all batches by (batch_id ascending, priority descending)
    
    Multi-format tolerance:
    - JSON objects: {"batch_id": 1, "type": "data", ...}
    - JSON arrays: [10, 20, 30] → auto-converts to key "A" data batch
    - CSV legacy: A:10:2,B:20 → parsed to data batch with records
    
    Args:
        stream_path (Path): Path to the stream.jsonl input file
        
    Returns:
        List[dict]: Sorted list of normalized batch objects, ready for sequential processing
        
    Sorting behavior:
        Primary: batch_id (ascending) - ensures batches process in ID order
        Secondary: -priority (descending) - higher priority processed first for tied batch_ids
        
    Error handling:
        - Malformed lines that fail all parsing attempts are silently skipped
        - Empty file returns empty list (no error)
        - Missing file returns empty list (no error)
        
    Note:
        Auto-assigned batch_ids increment globally across all line types,
        ensuring no collisions even when some lines have explicit IDs.
    """
    objs = []
    if not stream_path.exists():
        return objs
    
    next_bid = 1
    for raw in stream_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
            
        try:
            parsed = json.loads(s)
        except:
            if ':' in s and ',' in s:
                parsed = parse_csv_line(s)
            else:
                continue
                
        v = normalize_line(parsed, next_bid)
        if v is None:
            continue
            
        objs.append(v)
        next_bid = max(next_bid, v.get("batch_id", next_bid)) + 1
        
    objs.sort(key=lambda x: (x.get("batch_id", 0), -x.get("metadata", {}).get("priority", 0)))
    return objs

def apply_time_decay(value: int, weight: float, timestamp: Optional[int], 
                     current_ts: Optional[int]) -> float:
    """
    Apply exponential time-based weight decay to a record.
    
    Decay model:
        effective_weight = original_weight × (DECAY_FACTOR ^ hours_elapsed)
    
    Where:
        - DECAY_FACTOR = 0.95 (5% decay per hour)
        - hours_elapsed = (current_ts - record_timestamp) / 3600.0
        - current_ts = most recent timestamp seen in entire stream
    
    Args:
        value (int): Record value (unused, kept for signature consistency)
        weight (float): Original weight from the record
        timestamp (Optional[int]): Unix epoch timestamp of this record (seconds)
        current_ts (Optional[int]): Latest timestamp observed in stream (seconds)
        
    Returns:
        float: Effective weight after exponential decay, rounded to exactly 2 decimal places
               Returns original weight unchanged if either timestamp is None
               
    Time delta calculation:
        - Clamped to non-negative: max(0, current_ts - timestamp)
        - Converted to hours by dividing by 3600.0
        
    Examples:
        - No decay: timestamps are None → returns weight as-is
        - 1 hour elapsed: weight × 0.95 ≈ weight × 0.95
        - 10 hours elapsed: weight × (0.95^10) ≈ weight × 0.599
        - 50 hours elapsed: weight × (0.95^50) ≈ weight × 0.077
        
    Determinism:
        Result is always rounded to exactly 2 decimal places for consistent
        floating-point comparison across platforms.
    """
    if timestamp is None or current_ts is None:
        return weight
    
    time_delta_hours = max(0, (current_ts - timestamp) / 3600.0)
    effective = weight * (DECAY_FACTOR ** time_delta_hours)
    return round(effective, 2)

def nearest_rank(counter: Dict[int, float], smooth_history: Optional[List[Dict[int, float]]] = None) -> List[str]:
    """
    Calculate weighted nearest-rank percentiles with optional moving average smoothing.
    
    Nearest-rank algorithm:
    1. Compute total weight N = sum of all weights in counter
    2. For each percentile p ∈ [10, 25, 50, 75, 90, 95, 99]:
       a. Calculate rank position: r = ceil(p/100 × N)
       b. Sort values ascending
       c. Accumulate weights until cumulative weight ≥ r
       d. Return the value at that position
    
    Smoothing (when SMOOTH_PERCENTILES=true and history provided):
    - Applies weighted moving average to GLOBAL scope percentiles only
    - Weights: current=0.5, previous_1=0.3, previous_2=0.2
    - Fallback for limited history:
        * 2 ingests available: 0.625 current, 0.375 previous
        * 1 ingest available: no smoothing (use current only)
    
    Args:
        counter (Dict[int, float]): Mapping from value → effective_weight
        smooth_history (Optional[List[Dict[int, float]]]): Previous counter states for smoothing
            List is ordered chronologically (oldest to newest)
            Only last 2 entries are used for smoothing
            
    Returns:
        List[str]: Seven percentile values as strings ["p10", "p25", ..., "p99"]
                   Empty string "" for each percentile if total weight < MIN_WEIGHT (0.1)
                   All values rounded to integers before string conversion
                   
    Edge cases:
        - Empty counter or N < 0.1 → returns ["", "", "", "", "", "", ""]
        - Single value in counter → all percentiles return that value
        - Smoothing with insufficient history → gracefully degrades
        
    Example:
        counter = {10: 2.5, 20: 3.0, 30: 1.5}  # N = 7.0
        For p50: r = ceil(50/100 × 7) = ceil(3.5) = 4
        Cumulative: 10(2.5), 20(5.5) → p50 = "20"
        
    Note:
        Smoothing only applies when SMOOTH environment variable is explicitly true
        and smooth_history contains valid data. Otherwise, raw percentiles are returned.
    """
    N = sum(counter.values())
    if N < MIN_WEIGHT:
        return [""] * len(PCTS)
    
    items = sorted(counter.items())
    out = []
    
    for p in PCTS:
        r = math.ceil(p / 100 * N)
        cum = 0
        pval = None
        for v, c in items:
            cum += c
            if cum >= r:
                pval = v
                break
        
        if pval is not None and SMOOTH and smooth_history:
            vals = [pval]
            for hist in smooth_history[-2:]:
                hist_items = sorted(hist.items())
                hist_N = sum(hist.values())
                if hist_N < MIN_WEIGHT:
                    continue
                hist_r = math.ceil(p / 100 * hist_N)
                hist_cum = 0
                for hv, hc in hist_items:
                    hist_cum += hc
                    if hist_cum >= hist_r:
                        vals.append(hv)
                        break
            
            if len(vals) == 3:
                pval = int(0.5 * vals[0] + 0.3 * vals[1] + 0.2 * vals[2])
            elif len(vals) == 2:
                pval = int(0.625 * vals[0] + 0.375 * vals[1])
        
        out.append(str(int(pval)) if pval is not None else "")
    
    return out

def process_stream(objs: List[dict]) -> List[List[str]]:
    """
    Process the complete ingestion stream with all advanced features.
    
    This is the main processing engine that orchestrates:
    - Sequential ingest processing with state management
    - Dynamic window sizing (adjustable 3-20 batches)
    - Checkpointing (save/restore complete state snapshots)
    - Data ingestion with time-based weight decay
    - Retraction handling (full or selective by key)
    - Conditional aggregations across multiple scopes
    - Percentile calculation with optional smoothing
    
    Processing flow per ingest:
    ┌─────────────────────────────────────────────┐
    │ 1. Process command by type                  │
    │    - data: apply decay, store contributions │
    │    - retract: mark target for removal       │
    │    - checkpoint: save/restore state         │
    │    - adjust_window: change window size      │
    ├─────────────────────────────────────────────┤
    │ 2. Define sliding window [ws, we]           │
    │    ws = max(1, current - window_size + 1)   │
    │    we = current                             │
    ├─────────────────────────────────────────────┤
    │ 3. Aggregate window data by key             │
    │    - Sum contributions from data batches    │
    │    - Apply retractions if target in window  │
    ├─────────────────────────────────────────────┤
    │ 4. Generate conditional aggregations        │
    │    - GLOBAL: all data combined              │
    │    - HIGH_WEIGHT: weight ≥ 5                │
    │    - LOW_WEIGHT: weight < 5                 │
    │    - POSITIVE: value > 0                    │
    │    - NEGATIVE: value ≤ 0                    │
    ├─────────────────────────────────────────────┤
    │ 5. Calculate percentiles for each scope     │
    │    - Use nearest-rank method                │
    │    - Apply smoothing if enabled             │
    ├─────────────────────────────────────────────┤
    │ 6. Emit output rows in canonical order      │
    │    - GLOBAL, HIGH_WEIGHT, LOW_WEIGHT,       │
    │      POSITIVE, NEGATIVE, then KEY rows      │
    │    - Skip scopes with weight < MIN_WEIGHT   │
    └─────────────────────────────────────────────┘
    
    Args:
        objs (List[dict]): Normalized and sorted batch objects from load_stream()
        
    Returns:
        List[List[str]]: Output rows for CSV, each row containing:
            [ingest_idx, window_start, window_end, window_size, scope, key,
             p10, p25, p50, p75, p90, p95, p99]
            
    Row ordering per ingest (conditional scopes only if weight ≥ 0.1):
        1. GLOBAL (scope="GLOBAL", key="")
        2. HIGH_WEIGHT (scope="HIGH_WEIGHT", key="")
        3. LOW_WEIGHT (scope="LOW_WEIGHT", key="")
        4. POSITIVE (scope="POSITIVE", key="")
        5. NEGATIVE (scope="NEGATIVE", key="")
        6. KEY rows (scope="KEY", key=<actual_key>), lexicographically sorted
        
    Special behaviors:
        - Checkpoint restore: Jumps processing to saved state, skips intervening ingests
        - Window size changes: Persist across checkpoints, take effect immediately
        - Selective retractions: Only specified keys removed from target batch
        - Time decay: Recalculated based on latest_timestamp across entire stream
        - Empty scopes: Automatically omitted from output (no empty rows)
        
    State management:
        Maintains complete state via StreamState object, enabling:
        - Multiple named checkpoints (cp_id → state mapping)
        - State restoration with full rollback
        - History tracking for smoothing calculations
        
    Note:
        Loop uses manual index management (while i < len(objs)) because
        checkpoint restore needs to skip forward without resuming the
        normal iteration sequence.
    """
    state = StreamState()
    checkpoints: Dict[str, StreamState] = {}
    rows = []
    
    i = 0
    while i < len(objs):
        obj = objs[i]
        state.current_ingest += 1
        ingest_idx = state.current_ingest
        
        t = obj["type"]
        state.typ[ingest_idx] = t
        
        if t == "data":
            for r in obj.get("records", []):
                if "timestamp" in r:
                    if state.latest_timestamp is None or r["timestamp"] > state.latest_timestamp:
                        state.latest_timestamp = r["timestamp"]
            
            m = defaultdict(Counter)
            ts_map = defaultdict(list)
            for r in obj.get("records", []):
                effective_w = apply_time_decay(
                    r["value"], 
                    r["weight"], 
                    r.get("timestamp"),
                    state.latest_timestamp
                )
                if effective_w >= MIN_WEIGHT:
                    m[r["key"]][r["value"]] += effective_w
                    if "timestamp" in r:
                        ts_map[r["key"]].append((r["value"], r["timestamp"]))
            
            state.contrib[ingest_idx] = m
            state.timestamps[ingest_idx] = ts_map
            state.metadata[ingest_idx] = obj.get("metadata", {})
            
        elif t == "retract":
            state.target[ingest_idx] = obj.get("target_batch_id")
            if "selective" in obj and "keys" in obj["selective"]:
                state.selective_keys[ingest_idx] = obj["selective"]["keys"]
                
        elif t == "checkpoint":
            cp_id = obj.get("checkpoint_id")
            action = obj.get("action")
            
            if action == "save" and cp_id:
                checkpoints[cp_id] = state.clone()
            elif action == "restore" and cp_id and cp_id in checkpoints:
                state = checkpoints[cp_id].clone()
                i += 1
                continue
                
        elif t == "adjust_window":
            new_size = obj.get("new_window_size", state.window_size)
            state.window_size = max(3, min(20, new_size))
        
        ws = max(1, ingest_idx - state.window_size + 1)
        we = ingest_idx
        idxs = range(ws, we + 1)
        
        by_key = defaultdict(lambda: Counter())
        
        for j in idxs:
            if state.typ.get(j) == "data":
                for k, cnt in state.contrib.get(j, {}).items():
                    by_key[k].update(cnt)
        
        for j in idxs:
            if state.typ.get(j) == "retract":
                tgt = state.target.get(j)
                if tgt and ws <= tgt <= we and state.typ.get(tgt) == "data":
                    sel_keys = state.selective_keys.get(j)
                    for k, cnt in state.contrib.get(tgt, {}).items():
                        if sel_keys and k not in sel_keys:
                            continue
                        for v, c in cnt.items():
                            by_key[k][v] -= c
                            if by_key[k][v] <= MIN_WEIGHT:
                                del by_key[k][v]
        
        global_counter = Counter()
        high_weight = Counter()
        low_weight = Counter()
        positive = Counter()
        negative = Counter()
        
        for k, cnt in by_key.items():
            for v, w in cnt.items():
                global_counter[v] += w
                if w >= 5:
                    high_weight[v] += w
                else:
                    low_weight[v] += w
                if v > 0:
                    positive[v] += w
                else:
                    negative[v] += w
        
        base = [str(ingest_idx), str(ws), str(we), str(state.window_size)]
        
        if sum(global_counter.values()) >= MIN_WEIGHT:
            rows.append(base + ["GLOBAL", ""] + nearest_rank(global_counter, state.prev_percentiles))
        
        if sum(high_weight.values()) >= MIN_WEIGHT:
            rows.append(base + ["HIGH_WEIGHT", ""] + nearest_rank(high_weight))
        
        if sum(low_weight.values()) >= MIN_WEIGHT:
            rows.append(base + ["LOW_WEIGHT", ""] + nearest_rank(low_weight))
        
        if sum(positive.values()) >= MIN_WEIGHT:
            rows.append(base + ["POSITIVE", ""] + nearest_rank(positive))
        
        if sum(negative.values()) >= MIN_WEIGHT:
            rows.append(base + ["NEGATIVE", ""] + nearest_rank(negative))
        
        for k in sorted(k for k, c in by_key.items() if sum(c.values()) >= MIN_WEIGHT):
            rows.append(base + ["KEY", k] + nearest_rank(by_key[k]))
        
        state.prev_percentiles.append(global_counter)
        
        i += 1
    
    return rows

# Main execution
objs = load_stream(STREAM)
rows = process_stream(objs)

# Write CSV
with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    cols = ["ingest_index", "window_start_ingest", "window_end_ingest", "window_size", "scope", "key"] + [f"p{p}" for p in PCTS]
    w.writerow(cols)
    w.writerows(rows)
PY
