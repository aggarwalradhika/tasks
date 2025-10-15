#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PY'
#!/usr/bin/env python3
"""
Advanced reference solution with:
- Dynamic window sizing
- Checkpointing and state restoration
- Selective retractions
- Time-based weight decay
- Multiple percentiles and conditional aggregations
- CSV legacy format support
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
    """Encapsulates complete stream processing state for checkpointing"""
    def __init__(self):
        self.window_size = INITIAL_W
        self.contrib: Dict[int, Dict[str, Counter]] = {}  # ingest -> key -> Counter[value->weight]
        self.typ: Dict[int, str] = {}
        self.target: Dict[int, int] = {}
        self.selective_keys: Dict[int, List[str]] = {}
        self.timestamps: Dict[int, Dict[str, List[Tuple[int, int]]]] = {}  # ingest -> key -> [(value, timestamp)]
        self.metadata: Dict[int, dict] = {}
        self.current_ingest = 0
        self.latest_timestamp = None
        self.prev_percentiles: List[Dict[str, List[float]]] = []  # for smoothing
        
    def clone(self):
        """Deep copy for checkpointing"""
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
    """Parse legacy CSV format: key1:val1:weight1,key2:val2:weight2"""
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
    """Normalize various line formats to canonical object form"""
    # Array line → data for key 'A'
    if isinstance(obj, list):
        recs = [{"key": "A", "value": int(v), "weight": 1} for v in obj]
        return {"batch_id": next_batch_id, "type": "data", "records": recs, "metadata": {}}
    
    if isinstance(obj, dict):
        t = obj.get("type")
        
        # Ensure batch_id
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
    """Load and normalize all line formats"""
    objs = []
    if not stream_path.exists():
        return objs
    
    next_bid = 1
    for raw in stream_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
            
        # Try JSON first
        try:
            parsed = json.loads(s)
        except:
            # Try CSV format
            if ':' in s and ',' in s:
                parsed = parse_csv_line(s)
            else:
                continue
                
        v = normalize_line(parsed, next_bid)
        if v is None:
            continue
            
        objs.append(v)
        next_bid = max(next_bid, v.get("batch_id", next_bid)) + 1
        
    # Sort by batch_id, then by priority (descending)
    objs.sort(key=lambda x: (x.get("batch_id", 0), -x.get("metadata", {}).get("priority", 0)))
    return objs

def apply_time_decay(value: int, weight: float, timestamp: Optional[int], 
                     current_ts: Optional[int]) -> float:
    """Apply exponential time decay if timestamps present"""
    if timestamp is None or current_ts is None:
        return weight
    
    time_delta_hours = max(0, (current_ts - timestamp) / 3600.0)
    effective = weight * (DECAY_FACTOR ** time_delta_hours)
    return round(effective, 2)

def nearest_rank(counter: Dict[int, float], smooth_history: Optional[List[Dict[int, float]]] = None) -> List[str]:
    """Weighted nearest-rank percentiles with optional smoothing"""
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
            # Apply moving average smoothing
            vals = [pval]
            for hist in smooth_history[-2:]:  # prev 2
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
    """Process stream with all advanced features"""
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
            # Update latest timestamp
            for r in obj.get("records", []):
                if "timestamp" in r:
                    if state.latest_timestamp is None or r["timestamp"] > state.latest_timestamp:
                        state.latest_timestamp = r["timestamp"]
            
            # Store contributions
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
                # Continue from next line
                i += 1
                continue
                
        elif t == "adjust_window":
            new_size = obj.get("new_window_size", state.window_size)
            state.window_size = max(3, min(20, new_size))
        
        # Compute window
        ws = max(1, ingest_idx - state.window_size + 1)
        we = ingest_idx
        idxs = range(ws, we + 1)
        
        # Aggregate with time decay
        by_key = defaultdict(lambda: Counter())
        
        for j in idxs:
            if state.typ.get(j) == "data":
                for k, cnt in state.contrib.get(j, {}).items():
                    by_key[k].update(cnt)
        
        # Apply retractions
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
        
        # Conditional aggregations
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
        
        # Output rows
        base = [str(ingest_idx), str(ws), str(we), str(state.window_size)]
        
        # GLOBAL
        if sum(global_counter.values()) >= MIN_WEIGHT:
            rows.append(base + ["GLOBAL", ""] + nearest_rank(global_counter, state.prev_percentiles))
        
        # HIGH_WEIGHT
        if sum(high_weight.values()) >= MIN_WEIGHT:
            rows.append(base + ["HIGH_WEIGHT", ""] + nearest_rank(high_weight))
        
        # LOW_WEIGHT
        if sum(low_weight.values()) >= MIN_WEIGHT:
            rows.append(base + ["LOW_WEIGHT", ""] + nearest_rank(low_weight))
        
        # POSITIVE
        if sum(positive.values()) >= MIN_WEIGHT:
            rows.append(base + ["POSITIVE", ""] + nearest_rank(positive))
        
        # NEGATIVE
        if sum(negative.values()) >= MIN_WEIGHT:
            rows.append(base + ["NEGATIVE", ""] + nearest_rank(negative))
        
        # Per-KEY
        for k in sorted(k for k, c in by_key.items() if sum(c.values()) >= MIN_WEIGHT):
            rows.append(base + ["KEY", k] + nearest_rank(by_key[k]))
        
        # Store for smoothing
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
