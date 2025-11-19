#!/usr/bin/env python3
"""
Advanced grader for Real-Time Feature Store Pipeline task.
Recomputes expected output with all features and validates against submission.
Uses binary all_correct scoring: 1.0 if perfect match, 0.0 otherwise.
"""
import json, csv, os, math, heapq
from pathlib import Path
from collections import defaultdict, deque
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

# Constants matching task specification
MAX_LATE_ARRIVAL_MS = 3600000
CACHE_TTL_MS = 300000
MAX_ENTITIES_IN_MEMORY = 10000
QUANTILE_ERROR = 0.01
STDDEV_DDOF = 1
AGGREGATION_BUCKET_SIZE_MS = 60000

# Window sizes in milliseconds
WINDOWS = {
    "1h": 3600000,
    "24h": 86400000,
    "7d": 604800000,
    "30d": 2592000000,
    "all_time": float('inf')
}

WORKDIR = Path("/workdir")
STREAM = WORKDIR / "data" / "event_stream.jsonl"
SOL = WORKDIR / "sol.csv"

COLS = ["stream_id", "entity_type", "entity_id", "feature_name", "feature_value", 
        "cache_hit", "computation_time_ms", "num_events", "window_start_ts", 
        "window_end_ts", "is_stable"]

class TDigest:
    """Simplified t-digest for approximate quantile computation"""
    def __init__(self, delta=100):
        self.delta = delta
        self.centroids = []  # [(mean, weight), ...]
        self.count = 0
        
    def add(self, value, weight=1):
        self.centroids.append((float(value), weight))
        self.count += weight
        if len(self.centroids) > self.delta * 2:
            self._compress()
    
    def _compress(self):
        if not self.centroids:
            return
        # Sort by mean
        self.centroids.sort(key=lambda x: x[0])
        compressed = []
        current_mean, current_weight = self.centroids[0]
        
        for mean, weight in self.centroids[1:]:
            # Merge if weight allows
            if current_weight + weight <= 2 * self.count / self.delta:
                total_weight = current_weight + weight
                current_mean = (current_mean * current_weight + mean * weight) / total_weight
                current_weight = total_weight
            else:
                compressed.append((current_mean, current_weight))
                current_mean, current_weight = mean, weight
        
        compressed.append((current_mean, current_weight))
        self.centroids = compressed
    
    def quantile(self, q):
        if not self.centroids:
            return None
        if len(self.centroids) == 1:
            return self.centroids[0][0]
        
        self._compress()
        self.centroids.sort(key=lambda x: x[0])
        
        target = q * self.count
        cumulative = 0
        
        for i, (mean, weight) in enumerate(self.centroids):
            if cumulative + weight >= target:
                if i == 0:
                    return mean
                # Interpolate
                prev_mean, prev_weight = self.centroids[i-1]
                fraction = (target - cumulative) / weight
                return prev_mean + fraction * (mean - prev_mean)
            cumulative += weight
        
        return self.centroids[-1][0]
    
    @staticmethod
    def merge(digests):
        """Merge multiple t-digests"""
        merged = TDigest()
        for digest in digests:
            for mean, weight in digest.centroids:
                merged.add(mean, weight)
        return merged

class TimeBucket:
    """Stores aggregates for a time bucket"""
    def __init__(self, timestamp_bucket):
        self.timestamp_bucket = timestamp_bucket
        self.sum = 0.0
        self.count = 0
        self.min = float('inf')
        self.max = float('-inf')
        self.values = []  # For stddev
        self.last_value = None
        self.last_timestamp = 0
        self.digest = TDigest()
        
    def add(self, value, timestamp):
        if not isinstance(value, (int, float)):
            # Categorical - store as last
            self.last_value = value
            self.last_timestamp = timestamp
            self.count += 1
            return
        
        self.sum += value
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.values.append(value)
        self.digest.add(value)
        self.last_value = value
        self.last_timestamp = timestamp

class EntityState:
    """State for a single entity"""
    def __init__(self, entity_type, entity_id):
        self.entity_type = entity_type
        self.entity_id = entity_id
        # Map: (attribute_name, timestamp_bucket) -> TimeBucket
        self.buckets: Dict[Tuple[str, int], TimeBucket] = {}
        self.last_access = 0
        
    def add_update(self, timestamp, attributes):
        bucket_ts = (timestamp // AGGREGATION_BUCKET_SIZE_MS) * AGGREGATION_BUCKET_SIZE_MS
        
        for attr, value in attributes.items():
            key = (attr, bucket_ts)
            if key not in self.buckets:
                self.buckets[key] = TimeBucket(bucket_ts)
            self.buckets[key].add(value, timestamp)
        
        self.last_access = timestamp
    
    def get_buckets_in_window(self, attr, window_start, window_end):
        """Get all buckets for attribute in time window"""
        buckets = []
        for (bucket_attr, bucket_ts), bucket in self.buckets.items():
            if bucket_attr == attr and window_start <= bucket_ts <= window_end:
                buckets.append(bucket)
        return sorted(buckets, key=lambda b: b.timestamp_bucket)
    
    def prune_old_buckets(self, min_timestamp):
        """Remove buckets older than threshold"""
        to_remove = []
        for key, bucket in self.buckets.items():
            attr, bucket_ts = key
            if bucket_ts < min_timestamp:
                to_remove.append(key)
        for key in to_remove:
            del self.buckets[key]

class CacheEntry:
    """Cache entry with metadata"""
    def __init__(self, value, computed_at, num_events, window_start, window_end, stable):
        self.value = value
        self.computed_at = computed_at
        self.num_events = num_events
        self.window_start = window_start
        self.window_end = window_end
        self.stable = stable

class FeatureStore:
    """Main feature store implementation"""
    def __init__(self):
        # Entity states
        self.entities: Dict[Tuple[str, str], EntityState] = {}
        
        # Feature definitions: feature_name -> {entity_type, aggregation, source_attribute, window, version}
        self.features: Dict[str, Dict] = {}
        
        # Cache: (entity_type, entity_id, feature_name, asof_bucket) -> CacheEntry
        self.cache: Dict[Tuple[str, str, str, int], CacheEntry] = {}
        
        # Watermarks: entity_type -> timestamp
        self.watermarks: Dict[str, int] = {}
        
        # Current processing timestamp
        self.current_time = 0
        
    def register_feature(self, feature_name, entity_type, aggregation, source_attribute, window, version=1):
        self.features[feature_name] = {
            'entity_type': entity_type,
            'aggregation': aggregation,
            'source_attribute': source_attribute,
            'window': window,
            'version': version
        }
    
    def get_entity(self, entity_type, entity_id):
        key = (entity_type, entity_id)
        if key not in self.entities:
            self.entities[key] = EntityState(entity_type, entity_id)
        return self.entities[key]
    
    def add_update(self, entity_type, entity_id, timestamp, attributes):
        entity = self.get_entity(entity_type, entity_id)
        entity.add_update(timestamp, attributes)
        self.current_time = max(self.current_time, timestamp)
        
        # Invalidate affected cache entries
        self._invalidate_cache_for_update(entity_type, entity_id, timestamp)
    
    def add_backfill(self, entity_type, entity_id, timestamp, attributes):
        # Same as update but marks as correction
        self.add_update(entity_type, entity_id, timestamp, attributes)
    
    def set_watermark(self, entity_type, watermark_timestamp):
        self.watermarks[entity_type] = watermark_timestamp
        # Mark old cache entries as stable
        for key, entry in self.cache.items():
            etype, eid, fname, asof_bucket = key
            if etype == entity_type and entry.window_end <= watermark_timestamp:
                entry.stable = True
    
    def _invalidate_cache_for_update(self, entity_type, entity_id, timestamp):
        """Invalidate cache entries affected by update"""
        to_remove = []
        for key, entry in self.cache.items():
            etype, eid, fname, asof_bucket = key
            if etype == entity_type and eid == entity_id:
                # Invalidate if update affects window
                if entry.window_start <= timestamp <= entry.window_end:
                    to_remove.append(key)
        for key in to_remove:
            del self.cache[key]
    
    def invalidate_cache(self, entity_type=None, entity_id=None):
        """Invalidate cache entries matching criteria"""
        if entity_type is None and entity_id is None:
            self.cache.clear()
        else:
            to_remove = []
            for key in self.cache:
                etype, eid, fname, asof_bucket = key
                if entity_type and etype != entity_type:
                    continue
                if entity_id and eid != entity_id:
                    continue
                to_remove.append(key)
            for key in to_remove:
                del self.cache[key]
    
    def compute_feature(self, entity_type, entity_id, feature_name, asof_timestamp):
        """Compute feature value at point-in-time"""
        # Check if feature exists
        if feature_name not in self.features:
            return None, 0, None, None
        
        feature_def = self.features[feature_name]
        if feature_def['entity_type'] != entity_type:
            return None, 0, None, None
        
        # Get entity
        key = (entity_type, entity_id)
        if key not in self.entities:
            # Cold start
            if feature_def['aggregation'] in ['sum', 'count']:
                return 0, 0, None, None
            else:
                return None, 0, None, None
        
        entity = self.entities[key]
        
        # Determine window
        window_size = WINDOWS[feature_def['window']]
        if window_size == float('inf'):
            window_start = 0
        else:
            window_start = asof_timestamp - window_size
        window_end = asof_timestamp
        
        # Get buckets in window
        buckets = entity.get_buckets_in_window(feature_def['source_attribute'], window_start, window_end)
        
        # Compute aggregation
        value = self._compute_aggregation(
            buckets, 
            feature_def['aggregation'],
            window_start,
            window_end
        )
        
        num_events = sum(b.count for b in buckets)
        
        return value, num_events, window_start, window_end
    
    def _compute_aggregation(self, buckets, aggregation, window_start, window_end):
        """Compute aggregation over buckets"""
        if not buckets:
            if aggregation in ['sum', 'count']:
                return 0
            return None
        
        if aggregation == 'sum':
            return sum(b.sum for b in buckets)
        
        elif aggregation == 'count':
            return sum(b.count for b in buckets)
        
        elif aggregation == 'avg':
            total_sum = sum(b.sum for b in buckets)
            total_count = sum(b.count for b in buckets)
            return total_sum / total_count if total_count > 0 else None
        
        elif aggregation == 'min':
            mins = [b.min for b in buckets if b.min != float('inf')]
            return min(mins) if mins else None
        
        elif aggregation == 'max':
            maxs = [b.max for b in buckets if b.max != float('-inf')]
            return max(maxs) if maxs else None
        
        elif aggregation == 'last':
            # Find bucket with latest timestamp
            latest_bucket = max(buckets, key=lambda b: b.last_timestamp)
            return latest_bucket.last_value
        
        elif aggregation == 'stddev':
            # Collect all values
            all_values = []
            for b in buckets:
                all_values.extend(b.values)
            
            if len(all_values) <= STDDEV_DDOF:
                return None
            
            mean = sum(all_values) / len(all_values)
            variance = sum((x - mean) ** 2 for x in all_values) / (len(all_values) - STDDEV_DDOF)
            return math.sqrt(variance)
        
        elif aggregation == 'p50':
            # Merge t-digests
            digest = TDigest.merge([b.digest for b in buckets])
            result = digest.quantile(0.5)
            return result
        
        elif aggregation == 'p95':
            digest = TDigest.merge([b.digest for b in buckets])
            result = digest.quantile(0.95)
            return result
        
        return None
    
    def request_feature(self, stream_id, entity_type, entity_id, feature_name, 
                       request_timestamp, asof_timestamp):
        """Handle feature request with caching"""
        # Check cache
        asof_bucket = asof_timestamp // CACHE_TTL_MS
        cache_key = (entity_type, entity_id, feature_name, asof_bucket)
        
        is_stable = entity_type in self.watermarks and asof_timestamp <= self.watermarks[entity_type]
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            # Use cache
            return {
                'stream_id': stream_id,
                'entity_type': entity_type,
                'entity_id': entity_id,
                'feature_name': feature_name,
                'feature_value': entry.value,
                'cache_hit': True,
                'computation_time_ms': 0,
                'num_events': entry.num_events,
                'window_start_ts': entry.window_start,
                'window_end_ts': entry.window_end,
                'is_stable': entry.stable
            }
        
        # Compute feature
        value, num_events, window_start, window_end = self.compute_feature(
            entity_type, entity_id, feature_name, asof_timestamp
        )
        
        # Store in cache
        entry = CacheEntry(value, request_timestamp, num_events, window_start, window_end, is_stable)
        self.cache[cache_key] = entry
        
        return {
            'stream_id': stream_id,
            'entity_type': entity_type,
            'entity_id': entity_id,
            'feature_name': feature_name,
            'feature_value': value,
            'cache_hit': False,
            'computation_time_ms': 0,
            'num_events': num_events,
            'window_start_ts': window_start,
            'window_end_ts': window_end,
            'is_stable': is_stable
        }

def load_stream(stream_path: Path) -> List[dict]:
    """Load and normalize stream entries"""
    entries = []
    if not stream_path.exists():
        return entries
    
    next_id = 1
    
    for raw in stream_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        
        try:
            obj = json.loads(s)
        except:
            continue
        
        if not isinstance(obj, dict):
            continue
        
        # Auto-assign stream_id if missing
        if not isinstance(obj.get("stream_id"), int):
            obj["stream_id"] = next_id
        
        next_id = max(next_id, obj["stream_id"]) + 1
        
        entries.append(obj)
    
    entries.sort(key=lambda x: x.get("stream_id", 0))
    return entries

def process_stream(entries: List[dict]) -> List[Dict]:
    """Process complete stream with feature store"""
    store = FeatureStore()
    results = []
    
    for entry in entries:
        entry_type = entry.get("type")
        stream_id = entry.get("stream_id")
        
        if entry_type == "entity_update":
            entity_type = entry.get("entity_type")
            entity_id = entry.get("entity_id")
            timestamp = entry.get("timestamp")
            attributes = entry.get("attributes", {})
            
            store.add_update(entity_type, entity_id, timestamp, attributes)
        
        elif entry_type == "feature_definition":
            feature_name = entry.get("feature_name")
            entity_type = entry.get("entity_type")
            aggregation = entry.get("aggregation")
            source_attribute = entry.get("source_attribute")
            window = entry.get("window")
            version = entry.get("version", 1)
            
            store.register_feature(feature_name, entity_type, aggregation, 
                                 source_attribute, window, version)
        
        elif entry_type == "feature_request":
            entity_type = entry.get("entity_type")
            entity_id = entry.get("entity_id")
            request_timestamp = entry.get("request_timestamp")
            features = entry.get("features", [])
            asof_timestamp = entry.get("asof_timestamp", request_timestamp)
            
            # Process each requested feature
            for feature_name in features:
                result = store.request_feature(
                    stream_id, entity_type, entity_id, feature_name,
                    request_timestamp, asof_timestamp
                )
                results.append(result)
        
        elif entry_type == "backfill":
            entity_type = entry.get("entity_type")
            entity_id = entry.get("entity_id")
            timestamp = entry.get("timestamp")
            attributes = entry.get("attributes", {})
            
            store.add_backfill(entity_type, entity_id, timestamp, attributes)
        
        elif entry_type == "watermark":
            entity_type = entry.get("entity_type")
            watermark_timestamp = entry.get("watermark_timestamp")
            
            store.set_watermark(entity_type, watermark_timestamp)
        
        elif entry_type == "cache_control":
            action = entry.get("action")
            entity_type = entry.get("entity_type")
            entity_id = entry.get("entity_id")
            
            if action in ["invalidate", "clear"]:
                store.invalidate_cache(entity_type, entity_id)
    
    return results

def format_value(value):
    """Format value for CSV output"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)

def results_to_rows(results: List[Dict]) -> List[List[str]]:
    """Convert results to CSV rows"""
    rows = []
    for result in results:
        row = [
            str(result['stream_id']),
            result['entity_type'],
            result['entity_id'],
            result['feature_name'],
            format_value(result['feature_value']),
            "true" if result['cache_hit'] else "false",
            str(result['computation_time_ms']),
            str(result['num_events']),
            format_value(result['window_start_ts']),
            format_value(result['window_end_ts']),
            "true" if result['is_stable'] else "false"
        ]
        rows.append(row)
    return rows

def read_solution(sol_path: Path) -> Optional[list]:
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
    Grade by recomputing expected output and comparing with submission.
    Uses binary all_correct scoring: 1.0 if perfect match, 0.0 otherwise.
    """
    subscores = {"all_correct": 0.0}
    weights = {"all_correct": 1.0}
    
    # Load stream
    entries = load_stream(STREAM)
    if not entries:
        return GradingResult(
            score=0.0,
            feedback="No valid entries in event_stream.jsonl",
            subscores=subscores,
            weights=weights
        )
    
    # Compute expected output
    expected_results = process_stream(entries)
    expected_rows = results_to_rows(expected_results)
    
    # Load solution
    solution_rows = read_solution(SOL)
    if solution_rows is None:
        return GradingResult(
            score=0.0,
            feedback="Missing or invalid /workdir/sol.csv",
            subscores=subscores,
            weights=weights
        )
    
    # Check row count
    expected_count = len(expected_rows)
    if len(solution_rows) != expected_count:
        return GradingResult(
            score=0.0,
            feedback=f"Row count mismatch. Expected {expected_count}, got {len(solution_rows)}",
            subscores=subscores,
            weights=weights
        )
    
    # Compare rows
    mismatches = []
    float_cols = {"feature_value"}
    int_cols = {"computation_time_ms", "num_events", "window_start_ts", "window_end_ts"}
    
    for i, (exp_row, sol_row) in enumerate(zip(expected_rows, solution_rows)):
        if len(sol_row) != len(COLS):
            mismatches.append({
                "row": i + 1,
                "error": f"Column count mismatch: expected {len(COLS)}, got {len(sol_row)}"
            })
            continue
        
        for j, col in enumerate(COLS):
            exp_val = exp_row[j]
            sol_val = sol_row[j]
            
            if col in float_cols:
                # Special handling for null
                if exp_val == "null" or sol_val == "null":
                    if exp_val != sol_val:
                        mismatches.append({
                            "row": i + 1,
                            "column": col,
                            "expected": exp_val,
                            "got": sol_val
                        })
                else:
                    # Floating point comparison with tolerance
                    try:
                        exp_float = float(exp_val)
                        sol_float = float(sol_val)
                        # 1% tolerance for quantiles, 0.1% for others
                        tolerance = 0.01 if 'p50' in str(exp_row[3]) or 'p95' in str(exp_row[3]) else 0.001
                        if abs(exp_float - sol_float) > tolerance * max(abs(exp_float), 1.0):
                            mismatches.append({
                                "row": i + 1,
                                "column": col,
                                "expected": exp_val,
                                "got": sol_val
                            })
                    except ValueError:
                        mismatches.append({
                            "row": i + 1,
                            "column": col,
                            "error": "Invalid float format"
                        })
            elif col in int_cols:
                # Integer comparison or null
                if exp_val == "null" or sol_val == "null":
                    if exp_val != sol_val:
                        mismatches.append({
                            "row": i + 1,
                            "column": col,
                            "expected": exp_val,
                            "got": sol_val
                        })
                elif exp_val != sol_val:
                    mismatches.append({
                        "row": i + 1,
                        "column": col,
                        "expected": exp_val,
                        "got": sol_val
                    })
            else:
                if exp_val != sol_val:
                    mismatches.append({
                        "row": i + 1,
                        "column": col,
                        "expected": exp_val,
                        "got": sol_val
                    })
        
        if len(mismatches) >= 50:
            break
    
    # Binary scoring: all correct or nothing
    if mismatches:
        subscores["all_correct"] = 0.0
        score = 0.0
        
        return GradingResult(
            score=score,
            feedback=f"Found {len(mismatches)} mismatches. Binary scoring: all must be correct to pass.",
            subscores=subscores,
            weights=weights,
            details={"mismatches": mismatches[:25]}
        )
    
    # Perfect match - award full score
    subscores["all_correct"] = 1.0
    score = 1.0
    
    return GradingResult(
        score=score,
        feedback="All checks passed! Feature store pipeline implemented correctly.",
        subscores=subscores,
        weights=weights
    )

if __name__ == "__main__":
    result = grade(None)
    print(result.model_dump() if hasattr(result, "model_dump") else result.__dict__)