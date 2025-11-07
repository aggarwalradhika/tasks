#!/usr/bin/env python3
"""
Advanced grader for Adaptive Anomaly Detection task.
Recomputes expected output with all features and validates against submission.
"""
import json, csv, os, math, random
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
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
INITIAL_WINDOW_SIZE = 100
MIN_BASELINE_SAMPLES = 30
ANOMALY_THRESHOLD = 0.75
MEMORY_BUDGET = 1000
DRIFT_DETECTION_WINDOW = 50
ENSEMBLE_UPDATE_FREQ = 20
CONFIDENCE_DECAY = 0.95
EPSILON = 0.001
RANDOM_SEED = 42

WORKDIR = Path("/workdir")
STREAM = WORKDIR / "data" / "sensor_stream.jsonl"
SOL = WORKDIR / "sol.csv"

COLS = ["stream_id", "anomaly_score", "is_anomaly", "confidence", 
        "strategy_scores", "ensemble_weights", "drift_detected", "adaptation_triggered"]

class DetectionState:
    """Complete state for anomaly detection system"""
    def __init__(self):
        # Configuration
        self.window_size = INITIAL_WINDOW_SIZE
        self.anomaly_threshold = ANOMALY_THRESHOLD
        self.sensitivity = 1.0
        
        # Statistics per sensor
        self.sensor_means: Dict[str, float] = {}
        self.sensor_stds: Dict[str, float] = {}
        self.sensor_mins: Dict[str, float] = {}
        self.sensor_maxs: Dict[str, float] = {}
        self.sensor_counts: Dict[str, int] = {}
        
        # Categorical frequency tracking
        self.categorical_freqs: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        
        # Reservoir sample for distance/isolation strategies
        self.reservoir: List[Dict[str, Any]] = []
        self.reservoir_size = MEMORY_BUDGET // 3
        
        # Ensemble weights
        self.ensemble_weights = {"stat": 0.4, "dist": 0.3, "iso": 0.3}
        
        # Drift detection
        self.recent_window: deque = deque(maxlen=25)
        self.reference_window: deque = deque(maxlen=25)
        self.drift_detected_count = 0
        self.adaptation_active = False
        
        # Labeled observations
        self.labeled_anomalies: set = set()
        self.labeled_normal: set = set()
        
        # History
        self.observation_count = 0
        self.observations_history: Dict[int, Dict] = {}
        
        # Isolation forest state
        self.iso_trees: List = []
        self.iso_rebuild_counter = 0
        
        # Performance tracking for ensemble
        self.strategy_correct: Dict[str, int] = {"stat": 0, "dist": 0, "iso": 0}
        self.strategy_total: Dict[str, int] = {"stat": 0, "dist": 0, "iso": 0}
        
    def clone(self):
        """Deep copy for checkpointing"""
        new_state = DetectionState()
        new_state.window_size = self.window_size
        new_state.anomaly_threshold = self.anomaly_threshold
        new_state.sensitivity = self.sensitivity
        new_state.sensor_means = deepcopy(self.sensor_means)
        new_state.sensor_stds = deepcopy(self.sensor_stds)
        new_state.sensor_mins = deepcopy(self.sensor_mins)
        new_state.sensor_maxs = deepcopy(self.sensor_maxs)
        new_state.sensor_counts = deepcopy(self.sensor_counts)
        new_state.categorical_freqs = deepcopy(self.categorical_freqs)
        new_state.reservoir = deepcopy(self.reservoir)
        new_state.ensemble_weights = deepcopy(self.ensemble_weights)
        new_state.recent_window = deepcopy(self.recent_window)
        new_state.reference_window = deepcopy(self.reference_window)
        new_state.drift_detected_count = self.drift_detected_count
        new_state.adaptation_active = self.adaptation_active
        new_state.labeled_anomalies = deepcopy(self.labeled_anomalies)
        new_state.labeled_normal = deepcopy(self.labeled_normal)
        new_state.observation_count = self.observation_count
        new_state.observations_history = deepcopy(self.observations_history)
        new_state.iso_trees = deepcopy(self.iso_trees)
        new_state.iso_rebuild_counter = self.iso_rebuild_counter
        new_state.strategy_correct = deepcopy(self.strategy_correct)
        new_state.strategy_total = deepcopy(self.strategy_total)
        return new_state

def load_stream(stream_path: Path) -> List[dict]:
    """Load and normalize stream entries"""
    entries = []
    if not stream_path.exists():
        return entries
    
    next_id = 1
    last_timestamp = 0
    
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
        
        # Handle missing timestamps for observations
        if obj.get("type") == "observation":
            if "timestamp" not in obj or not isinstance(obj["timestamp"], int):
                obj["timestamp"] = last_timestamp + 1000
            last_timestamp = obj["timestamp"]
        
        entries.append(obj)
    
    entries.sort(key=lambda x: x.get("stream_id", 0))
    return entries

def update_statistics(state: DetectionState, sensors: Dict[str, Any]):
    """Update running statistics with exponential moving average"""
    for sensor, value in sensors.items():
        if isinstance(value, (int, float)):
            if sensor not in state.sensor_means:
                state.sensor_means[sensor] = float(value)
                state.sensor_stds[sensor] = 0.0
                state.sensor_mins[sensor] = float(value)
                state.sensor_maxs[sensor] = float(value)
                state.sensor_counts[sensor] = 1
            else:
                # Exponential moving average
                alpha = 1.0 / (state.sensor_counts[sensor] + 1)
                old_mean = state.sensor_means[sensor]
                state.sensor_means[sensor] = (1 - alpha) * old_mean + alpha * value
                
                # Update std using Welford's online algorithm
                delta = value - old_mean
                delta2 = value - state.sensor_means[sensor]
                state.sensor_stds[sensor] = math.sqrt(
                    max(0, (state.sensor_stds[sensor] ** 2 * (state.sensor_counts[sensor] - 1) + 
                            delta * delta2) / state.sensor_counts[sensor])
                )
                
                state.sensor_mins[sensor] = min(state.sensor_mins[sensor], value)
                state.sensor_maxs[sensor] = max(state.sensor_maxs[sensor], value)
                state.sensor_counts[sensor] += 1
        else:
            # Categorical
            state.categorical_freqs[sensor][value] += 1

def statistical_score(state: DetectionState, sensors: Dict[str, Any]) -> float:
    """Z-score based anomaly scoring"""
    if state.observation_count < MIN_BASELINE_SAMPLES:
        return 0.0
    
    scores = []
    for sensor, value in sensors.items():
        if isinstance(value, (int, float)) and sensor in state.sensor_means:
            mean = state.sensor_means[sensor]
            std = state.sensor_stds[sensor]
            z_score = abs(value - mean) / (std + EPSILON)
            # Sigmoid normalization
            score = 1.0 / (1.0 + math.exp(-2 * (z_score - 2)))
            scores.append(score)
        elif sensor in state.categorical_freqs:
            # Frequency-based surprise
            freq = state.categorical_freqs[sensor].get(value, 0)
            total = sum(state.categorical_freqs[sensor].values())
            if total > 0:
                prob = freq / total
                surprise = 1.0 - prob
                scores.append(surprise)
    
    return sum(scores) / len(scores) if scores else 0.0

def euclidean_distance(obs1: Dict[str, Any], obs2: Dict[str, Any], 
                       state: DetectionState) -> float:
    """Compute normalized Euclidean distance"""
    dist_sq = 0.0
    count = 0
    
    all_sensors = set(obs1.keys()) | set(obs2.keys())
    for sensor in all_sensors:
        if sensor not in obs1 or sensor not in obs2:
            # Use last known value or distance penalty
            dist_sq += 1.0
            count += 1
            continue
        
        val1, val2 = obs1[sensor], obs2[sensor]
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Normalize by range
            range_val = state.sensor_maxs.get(sensor, 1.0) - state.sensor_mins.get(sensor, 0.0)
            if range_val > EPSILON:
                norm_dist = abs(val1 - val2) / range_val
            else:
                norm_dist = 0.0
            dist_sq += norm_dist ** 2
            count += 1
        else:
            # Categorical: 0 if match, 1 if different
            dist_sq += 0.0 if val1 == val2 else 1.0
            count += 1
    
    return math.sqrt(dist_sq / count) if count > 0 else 0.0

def distance_score(state: DetectionState, sensors: Dict[str, Any]) -> float:
    """k-NN based anomaly scoring"""
    if len(state.reservoir) < 5:
        return 0.0
    
    # Compute distances to all reservoir samples
    distances = []
    for res_obs in state.reservoir:
        dist = euclidean_distance(sensors, res_obs, state)
        distances.append(dist)
    
    # k-nearest neighbors (k=5)
    distances.sort()
    k = min(5, len(distances))
    avg_dist = sum(distances[:k]) / k
    
    # Normalize by max distance seen (approximate with 95th percentile)
    distances_sorted = sorted(distances)
    max_dist_idx = int(0.95 * len(distances_sorted))
    max_dist = distances_sorted[max_dist_idx] if max_dist_idx < len(distances_sorted) else 1.0
    
    return min(1.0, avg_dist / (max_dist + EPSILON))

def isolation_score(state: DetectionState, sensors: Dict[str, Any]) -> float:
    """Simplified isolation forest scoring"""
    if len(state.iso_trees) == 0 or state.observation_count < MIN_BASELINE_SAMPLES:
        return 0.0
    
    # Average path length across trees
    path_lengths = []
    for tree in state.iso_trees:
        path_len = compute_path_length(tree, sensors, 0)
        path_lengths.append(path_len)
    
    avg_path = sum(path_lengths) / len(path_lengths) if path_lengths else 8.0
    
    # Expected path length for normal data (approximate)
    n = max(len(state.reservoir), 10)
    expected_path = 2 * (math.log(n - 1) + 0.5772) - 2 * (n - 1) / n
    
    # Normalize
    score = avg_path / (expected_path + EPSILON)
    return min(1.0, max(0.0, 1.0 - score))  # Invert: shorter path = more anomalous

def compute_path_length(tree, obs, depth):
    """Traverse isolation tree and return path length"""
    if tree is None or depth > 8:
        return depth
    
    if "leaf" in tree:
        return depth
    
    sensor = tree["split_sensor"]
    split_val = tree["split_value"]
    
    if sensor not in obs:
        return depth + 1
    
    obs_val = obs[sensor]
    
    if isinstance(obs_val, (int, float)) and isinstance(split_val, (int, float)):
        if obs_val < split_val:
            return compute_path_length(tree.get("left"), obs, depth + 1)
        else:
            return compute_path_length(tree.get("right"), obs, depth + 1)
    else:
        # Categorical
        if obs_val == split_val:
            return compute_path_length(tree.get("left"), obs, depth + 1)
        else:
            return compute_path_length(tree.get("right"), obs, depth + 1)

def build_isolation_trees(samples: List[Dict], num_trees=3, max_depth=8):
    """Build simplified isolation trees"""
    random.seed(RANDOM_SEED)
    trees = []
    
    if len(samples) < 10:
        return trees
    
    for _ in range(num_trees):
        tree = build_iso_tree(samples, 0, max_depth)
        trees.append(tree)
    
    return trees

def build_iso_tree(samples, depth, max_depth):
    """Recursively build isolation tree"""
    if depth >= max_depth or len(samples) <= 1:
        return {"leaf": True, "size": len(samples)}
    
    # Random sensor and split
    if len(samples) == 0:
        return {"leaf": True, "size": 0}
    
    sensors = list(samples[0].keys())
    if not sensors:
        return {"leaf": True, "size": len(samples)}
    
    sensor = random.choice(sensors)
    
    # Get split value
    values = [s.get(sensor) for s in samples if sensor in s]
    if not values:
        return {"leaf": True, "size": len(samples)}
    
    if isinstance(values[0], (int, float)):
        split_val = random.uniform(min(values), max(values))
        left = [s for s in samples if sensor in s and s[sensor] < split_val]
        right = [s for s in samples if sensor in s and s[sensor] >= split_val]
    else:
        split_val = random.choice(values)
        left = [s for s in samples if sensor in s and s[sensor] == split_val]
        right = [s for s in samples if sensor in s and s[sensor] != split_val]
    
    if len(left) == 0 or len(right) == 0:
        return {"leaf": True, "size": len(samples)}
    
    return {
        "split_sensor": sensor,
        "split_value": split_val,
        "left": build_iso_tree(left, depth + 1, max_depth),
        "right": build_iso_tree(right, depth + 1, max_depth)
    }

def update_reservoir(state: DetectionState, sensors: Dict[str, Any], stream_id: int):
    """Reservoir sampling to maintain representative sample"""
    state.observation_count += 1
    
    if stream_id in state.labeled_anomalies:
        # Don't add labeled anomalies to reservoir
        return
    
    if len(state.reservoir) < state.reservoir_size:
        state.reservoir.append(sensors.copy())
    else:
        # Reservoir sampling with priority for recent and labeled normal
        priority_prob = 2.0 if stream_id in state.labeled_normal else 1.0
        prob = (state.reservoir_size * priority_prob) / state.observation_count
        
        if random.random() < prob:
            replace_idx = random.randint(0, len(state.reservoir) - 1)
            state.reservoir[replace_idx] = sensors.copy()

def detect_drift(state: DetectionState, sensors: Dict[str, Any]) -> bool:
    """ADWIN-style drift detection"""
    if state.observation_count < MIN_BASELINE_SAMPLES:
        return False
    
    # Add to recent window
    state.recent_window.append(sensors)
    
    # Move to reference when full
    if len(state.recent_window) == 25:
        if len(state.reference_window) > 0:
            # Check for drift
            drift = False
            for sensor in sensors.keys():
                if not isinstance(sensors.get(sensor), (int, float)):
                    continue
                
                recent_vals = [obs.get(sensor) for obs in state.recent_window 
                              if sensor in obs and isinstance(obs[sensor], (int, float))]
                ref_vals = [obs.get(sensor) for obs in state.reference_window 
                           if sensor in obs and isinstance(obs[sensor], (int, float))]
                
                if len(recent_vals) < 10 or len(ref_vals) < 10:
                    continue
                
                mean_recent = sum(recent_vals) / len(recent_vals)
                mean_ref = sum(ref_vals) / len(ref_vals)
                
                var_recent = sum((x - mean_recent) ** 2 for x in recent_vals) / len(recent_vals)
                var_ref = sum((x - mean_ref) ** 2 for x in ref_vals) / len(ref_vals)
                
                threshold = 2 * math.sqrt((var_recent + var_ref) / 25)
                
                if abs(mean_recent - mean_ref) > threshold:
                    drift = True
                    break
            
            if drift:
                return True
        
        # Move recent to reference
        state.reference_window = state.recent_window.copy()
        state.recent_window.clear()
    
    return False

def apply_adaptation(state: DetectionState, affected_sensors: List[str] = None, 
                     severity: float = 0.5):
    """Apply adaptation actions when drift detected"""
    if affected_sensors:
        for sensor in affected_sensors:
            if sensor in state.sensor_means:
                # Reset statistics for affected sensors
                del state.sensor_means[sensor]
                del state.sensor_stds[sensor]
                del state.sensor_mins[sensor]
                del state.sensor_maxs[sensor]
                del state.sensor_counts[sensor]
    
    # Clear reservoir of old data
    state.reservoir = state.reservoir[-state.reservoir_size // 2:]
    
    # Temporarily lower threshold
    state.anomaly_threshold = ANOMALY_THRESHOLD * (1 - 0.2 * severity)
    state.adaptation_active = True
    
    # Boost statistical strategy weight temporarily
    total = sum(state.ensemble_weights.values())
    state.ensemble_weights["stat"] = 0.5 * total
    state.ensemble_weights["dist"] = 0.25 * total
    state.ensemble_weights["iso"] = 0.25 * total

def update_ensemble_weights(state: DetectionState):
    """Update ensemble weights based on performance"""
    if state.observation_count % ENSEMBLE_UPDATE_FREQ != 0:
        return
    
    # Calculate accuracy for each strategy
    for strategy in ["stat", "dist", "iso"]:
        total = state.strategy_total[strategy]
        if total > 0:
            accuracy = state.strategy_correct[strategy] / total
            # Adjust weight proportionally
            state.ensemble_weights[strategy] *= (0.8 + 0.4 * accuracy)
    
    # Normalize
    total_weight = sum(state.ensemble_weights.values())
    for strategy in state.ensemble_weights:
        state.ensemble_weights[strategy] /= total_weight
    
    # Gradually restore threshold if adaptation was active
    if state.adaptation_active and state.observation_count % 50 == 0:
        state.anomaly_threshold = min(ANOMALY_THRESHOLD, 
                                     state.anomaly_threshold * 1.05)
        if abs(state.anomaly_threshold - ANOMALY_THRESHOLD) < 0.01:
            state.adaptation_active = False

def process_stream(entries: List[dict]) -> List[List[str]]:
    """Process complete stream with anomaly detection"""
    random.seed(RANDOM_SEED)
    state = DetectionState()
    checkpoints: Dict[str, DetectionState] = {}
    rows = []
    
    for entry in entries:
        entry_type = entry.get("type")
        stream_id = entry.get("stream_id")
        
        if entry_type == "observation":
            sensors = entry.get("sensors", {})
            
            # Store observation
            state.observations_history[stream_id] = sensors.copy()
            
            # Detect drift
            drift_detected = detect_drift(state, sensors)
            if state.drift_detected_count > 0:
                drift_detected = True
                state.drift_detected_count -= 1
            
            adaptation_triggered = False
            
            # Update baseline statistics (if not in cold start or if normal)
            if stream_id not in state.labeled_anomalies:
                update_statistics(state, sensors)
                update_reservoir(state, sensors, stream_id)
            
            # Rebuild isolation trees periodically
            state.iso_rebuild_counter += 1
            if state.iso_rebuild_counter >= 200 and len(state.reservoir) > 20:
                state.iso_trees = build_isolation_trees(state.reservoir)
                state.iso_rebuild_counter = 0
            
            # Compute strategy scores
            stat_score = statistical_score(state, sensors)
            dist_score = distance_score(state, sensors)
            iso_score = isolation_score(state, sensors)
            
            # Ensemble score
            weights = state.ensemble_weights
            anomaly_score = (weights["stat"] * stat_score + 
                           weights["dist"] * dist_score + 
                           weights["iso"] * iso_score)
            
            # Apply sensitivity adjustment
            anomaly_score *= state.sensitivity
            anomaly_score = min(1.0, max(0.0, anomaly_score))
            
            # Determine if anomaly
            is_anomaly = anomaly_score >= state.anomaly_threshold
            
            # Cold start handling
            if state.observation_count < MIN_BASELINE_SAMPLES:
                is_anomaly = False
                anomaly_score = 0.0
            
            # Compute confidence
            samples_ratio = min(1.0, state.observation_count / MIN_BASELINE_SAMPLES)
            score_dist_from_threshold = abs(anomaly_score - state.anomaly_threshold)
            confidence = samples_ratio * (1 - score_dist_from_threshold / 
                                         (state.anomaly_threshold + EPSILON))
            confidence = min(1.0, max(0.0, confidence))
            
            # Format output
            strategy_scores = f"stat:{stat_score:.3f}:dist:{dist_score:.3f}:iso:{iso_score:.3f}"
            ensemble_weights = f"stat:{weights['stat']:.3f}:dist:{weights['dist']:.3f}:iso:{weights['iso']:.3f}"
            
            rows.append([
                str(stream_id),
                f"{anomaly_score:.3f}",
                "true" if is_anomaly else "false",
                f"{confidence:.3f}",
                strategy_scores,
                ensemble_weights,
                "true" if drift_detected else "false",
                "true" if adaptation_triggered else "false"
            ])
            
            # Update ensemble weights
            update_ensemble_weights(state)
            
        elif entry_type == "label":
            target_id = entry.get("target_stream_id")
            is_anomaly = entry.get("is_anomaly", False)
            label_confidence = entry.get("confidence", 1.0)
            
            if is_anomaly:
                state.labeled_anomalies.add(target_id)
                # Remove from reservoir if present
                if target_id in state.observations_history:
                    target_obs = state.observations_history[target_id]
                    state.reservoir = [obs for obs in state.reservoir 
                                     if obs != target_obs]
                # Increase threshold slightly
                state.anomaly_threshold *= (1.0 + 0.02 * label_confidence)
            else:
                state.labeled_normal.add(target_id)
                # Add to reservoir with priority
                if target_id in state.observations_history:
                    target_obs = state.observations_history[target_id]
                    if len(state.reservoir) < state.reservoir_size:
                        state.reservoir.append(target_obs.copy())
                # Decrease threshold slightly
                state.anomaly_threshold *= (1.0 - 0.02 * label_confidence)
            
            state.anomaly_threshold = max(0.5, min(0.95, state.anomaly_threshold))
            
        elif entry_type == "drift_event":
            affected_sensors = entry.get("affected_sensors", [])
            drift_type = entry.get("drift_type", "gradual")
            severity = entry.get("severity", 0.5)
            
            # Mark drift for next N observations
            state.drift_detected_count = 10
            
            # Apply adaptation
            apply_adaptation(state, affected_sensors, severity)
            
        elif entry_type == "checkpoint":
            cp_id = entry.get("checkpoint_id")
            action = entry.get("action")
            
            if action == "save" and cp_id:
                checkpoints[cp_id] = state.clone()
            elif action == "restore" and cp_id and cp_id in checkpoints:
                state = checkpoints[cp_id].clone()
                
        elif entry_type == "adjust_config":
            param = entry.get("parameter")
            value = entry.get("value")
            
            if param == "window_size" and isinstance(value, int):
                state.window_size = max(10, min(200, value))
            elif param == "sensitivity" and isinstance(value, (int, float)):
                state.sensitivity = max(0.1, min(2.0, value))
            elif param == "memory_limit" and isinstance(value, int):
                state.reservoir_size = max(100, min(5000, value // 3))
    
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
    """
    subscores = {"correctness": 0.0}
    weights = {"correctness": 1.0}
    
    # Load stream
    entries = load_stream(STREAM)
    if not entries:
        return GradingResult(
            score=0.0,
            feedback="No valid entries in sensor_stream.jsonl",
            subscores=subscores,
            weights=weights
        )
    
    # Compute expected output
    expected_rows = process_stream(entries)
    
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
    expected_obs_count = sum(1 for e in entries if e.get("type") == "observation")
    if len(solution_rows) != expected_obs_count:
        return GradingResult(
            score=0.0,
            feedback=f"Row count mismatch. Expected {expected_obs_count}, got {len(solution_rows)}",
            subscores=subscores,
            weights=weights
        )
    
    
    # Compare rows with tolerance for floating point
    mismatches = []
    float_cols = {"anomaly_score", "confidence"}
    
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
                # Floating point comparison with tolerance
                try:
                    exp_float = float(exp_val)
                    sol_float = float(sol_val)
                    if abs(exp_float - sol_float) > 0.05:  # 5% tolerance
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
    
    if mismatches:
        accuracy = 1.0 - (len(mismatches) / len(expected_rows))
        subscores["correctness"] = max(0.0, accuracy)
        
        score = sum(subscores[k] * weights[k] for k in subscores)
        
        return GradingResult(
            score=score,
            feedback=f"Found {len(mismatches)} mismatches. Accuracy: {accuracy:.2%}",
            subscores=subscores,
            weights=weights,
            details={"mismatches": mismatches[:25]}
        )
    
    subscores["correctness"] = 1.0
    score = sum(subscores[k] * weights[k] for k in subscores)
    
    return GradingResult(
        score=score,
        feedback="All checks passed! Adaptive anomaly detection implemented correctly.",
        subscores=subscores,
        weights=weights
    )

if __name__ == "__main__":
    result = grade(None)
    print(result.model_dump() if hasattr(result, "model_dump") else result.__dict__)