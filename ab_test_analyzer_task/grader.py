# grader.py - CORRECTED VERSION WITH BINARY GRADING
import csv
import json
import bz2
import gzip
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy import stats
from collections import defaultdict

@dataclass
class GradingResult:
    """Structured result returned by the grader."""
    score: float
    feedback: str | None = None
    subscores: dict = field(default_factory=dict)
    details: dict | None = None
    weights: dict | None = None

# Task-specified constants
START = datetime.fromisoformat("2024-01-15T00:00:00+00:00")
END = datetime.fromisoformat("2024-01-31T23:59:59+00:00")
VALID_COUNTRIES = {"US", "UK", "CA", "DE", "FR"}
VALID_DEVICES = {"mobile", "desktop", "tablet"}
MIN_SAMPLE_SIZE = 30
FDR_ALPHA = 0.05

def _iter_files(root: Path):
    """
    Recursively iterate through all experiment data files.
    
    Supports compressed and uncompressed formats:
      - *.jsonl and *.jsonl.bz2
      - *.csv and *.csv.gz
    
    Returns tuples: (path, kind, file_handle)
      kind ∈ {"jsonl", "csv"}
    """
    if not root.exists():
        return
    
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            
            if p.suffix == ".gz":
                f = gzip.open(p, "rt", encoding="utf-8", errors="ignore")
                kind = "csv"
            elif p.suffix == ".bz2":
                f = bz2.open(p, "rt", encoding="utf-8", errors="ignore")
                kind = "jsonl"
            elif p.suffix == ".jsonl":
                f = open(p, "rt", encoding="utf-8", errors="ignore")
                kind = "jsonl"
            elif p.suffix == ".csv":
                f = open(p, "rt", encoding="utf-8", errors="ignore")
                kind = "csv"
            else:
                continue
            
            yield p, kind, f

def _parse_jsonl(f):
    """
    Parse a JSONL file into Python dicts.
    
    Skips blank lines and lines starting with '//' or '#' (treated as comments).
    Malformed lines are ignored.
    """
    for line in f:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
            yield obj
        except Exception:
            continue

def _parse_csv(f):
    """
    Parse a CSV file into dicts.
    
    Strips inline '#' comments from cell values.
    Skips comment lines.
    Malformed rows are ignored.
    """
    try:
        reader = csv.DictReader(f)
    except Exception:
        return
    
    for row in reader:
        # Skip rows that are comments
        if any(str(v).strip().startswith("#") or str(v).strip().startswith("//") 
               for v in row.values()):
            continue
        
        clean = {}
        for k, v in row.items():
            if isinstance(v, str):
                # Strip inline comments
                v = v.split("#")[0].strip()
            clean[k] = v
        yield clean

def _within_window(ts: str) -> bool:
    """Check if timestamp is within the valid date range."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except:
        return False
    return START <= dt <= END

def _is_binary_metric(values: List[float]) -> bool:
    """Check if all values are in {0, 1, 0.0, 1.0}."""
    unique_vals = set(values)
    return unique_vals.issubset({0, 1, 0.0, 1.0})

def _wilson_ci(successes: int, n: int, alpha: float = 0.05) -> List[float]:
    """Wilson score confidence interval for proportion."""
    if n == 0:
        return [0.0, 0.0]
    
    p = successes / n
    z = stats.norm.ppf(1 - alpha/2)
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denominator
    spread = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
    
    return [max(0, centre - spread), min(1, centre + spread)]

def _two_proportion_ztest(control: np.ndarray, treatment: np.ndarray) -> Tuple[float, float]:
    """
    Perform two-proportion z-test.
    Returns: z_statistic, p_value
    """
    n1, n2 = len(control), len(treatment)
    p1 = np.sum(control) / n1
    p2 = np.sum(treatment) / n2
    
    # Pooled proportion
    p_pool = (np.sum(control) + np.sum(treatment)) / (n1 + n2)
    
    # Standard error
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    if se == 0:
        return 0.0, 1.0
    
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value

def _welch_ttest(control: np.ndarray, treatment: np.ndarray) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variances).
    Returns: t_statistic, p_value
    """
    result = stats.ttest_ind(treatment, control, equal_var=False)
    return result.statistic, result.pvalue

def _cohens_d(control: np.ndarray, treatment: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(control), len(treatment)
    s1, s2 = np.std(control, ddof=1), np.std(treatment, ddof=1)
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(treatment) - np.mean(control)) / pooled_std

def _power_ztest(n1: int, n2: int, effect_size: float, alpha: float = 0.05) -> float:
    """Calculate statistical power for two-proportion z-test."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    noncentrality = effect_size * math.sqrt(n1 * n2 / (n1 + n2))
    power = 1 - stats.norm.cdf(z_alpha - abs(noncentrality)) + stats.norm.cdf(-z_alpha - abs(noncentrality))
    return max(0, min(1, power))

def _power_ttest(n1: int, n2: int, effect_size: float, alpha: float = 0.05) -> float:
    """Calculate statistical power for Welch's t-test."""
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    noncentrality = effect_size * math.sqrt(n1 * n2 / (n1 + n2))
    power = 1 - stats.nct.cdf(t_crit, df, noncentrality) + stats.nct.cdf(-t_crit, df, noncentrality)
    return max(0, min(1, power))

def _compute_expected(workdir: Path) -> dict:
    """
    Compute the expected analysis results from ground-truth data.
    
    Implements ALL task requirements:
    - Deduplicates by (experiment_id, user_id, metric_type) - keeps LAST occurrence
    - Filters to only control and treatment variants
    - Filters by date range and valid country/device
    - Performs SEGMENTED analysis by country and device
    - Applies sample size filtering (minimum 30)
    - Performs statistical tests with confidence intervals and power analysis
    - Applies Benjamini-Hochberg FDR correction
    """
    root = workdir / "data" / "experiments"
    
    # First pass: deduplicate by keeping LAST occurrence
    seen = {}  # (exp_id, user_id, metric_type) -> (timestamp, full_record)
    
    for p, kind, f in _iter_files(root):
        with f:
            it = _parse_jsonl(f) if kind == "jsonl" else _parse_csv(f)
            
            for obj in it:
                try:
                    exp_id = obj["experiment_id"]
                    user_id = obj["user_id"]
                    variant = obj["variant"]
                    metric_type = obj["metric_type"]
                    metric_value = float(obj["metric_value"])
                    timestamp = obj["timestamp"]
                    country = obj.get("country", "")
                    device = obj.get("device", "")
                except Exception:
                    continue
                
                # Filter: only control and treatment
                if variant not in ["control", "treatment"]:
                    continue
                
                # Filter: valid country and device
                if country not in VALID_COUNTRIES or device not in VALID_DEVICES:
                    continue
                
                # Filter: within time window
                if not _within_window(timestamp):
                    continue
                
                # Deduplication: keep LAST occurrence
                key = (exp_id, user_id, metric_type)
                if key in seen:
                    old_ts, _ = seen[key]
                    try:
                        old_dt = datetime.fromisoformat(old_ts.replace("Z", "+00:00"))
                        new_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        if new_dt <= old_dt:
                            continue  # Keep the old one (later timestamp)
                    except:
                        continue
                
                # Store the record
                seen[key] = (timestamp, {
                    "exp_id": exp_id,
                    "country": country,
                    "device": device,
                    "metric_type": metric_type,
                    "variant": variant,
                    "metric_value": metric_value
                })
    
    # Second pass: build segmented data structure
    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {"control": [], "treatment": []}
            )
        )
    )
    
    for timestamp, record in seen.values():
        exp_id = record["exp_id"]
        country = record["country"]
        device = record["device"]
        metric_type = record["metric_type"]
        variant = record["variant"]
        metric_value = record["metric_value"]
        
        data[exp_id][(country, device)][metric_type][variant].append(metric_value)
    
    # Third pass: perform statistical analysis
    experiments = []
    all_pvalues = []
    
    for exp_id in sorted(data.keys()):
        segments_data = data[exp_id]
        exp_segments = []
        
        for (country, device) in sorted(segments_data.keys()):
            metrics_data = segments_data[(country, device)]
            seg_metrics = []
            
            for metric_type in sorted(metrics_data.keys()):
                variants = metrics_data[metric_type]
                
                # Must have both control and treatment
                if not variants["control"] or not variants["treatment"]:
                    continue
                
                control = np.array(variants["control"])
                treatment = np.array(variants["treatment"])
                
                # Sample size filtering: minimum 30 in BOTH groups
                if len(control) < MIN_SAMPLE_SIZE or len(treatment) < MIN_SAMPLE_SIZE:
                    continue
                
                # Determine metric kind
                all_values = list(control) + list(treatment)
                is_binary = _is_binary_metric(all_values)
                
                control_value = float(np.mean(control))
                treatment_value = float(np.mean(treatment))
                
                # Calculate lift
                if control_value == 0:
                    lift_percent = 0.0
                else:
                    lift_percent = ((treatment_value - control_value) / control_value) * 100
                
                if is_binary:
                    metric_kind = "binary"
                    test_stat, p_value = _two_proportion_ztest(control, treatment)
                    
                    # Wilson CI for proportions, then convert to lift CI
                    ci_control = _wilson_ci(int(np.sum(control)), len(control))
                    ci_treatment = _wilson_ci(int(np.sum(treatment)), len(treatment))
                    
                    if control_value == 0:
                        lift_ci = [0.0, 0.0]
                    else:
                        # Conservative CI for lift
                        lift_ci = [
                            ((ci_treatment[0] - ci_control[1]) / control_value) * 100,
                            ((ci_treatment[1] - ci_control[0]) / control_value) * 100
                        ]
                    
                    # Power calculation
                    effect = abs(treatment_value - control_value)
                    power = _power_ztest(len(control), len(treatment), effect)
                    
                    metric_result = {
                        "metric_type": metric_type,
                        "metric_kind": metric_kind,
                        "control_n": len(control),
                        "treatment_n": len(treatment),
                        "control_value": round(control_value, 6),
                        "treatment_value": round(treatment_value, 6),
                        "lift_percent": round(lift_percent, 4),
                        "confidence_interval": [round(lift_ci[0], 4), round(lift_ci[1], 4)],
                        "test_statistic": round(test_stat, 4),
                        "p_value": round(p_value, 6),
                        "power": round(power, 4),
                    }
                else:
                    metric_kind = "continuous"
                    test_stat, p_value = _welch_ttest(control, treatment)
                    
                    # CI for difference in means
                    diff = treatment_value - control_value
                    se_diff = math.sqrt(
                        np.var(control, ddof=1)/len(control) + 
                        np.var(treatment, ddof=1)/len(treatment)
                    )
                    df = len(control) + len(treatment) - 2
                    t_crit = stats.t.ppf(0.975, df)
                    ci_diff = [diff - t_crit*se_diff, diff + t_crit*se_diff]
                    
                    # Effect size and power
                    effect_size = _cohens_d(control, treatment)
                    power = _power_ttest(len(control), len(treatment), abs(effect_size))
                    
                    metric_result = {
                        "metric_type": metric_type,
                        "metric_kind": metric_kind,
                        "control_n": len(control),
                        "treatment_n": len(treatment),
                        "control_value": round(control_value, 6),
                        "treatment_value": round(treatment_value, 6),
                        "lift_percent": round(lift_percent, 4),
                        "confidence_interval": [round(ci_diff[0], 4), round(ci_diff[1], 4)],
                        "test_statistic": round(test_stat, 4),
                        "p_value": round(p_value, 6),
                        "effect_size": round(effect_size, 4),
                        "power": round(power, 4),
                    }
                
                seg_metrics.append(metric_result)
                all_pvalues.append(p_value)
            
            if seg_metrics:
                exp_segments.append({
                    "country": country,
                    "device": device,
                    "metrics": seg_metrics
                })
        
        if exp_segments:
            experiments.append({
                "experiment_id": exp_id,
                "segments": exp_segments
            })
    
    # Apply Benjamini-Hochberg FDR correction
    total_tests = len(all_pvalues)
    fdr_threshold = 0.0
    
    if total_tests > 0:
        # Sort p-values with their indices
        sorted_pvals = sorted([(p, i) for i, p in enumerate(all_pvalues)])
        
        # Find the largest k such that P(k) <= (k/m) * q
        for rank, (pval, _) in enumerate(sorted_pvals, 1):
            threshold = (rank / total_tests) * FDR_ALPHA
            if pval <= threshold:
                fdr_threshold = threshold
        
        fdr_threshold = round(fdr_threshold, 6)
        
        # Mark significance based on FDR threshold
        for exp in experiments:
            for seg in exp["segments"]:
                for metric in seg["metrics"]:
                    metric["significant"] = bool(metric["p_value"] <= fdr_threshold)
    
    return {
        "total_tests": total_tests,
        "fdr_threshold": fdr_threshold,
        "experiments": experiments
    }

def _read_solution(path: Path) -> dict | None:
    """
    Load the contestant's /workdir/results.json.
    
    Returns None if file doesn't exist or is malformed.
    """
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _compare_results(expected: dict, solution: dict, tolerance: float = 1e-5) -> Tuple[bool, List[dict]]:
    """
    Compare expected and solution results with detailed error reporting.
    
    Returns: (is_correct, list_of_differences)
    """
    differences = []
    
    # Check top-level fields
    if expected["total_tests"] != solution.get("total_tests"):
        differences.append({
            "level": "top",
            "field": "total_tests",
            "expected": expected["total_tests"],
            "got": solution.get("total_tests")
        })
    
    exp_fdr = expected["fdr_threshold"]
    sol_fdr = solution.get("fdr_threshold")
    if sol_fdr is None or abs(exp_fdr - sol_fdr) > tolerance:
        differences.append({
            "level": "top",
            "field": "fdr_threshold",
            "expected": exp_fdr,
            "got": sol_fdr
        })
    
    # Check experiments structure
    exp_expected = {e["experiment_id"]: e for e in expected["experiments"]}
    exp_solution = {e["experiment_id"]: e for e in solution.get("experiments", [])}
    
    if set(exp_expected.keys()) != set(exp_solution.keys()):
        differences.append({
            "level": "experiments",
            "field": "experiment_ids",
            "expected": sorted(exp_expected.keys()),
            "got": sorted(exp_solution.keys())
        })
        return False, differences
    
    # Check each experiment's segments
    for exp_id in sorted(exp_expected.keys()):
        exp_e = exp_expected[exp_id]
        exp_s = exp_solution[exp_id]
        
        # Create lookup for segments
        segs_e = {(s["country"], s["device"]): s for s in exp_e["segments"]}
        segs_s = {(s["country"], s["device"]): s for s in exp_s.get("segments", [])}
        
        if set(segs_e.keys()) != set(segs_s.keys()):
            differences.append({
                "level": "segments",
                "experiment_id": exp_id,
                "field": "segment_keys",
                "expected": sorted(segs_e.keys()),
                "got": sorted(segs_s.keys())
            })
            continue
        
        # Check each segment's metrics
        for seg_key in sorted(segs_e.keys()):
            seg_e = segs_e[seg_key]
            seg_s = segs_s[seg_key]
            
            metrics_e = {m["metric_type"]: m for m in seg_e["metrics"]}
            metrics_s = {m["metric_type"]: m for m in seg_s.get("metrics", [])}
            
            if set(metrics_e.keys()) != set(metrics_s.keys()):
                differences.append({
                    "level": "metrics",
                    "experiment_id": exp_id,
                    "segment": seg_key,
                    "field": "metric_types",
                    "expected": sorted(metrics_e.keys()),
                    "got": sorted(metrics_s.keys())
                })
                continue
            
            # Check each metric's fields
            for metric_type in sorted(metrics_e.keys()):
                m_e = metrics_e[metric_type]
                m_s = metrics_s[metric_type]
                
                # Exact match fields
                exact_fields = ["metric_kind", "control_n", "treatment_n", "significant"]
                for field in exact_fields:
                    if m_e[field] != m_s.get(field):
                        differences.append({
                            "level": "metric_field",
                            "experiment_id": exp_id,
                            "segment": seg_key,
                            "metric_type": metric_type,
                            "field": field,
                            "expected": m_e[field],
                            "got": m_s.get(field)
                        })
                
                # Numerical fields with tolerance
                numeric_fields = [
                    "control_value", "treatment_value", "lift_percent",
                    "test_statistic", "p_value", "power"
                ]
                if m_e["metric_kind"] == "continuous":
                    numeric_fields.append("effect_size")
                
                for field in numeric_fields:
                    exp_val = m_e[field]
                    sol_val = m_s.get(field)
                    
                    if sol_val is None or abs(exp_val - sol_val) > tolerance:
                        differences.append({
                            "level": "metric_field",
                            "experiment_id": exp_id,
                            "segment": seg_key,
                            "metric_type": metric_type,
                            "field": field,
                            "expected": exp_val,
                            "got": sol_val
                        })
                
                # Check confidence interval
                ci_e = m_e["confidence_interval"]
                ci_s = m_s.get("confidence_interval", [None, None])
                
                if len(ci_s) != 2 or \
                   ci_s[0] is None or ci_s[1] is None or \
                   abs(ci_e[0] - ci_s[0]) > tolerance or \
                   abs(ci_e[1] - ci_s[1]) > tolerance:
                    differences.append({
                        "level": "metric_field",
                        "experiment_id": exp_id,
                        "segment": seg_key,
                        "metric_type": metric_type,
                        "field": "confidence_interval",
                        "expected": ci_e,
                        "got": ci_s
                    })
    
    return len(differences) == 0, differences

def grade(transcript: str | None = None) -> GradingResult:
    """
    Main grading entrypoint with BINARY scoring.
    
    Agent must get everything correct to pass (score = 1.0).
    Any errors result in failure (score = 0.0).
    
    Returns GradingResult with detailed feedback on what went wrong.
    """
    workdir = Path("/workdir")
    
    # Binary grading - single component with full weight
    weights = {"correctness": 1.0}
    subscores = {"correctness": 0.0}
    
    # Load solution
    sol = _read_solution(workdir / "results.json")
    
    if sol is None:
        if not (workdir / "results.json").exists():
            feedback = "FAIL: Missing /workdir/results.json"
        else:
            feedback = "FAIL: Invalid JSON in /workdir/results.json"
        return GradingResult(score=0.0, feedback=feedback, subscores=subscores, weights=weights)
    
    # Check basic structure
    required_keys = {"total_tests", "fdr_threshold", "experiments"}
    if not required_keys.issubset(sol.keys()):
        feedback = f"FAIL: Missing required top-level keys. Expected: {required_keys}, got: {set(sol.keys())}"
        return GradingResult(score=0.0, feedback=feedback, subscores=subscores, weights=weights)
    
    if not isinstance(sol["experiments"], list):
        feedback = "FAIL: 'experiments' must be a list"
        return GradingResult(score=0.0, feedback=feedback, subscores=subscores, weights=weights)
    
    # Compute expected results
    try:
        expected = _compute_expected(workdir)
    except Exception as e:
        feedback = f"FAIL: Error computing expected results: {str(e)}"
        return GradingResult(score=0.0, feedback=feedback, subscores=subscores, weights=weights)
    
    # Compare results - BINARY GRADING
    is_correct, differences = _compare_results(expected, sol)
    
    if is_correct:
        # PASS: Everything correct
        subscores["correctness"] = 1.0
        feedback = "✓ PASS: All tests passed! Statistical analysis is completely correct."
        return GradingResult(score=1.0, feedback=feedback, subscores=subscores, weights=weights)
    else:
        # FAIL: Any error means failure
        subscores["correctness"] = 0.0
        feedback = f"✗ FAIL: Found {len(differences)} error(s). All calculations must be correct to pass.\n\n"
        
        # Show first 10 differences for debugging
        for i, diff in enumerate(differences[:10], 1):
            if diff["level"] == "top":
                feedback += f"  {i}. Top-level field '{diff['field']}': expected {diff['expected']}, got {diff['got']}\n"
            elif diff["level"] == "experiments":
                feedback += f"  {i}. Experiment IDs mismatch: expected {diff['expected']}, got {diff['got']}\n"
            elif diff["level"] == "segments":
                feedback += f"  {i}. Segment keys for {diff['experiment_id']}: expected {diff['expected']}, got {diff['got']}\n"
            elif diff["level"] == "metrics":
                feedback += f"  {i}. Metric types for {diff['experiment_id']}/{diff['segment']}: expected {diff['expected']}, got {diff['got']}\n"
            elif diff["level"] == "metric_field":
                feedback += f"  {i}. {diff['experiment_id']}/{diff['segment']}/{diff['metric_type']}.{diff['field']}: expected {diff['expected']}, got {diff['got']}\n"
        
        if len(differences) > 10:
            feedback += f"\n  ... and {len(differences) - 10} more error(s)\n"
        
        feedback += "\nHint: Review data processing, deduplication, segmentation, and statistical calculations."
        
        details = {"differences": differences[:50], "total_differences": len(differences)}
        
        return GradingResult(
            score=0.0,
            feedback=feedback.strip(),
            subscores=subscores,
            weights=weights,
            details=details
        )