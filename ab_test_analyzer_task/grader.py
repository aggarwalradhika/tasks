# grader.py
import csv
import json
import bz2
import gzip
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from collections import defaultdict
import math

@dataclass
class GradingResult:
    """Structured result returned by the grader."""
    score: float
    feedback: str | None = None
    subscores: dict = field(default_factory=dict)
    details: dict | None = None
    weights: dict | None = None

def _iter_files(root: Path):
    """
    Yield open file handles for each experiment data file.
    
    Supports compressed and uncompressed formats:
      - *.jsonl and *.jsonl.bz2
      - *.csv and *.csv.gz
    
    Returns tuples: (filename, kind, file_handle)
      kind ∈ {"jsonl", "csv"}
    """
    logs = root / "data" / "experiments"
    if not logs.exists():
        return
    for p in logs.iterdir():
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                yield p.name, "csv", f
        elif p.suffix == ".bz2":
            with bz2.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                yield p.name, "jsonl", f
        elif p.suffix == ".jsonl":
            yield p.name, "jsonl", open(p, "rt", encoding="utf-8", errors="ignore")
        elif p.suffix == ".csv":
            yield p.name, "csv", open(p, "rt", encoding="utf-8", errors="ignore")

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
                v = v.split("#")[0].strip()
            clean[k] = v
        yield clean

def _is_binary_metric(values):
    """Check if all values are in {0, 1, 0.0, 1.0}."""
    unique_vals = set(values)
    return unique_vals.issubset({0, 1, 0.0, 1.0})

def _two_proportion_ztest(control, treatment):
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

def _welch_ttest(control, treatment):
    """
    Perform Welch's t-test (unequal variances).
    Returns: t_statistic, p_value
    """
    result = stats.ttest_ind(treatment, control, equal_var=False)
    return result.statistic, result.pvalue

def _compute_expected(workdir: Path) -> dict:
    """
    Compute the expected analysis results from ground-truth data.
    
    - Deduplicates by (experiment_id, user_id, metric_type)
    - Filters to only control and treatment variants
    - Performs statistical tests
    - Applies Bonferroni correction
    """
    seen = set()
    data = defaultdict(lambda: defaultdict(lambda: {"control": [], "treatment": []}))
    
    for name, kind, f in _iter_files(workdir) or []:
        with f:
            it = _parse_jsonl(f) if kind == "jsonl" else _parse_csv(f)
            for obj in it:
                try:
                    exp_id = obj["experiment_id"]
                    user_id = obj["user_id"]
                    variant = obj["variant"]
                    metric_type = obj["metric_type"]
                    metric_value = float(obj["metric_value"])
                except Exception:
                    continue
                
                # Only consider control and treatment
                if variant not in ["control", "treatment"]:
                    continue
                
                # Deduplicate: keep first occurrence
                key = (exp_id, user_id, metric_type)
                if key in seen:
                    continue
                seen.add(key)
                
                data[exp_id][metric_type][variant].append(metric_value)
    
    # Perform statistical analysis
    experiments = []
    
    for exp_id in sorted(data.keys()):
        metrics_data = data[exp_id]
        exp_metrics = []
        
        for metric_type in sorted(metrics_data.keys()):
            variants = metrics_data[metric_type]
            
            # Must have both control and treatment
            if not variants["control"] or not variants["treatment"]:
                continue
            
            control = np.array(variants["control"])
            treatment = np.array(variants["treatment"])
            
            # Determine metric kind
            all_values = list(control) + list(treatment)
            is_binary = _is_binary_metric(all_values)
            
            if is_binary:
                metric_kind = "binary"
                control_value = float(np.mean(control))
                treatment_value = float(np.mean(treatment))
                test_stat, p_value = _two_proportion_ztest(control, treatment)
            else:
                metric_kind = "continuous"
                control_value = float(np.mean(control))
                treatment_value = float(np.mean(treatment))
                test_stat, p_value = _welch_ttest(control, treatment)
            
            # Calculate lift
            if control_value == 0:
                lift_percent = 0.0
            else:
                lift_percent = ((treatment_value - control_value) / control_value) * 100
            
            exp_metrics.append({
                "metric_type": metric_type,
                "metric_kind": metric_kind,
                "control_value": round(control_value, 6),
                "treatment_value": round(treatment_value, 6),
                "lift_percent": round(lift_percent, 4),
                "test_statistic": round(test_stat, 4),
                "p_value": round(p_value, 6),
            })
        
        if exp_metrics:
            experiments.append({
                "experiment_id": exp_id,
                "metrics": exp_metrics
            })
    
    # Apply Bonferroni correction
    total_tests = sum(len(exp["metrics"]) for exp in experiments)
    corrected_alpha = 0.05 / total_tests if total_tests > 0 else 0.05
    
    # Mark significance
    for exp in experiments:
        for metric in exp["metrics"]:
            metric["significant"] = bool(metric["p_value"] < corrected_alpha)
    
    return {
        "total_tests": total_tests,
        "corrected_alpha": round(corrected_alpha, 6),
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

def _compare_results(expected: dict, solution: dict) -> tuple[bool, list]:
    """
    Compare expected and solution results.
    
    Returns: (is_correct, list_of_differences)
    """
    differences = []
    
    # Check top-level fields
    if expected["total_tests"] != solution.get("total_tests"):
        differences.append({
            "field": "total_tests",
            "expected": expected["total_tests"],
            "got": solution.get("total_tests")
        })
    
    if expected["corrected_alpha"] != solution.get("corrected_alpha"):
        differences.append({
            "field": "corrected_alpha",
            "expected": expected["corrected_alpha"],
            "got": solution.get("corrected_alpha")
        })
    
    # Check experiments
    exp_expected = {e["experiment_id"]: e for e in expected["experiments"]}
    exp_solution = {e["experiment_id"]: e for e in solution.get("experiments", [])}
    
    if set(exp_expected.keys()) != set(exp_solution.keys()):
        differences.append({
            "field": "experiment_ids",
            "expected": sorted(exp_expected.keys()),
            "got": sorted(exp_solution.keys())
        })
        return False, differences
    
    # Check each experiment's metrics
    for exp_id in sorted(exp_expected.keys()):
        exp_e = exp_expected[exp_id]
        exp_s = exp_solution[exp_id]
        
        # Create dictionaries for comparison
        metrics_e = {m["metric_type"]: m for m in exp_e["metrics"]}
        metrics_s = {m["metric_type"]: m for m in exp_s.get("metrics", [])}
        
        if set(metrics_e.keys()) != set(metrics_s.keys()):
            differences.append({
                "experiment_id": exp_id,
                "field": "metric_types",
                "expected": sorted(metrics_e.keys()),
                "got": sorted(metrics_s.keys())
            })
            continue
        
        # Check each metric
        for metric_type in sorted(metrics_e.keys()):
            m_e = metrics_e[metric_type]
            m_s = metrics_s[metric_type]
            
            for field in ["metric_kind", "control_value", "treatment_value", 
                         "lift_percent", "test_statistic", "p_value", "significant"]:
                if m_e[field] != m_s.get(field):
                    differences.append({
                        "experiment_id": exp_id,
                        "metric_type": metric_type,
                        "field": field,
                        "expected": m_e[field],
                        "got": m_s.get(field)
                    })
    
    return len(differences) == 0, differences

def grade(transcript: str | None = None) -> GradingResult:
    """
    Main grading entrypoint.
    
    Compares contestant solution (results.json) with expected output.
    Returns GradingResult with single all_passes subscore (1.0 or 0.0).
    """
    workdir = Path("/workdir")
    
    subs = {
        "all_passes": 0.0,
    }
    weights = {
        "all_passes": 1.0,
    }
    
    sol = _read_solution(workdir / "results.json")
    
    if sol is None:
        if not (workdir / "results.json").exists():
            feedback = "Missing /workdir/results.json"
        else:
            feedback = "Invalid JSON in /workdir/results.json"
        return GradingResult(score=0.0, feedback=feedback, subscores=subs, weights=weights)
    
    # Check basic structure
    required_keys = {"total_tests", "fdr_threshold", "experiments"}
    if not required_keys.issubset(sol.keys()):
        feedback = f"Missing required keys. Expected: {required_keys}"
        return GradingResult(score=0.0, feedback=feedback, subscores=subs, weights=weights)
    
    if not isinstance(sol["experiments"], list):
        feedback = "'experiments' must be a list"
        return GradingResult(score=0.0, feedback=feedback, subscores=subs, weights=weights)
    
    # Compute expected and compare
    expected = _compute_expected(workdir)
    is_correct, differences = _compare_results(expected, sol)
    
    if is_correct:
        subs["all_passes"] = 1.0
        feedback = "All tests passed! Statistical analysis is correct."
    else:
        feedback = f"Found {len(differences)} difference(s) in the analysis."
        details = {"differences": differences[:10]}  # Show first 10 differences
        return GradingResult(score=0.0, feedback=feedback, subscores=subs, 
                           weights=weights, details=details)
    
    return GradingResult(score=1.0, feedback=feedback, subscores=subs, weights=weights)