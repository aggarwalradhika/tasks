# grader.py - Simplified version using ground truth files
import json
from pathlib import Path
from typing import List, Tuple
from apex_arena._types import GradingResult

def _read_json(path: Path) -> dict | None:
    """
    Load a JSON file.
    
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
    exp_expected = {e["experiment_id"]: e for e in expected.get("experiments", [])}
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
        segs_e = {(s["country"], s["device"]): s for s in exp_e.get("segments", [])}
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
            
            metrics_e = {m["metric_type"]: m for m in seg_e.get("metrics", [])}
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
                    if m_e.get(field) != m_s.get(field):
                        differences.append({
                            "level": "metric_field",
                            "experiment_id": exp_id,
                            "segment": seg_key,
                            "metric_type": metric_type,
                            "field": field,
                            "expected": m_e.get(field),
                            "got": m_s.get(field)
                        })
                
                # Numerical fields with tolerance
                numeric_fields = [
                    "control_value", "treatment_value", "lift_percent",
                    "test_statistic", "p_value", "power"
                ]
                if m_e.get("metric_kind") == "continuous":
                    numeric_fields.append("effect_size")
                
                for field in numeric_fields:
                    exp_val = m_e.get(field)
                    sol_val = m_s.get(field)
                    
                    if exp_val is None or sol_val is None or abs(exp_val - sol_val) > tolerance:
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
                ci_e = m_e.get("confidence_interval", [])
                ci_s = m_s.get("confidence_interval", [])
                
                if len(ci_e) != 2 or len(ci_s) != 2 or \
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
    sol = _read_json(workdir / "results.json")
    
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
    
    # Load ground truth
    expected = _read_json(Path("/tests/ground_truth.json"))
    
    if expected is None:
        feedback = "FAIL: Ground truth file missing or invalid"
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