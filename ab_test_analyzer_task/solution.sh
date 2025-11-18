#!/usr/bin/env bash
set -euo pipefail

# Parse experiment data with segmentation, FDR correction, power analysis, confidence intervals
python3 - << 'PY'
import csv, json, gzip, bz2, math, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats

START = datetime.fromisoformat("2024-01-15T00:00:00+00:00")
END = datetime.fromisoformat("2024-01-31T23:59:59+00:00")

VALID_COUNTRIES = {"US", "UK", "CA", "DE", "FR"}
VALID_DEVICES = {"mobile", "desktop", "tablet"}

def iter_files(root: Path):
    """Recursively iterate through all experiment data files."""
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

def parse_jsonl(f):
    for line in f:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        try:
            yield json.loads(line)
        except:
            pass

def parse_csv(f):
    try:
        reader = csv.DictReader(f)
    except:
        return
    for row in reader:
        if any(str(v).strip().startswith("#") or str(v).strip().startswith("//") 
               for v in row.values()):
            continue
        clean = {k: (v.split("#")[0].strip() if isinstance(v, str) else v) 
                 for k, v in row.items()}
        yield clean

def within_window(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except:
        return False
    return START <= dt <= END

def is_binary_metric(values):
    unique_vals = set(values)
    return unique_vals.issubset({0, 1, 0.0, 1.0})

def two_proportion_ztest(control, treatment):
    n1, n2 = len(control), len(treatment)
    p1 = np.sum(control) / n1
    p2 = np.sum(treatment) / n2
    p_pool = (np.sum(control) + np.sum(treatment)) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

def proportion_diff_ci(p1, n1, p2, n2, alpha=0.05):
    """
    Calculate confidence interval for difference in proportions (p2 - p1).
    Uses standard normal approximation with unpooled standard error.
    """
    diff = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_lower = diff - z_crit * se
    ci_upper = diff + z_crit * se
    return [ci_lower, ci_upper]

def welch_ttest(control, treatment):
    result = stats.ttest_ind(treatment, control, equal_var=False)
    return result.statistic, result.pvalue

def cohens_d(control, treatment):
    n1, n2 = len(control), len(treatment)
    s1, s2 = np.std(control, ddof=1), np.std(treatment, ddof=1)
    pooled_std = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(treatment) - np.mean(control)) / pooled_std

def power_ztest(n1, n2, effect_size, alpha=0.05):
    """Power for two-proportion z-test."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    noncentrality = effect_size * math.sqrt(n1 * n2 / (n1 + n2))
    power = 1 - stats.norm.cdf(z_alpha - abs(noncentrality)) + stats.norm.cdf(-z_alpha - abs(noncentrality))
    return max(0, min(1, power))

def power_ttest(n1, n2, effect_size, alpha=0.05):
    """Power for Welch's t-test."""
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    noncentrality = effect_size * math.sqrt(n1 * n2 / (n1 + n2))
    power = 1 - stats.nct.cdf(t_crit, df, noncentrality) + stats.nct.cdf(-t_crit, df, noncentrality)
    return max(0, min(1, power))

# Load data
root = Path("/workdir/data/experiments")
seen = {}  # (exp_id, user_id, variant, metric_type) -> (timestamp, full_record)
data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"control": [], "treatment": []})))

for p, kind, f in iter_files(root):
    with f:
        it = parse_jsonl(f) if kind == "jsonl" else parse_csv(f)
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
            except:
                continue
            
            # Filter: only control and treatment
            if variant not in ["control", "treatment"]:
                continue
            
            # Filter: valid country and device
            if country not in VALID_COUNTRIES or device not in VALID_DEVICES:
                continue
            
            # Filter: within time window
            if not within_window(timestamp):
                continue
            
            # Deduplication: keep LAST occurrence (now includes variant in key)
            key = (exp_id, user_id, variant, metric_type)
            if key in seen:
                old_ts, _ = seen[key]
                try:
                    if datetime.fromisoformat(timestamp.replace("Z", "+00:00")) <= \
                       datetime.fromisoformat(old_ts.replace("Z", "+00:00")):
                        continue
                except:
                    continue
            
            seen[key] = (timestamp, (exp_id, country, device, metric_type, variant, metric_value))

# Build data structure from deduplicated records
for (exp_id, country, device, metric_type, variant, metric_value) in [r[1] for r in seen.values()]:
    data[exp_id][(country, device)][metric_type][variant].append(metric_value)

# Perform analysis
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
            
            if not variants["control"] or not variants["treatment"]:
                continue
            
            control = np.array(variants["control"])
            treatment = np.array(variants["treatment"])
            
            # Sample size filter
            if len(control) < 30 or len(treatment) < 30:
                continue
            
            all_values = list(control) + list(treatment)
            is_binary = is_binary_metric(all_values)
            
            control_value = float(np.mean(control))
            treatment_value = float(np.mean(treatment))
            
            if is_binary:
                metric_kind = "binary"
                test_stat, p_value = two_proportion_ztest(control, treatment)
                
                # CI for difference in proportions (not lift)
                ci_diff = proportion_diff_ci(control_value, len(control), 
                                            treatment_value, len(treatment))
                
                effect = abs(treatment_value - control_value)
                pwr = power_ztest(len(control), len(treatment), effect)
                
                metric_result = {
                    "metric_type": metric_type,
                    "metric_kind": metric_kind,
                    "control_n": len(control),
                    "treatment_n": len(treatment),
                    "control_value": round(control_value, 6),
                    "treatment_value": round(treatment_value, 6),
                    "lift_percent": round(((treatment_value - control_value) / control_value * 100) if control_value != 0 else 0.0, 4),
                    "confidence_interval": [round(ci_diff[0], 4), round(ci_diff[1], 4)],
                    "test_statistic": round(test_stat, 4),
                    "p_value": round(p_value, 6),
                    "power": round(pwr, 4),
                }
            else:
                metric_kind = "continuous"
                test_stat, p_value = welch_ttest(control, treatment)
                
                # CI for difference in means
                diff = treatment_value - control_value
                se_diff = math.sqrt(np.var(control, ddof=1)/len(control) + np.var(treatment, ddof=1)/len(treatment))
                df = len(control) + len(treatment) - 2
                t_crit = stats.t.ppf(0.975, df)
                ci_diff = [diff - t_crit*se_diff, diff + t_crit*se_diff]
                
                effect_size = cohens_d(control, treatment)
                pwr = power_ttest(len(control), len(treatment), abs(effect_size))
                
                metric_result = {
                    "metric_type": metric_type,
                    "metric_kind": metric_kind,
                    "control_n": len(control),
                    "treatment_n": len(treatment),
                    "control_value": round(control_value, 6),
                    "treatment_value": round(treatment_value, 6),
                    "lift_percent": round(((treatment_value - control_value) / control_value * 100) if control_value != 0 else 0.0, 4),
                    "confidence_interval": [round(ci_diff[0], 4), round(ci_diff[1], 4)],
                    "test_statistic": round(test_stat, 4),
                    "p_value": round(p_value, 6),
                    "effect_size": round(effect_size, 4),
                    "power": round(pwr, 4),
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
if total_tests > 0:
    sorted_pvals = sorted([(p, i) for i, p in enumerate(all_pvalues)])
    fdr_threshold = 0.0
    for rank, (pval, _) in enumerate(sorted_pvals, 1):
        threshold = (rank / total_tests) * 0.05
        if pval <= threshold:
            fdr_threshold = threshold
    fdr_threshold = round(fdr_threshold, 6)
    
    # Mark significance
    for exp in experiments:
        for seg in exp["segments"]:
            for metric in seg["metrics"]:
                metric["significant"] = bool(metric["p_value"] <= fdr_threshold)
else:
    fdr_threshold = 0.0

output = {
    "total_tests": total_tests,
    "fdr_threshold": fdr_threshold,
    "experiments": experiments
}

with open("/workdir/results.json", "w") as f:
    json.dump(output, f, indent=2)

PY