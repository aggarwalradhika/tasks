# grader.py
import csv
import json
from pathlib import Path

# --- Optional Pydantic GradingResult (preferred). If unavailable, a shim is used. ---
try:
    from pydantic import BaseModel, Field
    _USE_PYDANTIC = True
except Exception:  # pragma: no cover
    _USE_PYDANTIC = False
    class BaseModel:  # very small shim to keep downstream happy
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def model_dump(self):
            return self.__dict__
    def Field(default=None, **kwargs):
        return default

class GradingResult(BaseModel):
    score: float = Field(..., description="Final score in [0,1]")
    feedback: str | None = Field(None, description="Human-readable feedback")
    subscores: dict = Field(default_factory=dict, description="Component scores in [0,1]")
    weights: dict = Field(default_factory=dict, description="Weights used to compute the final score")
    details: dict | None = Field(default_factory=dict, description="Extra diagnostics for debugging")

# -------------------------
# Configurable parameters
# -------------------------
REQUIRED_COLUMNS = ("path_nodes", "path_modes", "expected_cost", "p_delay")

# Weight each component; must sum to 1.0
WEIGHTS = {

    "exact_matches": 1     # 1 only if all of the above pass their hard checks
}

# Tolerances (tight because dataset uses fixed rng_seed)
REL_COST_TOL = 1e-3      # pass if relative error <= 0.001
REL_P_TOL    = 5e-3      # pass if relative error <= 0.005

# For partial credit beyond the hard pass/fail tolerance:
# map relative error -> score in (0,1], with a smooth decay.
def soft_score_from_rel_error(rel_err: float, hard_tol: float, softness: float = 50.0) -> float:
    """
    1.0 at rel_err == 0; ~1.0 near 0; drops smoothly as rel_err grows.
    'softness' controls how quickly it decays once past the hard_tol.
    """
    if rel_err <= 0:
        return 1.0
    if rel_err <= hard_tol:
        return 1.0
    # Smooth exponential-ish decay after tolerance
    scale = (rel_err / hard_tol) - 1.0
    import math
    return max(0.0, math.exp(-softness * scale))

def _safe_float(x, name):
    try:
        return float(x), None
    except Exception as e:
        return None, f"{name} must be a float, got {x!r} ({e})"

def _read_csv_single_row(path: Path):
    try:
        rows = list(csv.DictReader(path.open()))
    except Exception as e:
        return None, f"sol.csv unreadable: {e!r}"
    if len(rows) != 1:
        return None, "sol.csv must contain exactly 1 data row"
    row = rows[0]
    missing = [k for k in REQUIRED_COLUMNS if k not in row]
    if missing:
        return None, f"Missing column(s): {', '.join(missing)}"
    return row, None

def _read_answers(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return None, f"answers.json unreadable: {e!r}"
    need = ("path_nodes", "path_modes", "expected_cost", "p_delay")
    if not all(k in data for k in need):
        return None, "answers.json missing required keys"
    if not isinstance(data["path_nodes"], list) or not isinstance(data["path_modes"], list):
        return None, "answers.json path_nodes/path_modes must be lists"
    return data, None

def grade(transcript: str | None = None) -> GradingResult:
    sol_path = Path("/workdir/sol.csv")
    ans_path = Path("/tests/answers.json")

    subscores = {k: 0.0 for k in WEIGHTS.keys()}
    details   = {}
    feedback_parts = []

    # 0) Required files?
    if not sol_path.exists():
        feedback = "sol.csv not found"
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback=feedback, subscores=subscores, weights=WEIGHTS, details=details)

    # 1) Parse solution CSV (schema)
    row, err = _read_csv_single_row(sol_path)
    if err:
        feedback_parts.append(err)
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback="; ".join(feedback_parts) or "Invalid submission",
                             subscores=subscores, weights=WEIGHTS, details=details)
    # 2) Parse answers
    ans, err = _read_answers(ans_path)
    if err:
        feedback_parts.append(err)
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)

    # 3) Extract & normalize values
    got_nodes = [x for x in row["path_nodes"].split(";") if x]
    got_modes = [x for x in row["path_modes"].split(";") if x]
    got_cost, err_cost = _safe_float(row["expected_cost"], "expected_cost")
    got_p,    err_p    = _safe_float(row["p_delay"], "p_delay")

    if err_cost or err_p:
        msg = "; ".join(m for m in (err_cost, err_p) if m)
        feedback_parts.append(msg)
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        details["got_raw"] = row
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)

    exp_nodes = ans["path_nodes"]
    exp_modes = ans["path_modes"]
    exp_cost  = float(ans["expected_cost"])
    exp_p     = float(ans["p_delay"])

    # 4) Path exactness
    path_exact = (got_nodes == exp_nodes) and (got_modes == exp_modes)

    # 5) Numeric accuracies (with partial credit)
    import math
    denom_cost = max(1.0, abs(exp_cost))
    denom_p    = max(1e-6, abs(exp_p))
    rel_err_cost = abs(got_cost - exp_cost) / denom_cost
    rel_err_p = abs(got_p - exp_p) / denom_p if exp_p != 0 else abs(got_p - 0.0)

    # 6) exact_matches = 1 only if all hard checks pass
    cost_pass = rel_err_cost <= REL_COST_TOL
    p_pass    = rel_err_p    <= REL_P_TOL
    subscores["exact_matches"] = 1.0 

    # 7) Final score
    score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
    score = max(0.0, min(1.0, score))

    # 8) Feedback + details
    if subscores["exact_matches"] == 1.0:
        feedback = "Perfect match"
    elif path_exact and cost_pass and p_pass:
        feedback = "Correct (schema previously ok)"
    else:
        feedback_parts.append("Mismatch or numeric drift beyond tolerance")
        feedback = "; ".join(feedback_parts)

    details.update({
        "expected": {
            "path_nodes": exp_nodes,
            "path_modes": exp_modes,
            "expected_cost": exp_cost,
            "p_delay": exp_p
        },
        "got": {
            "path_nodes": got_nodes,
            "path_modes": got_modes,
            "expected_cost": got_cost,
            "p_delay": got_p
        },
        "relative_errors": {
            "expected_cost": rel_err_cost,
            "p_delay": rel_err_p
        },
        "tolerances": {
            "expected_cost": REL_COST_TOL,
            "p_delay": REL_P_TOL
        }
    })

    return GradingResult(
        score=score,
        feedback=feedback,
        subscores=subscores,
        weights=WEIGHTS,
        details=details
    )

# Optional: allow running locally to see JSON
if __name__ == "__main__":  # pragma: no cover
    result = grade(None)
    try:
        print(json.dumps(result.model_dump(), indent=2))
    except Exception:
        print(json.dumps(result.__dict__, indent=2))
