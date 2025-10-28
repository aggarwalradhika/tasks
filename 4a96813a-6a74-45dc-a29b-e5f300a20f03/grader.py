# grader.py
import csv
import json
import math
import random
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

# Weight: single component - all must pass
WEIGHTS = {
    "exact_matches": 1.0    # 1.0 only if all validations pass
}

# Tolerances for stochastic optimization
# As specified in task.yaml:
# - 5% relative error tolerance for expected_cost
# - 12% relative error tolerance for p_delay (accounts for MC variance with 1500 samples)
REL_COST_TOL = 0.05      # 5% relative error tolerance
REL_P_TOL = 0.12         # 12% relative error tolerance

def lognormal_params(mean, cv):
    """
    Convert mean & CoV to lognormal (mu, sigma) parameters.
    
    For a lognormal distribution:
    - mu = log(mean) - 0.5 * sigma^2
    - sigma = sqrt(log(1 + CV^2))
    """
    var = (cv * mean) ** 2
    sigma2 = math.log(1 + var / (mean ** 2))
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, sigma

def validate_params_schema(params):
    """
    Validate that params.json contains all required fields.
    Returns (is_valid, error_msg)
    """
    required_params = [
        "origin", "destination", "deadline_h", "penalty_lambda",
        "fuel_price_usd_per_litre", "congestion_index", 
        "weather_disruption_prob", "mc_samples", "rng_seed"
    ]
    
    missing = [p for p in required_params if p not in params]
    if missing:
        return False, f"Missing required parameters in params.json: {', '.join(missing)}"
    
    return True, None

def validate_graph_schema(graph_data):
    """
    Validate that graph.json has expected structure.
    Returns (is_valid, error_msg)
    """
    required_edge_fields = [
        "u", "v", "mode", "distance_km", "base_time_mean_h", 
        "base_time_cv", "fuel_rate_l_per_km", "handling_fee",
        "congestion_sensitivity", "weather_sensitivity"
    ]
    
    if "nodes" not in graph_data:
        return False, "Graph must have 'nodes' field"
    
    if "edges" not in graph_data:
        return False, "Graph must have 'edges' field"
    
    if not isinstance(graph_data["nodes"], list):
        return False, "'nodes' must be a list"
    
    if not isinstance(graph_data["edges"], list):
        return False, "'edges' must be a list"
    
    for i, edge in enumerate(graph_data["edges"]):
        missing = [f for f in required_edge_fields if f not in edge]
        if missing:
            return False, f"Edge {i} missing required fields: {', '.join(missing)}"
    
    return True, None

def validate_path_in_graph(nodes, modes, graph_data):
    """
    Validate that the path exists in the graph with correct modes.
    
    Checks:
    - Path has at least 2 nodes
    - Number of modes equals number of edges
    - Each edge exists in the graph
    
    Returns (is_valid, edges_list, error_msg)
    """
    if len(nodes) < 2:
        return False, None, "Path must have at least 2 nodes"
    
    if len(modes) != len(nodes) - 1:
        return False, None, f"Number of modes ({len(modes)}) must equal number of edges ({len(nodes)-1})"
    
    # Build edge lookup: (u, v, mode) -> edge_data
    edge_map = {}
    for edge in graph_data["edges"]:
        key = (edge["u"], edge["v"], edge["mode"])
        edge_map[key] = edge
    
    # Validate each edge in the path
    edges = []
    for i in range(len(nodes) - 1):
        u, v, mode = nodes[i], nodes[i+1], modes[i]
        key = (u, v, mode)
        
        if key not in edge_map:
            return False, None, f"Edge from {u} to {v} via {mode} does not exist in graph"
        
        edges.append(edge_map[key])
    
    return True, edges, None

def compute_expected_cost_and_delay(edges, params):
    """
    Recompute the expected cost and delay probability using Monte Carlo simulation.
    
    Uses the same methodology as required in task.yaml:
    - Fixed random seed from params
    - Exact number of mc_samples
    - Lognormal distribution for travel times
    - Quadratic penalty for deadline violations
    
    Returns (expected_cost, p_delay)
    """
    fuel_price = params["fuel_price_usd_per_litre"]
    C = params["congestion_index"]
    W = params["weather_disruption_prob"]
    lam = params["penalty_lambda"]
    deadline = params["deadline_h"]
    mc_samples = params["mc_samples"]
    rng_seed = params["rng_seed"]
    
    # Reset random seed for reproducibility
    random.seed(rng_seed)
    
    # Calculate deterministic transport cost
    transport_cost = 0.0
    for e in edges:
        fuel = e["fuel_rate_l_per_km"] * e["distance_km"] * fuel_price
        transport_cost += e["handling_fee"] + fuel
    
    # Prepare lognormal parameters for each edge
    edge_params = []
    for e in edges:
        base_time = e["base_time_mean_h"]
        # Adjust mean for congestion and weather
        adjusted_mean = base_time * (1 + e["congestion_sensitivity"] * C) * (1 + e["weather_sensitivity"] * W)
        mu, sigma = lognormal_params(adjusted_mean, e["base_time_cv"])
        edge_params.append((mu, sigma))
    
    # Monte Carlo simulation
    sq_excess = 0.0
    delay_count = 0
    
    for _ in range(mc_samples):
        total_time = 0.0
        for (mu, sigma) in edge_params:
            total_time += math.exp(random.gauss(mu, sigma))
        
        if total_time > deadline:
            delay_count += 1
        
        excess = max(0.0, total_time - deadline)
        sq_excess += excess * excess
    
    p_delay = delay_count / mc_samples
    expected_penalty = lam * (sq_excess / mc_samples)
    expected_total = transport_cost + expected_penalty
    
    return expected_total, p_delay

def _safe_float(x, name):
    """Convert value to float with error handling."""
    try:
        return float(x), None
    except Exception as e:
        return None, f"{name} must be a float, got {x!r} ({e})"

def _read_csv_single_row(path: Path):
    """Read and validate CSV has exactly one data row with required columns."""
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

def grade(transcript: str | None = None) -> GradingResult:
    """
    Grade the solution for the multi-modal logistics optimization task.
    
    Validates:
    1. File existence and format
    2. Parameter schema completeness
    3. Graph schema correctness
    4. Path validity (connectivity, origin, destination)
    5. Cost accuracy (within 5% tolerance)
    6. Delay probability accuracy (within 12% tolerance)
    """
    sol_path = Path("/workdir/sol.csv")
    graph_path = Path("/workdir/data/graph.json")
    params_path = Path("/workdir/data/params.json")

    subscores = {k: 0.0 for k in WEIGHTS.keys()}
    details = {}
    feedback_parts = []

    # 0) Check required files exist
    if not sol_path.exists():
        feedback = "sol.csv not found"
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback=feedback, subscores=subscores, weights=WEIGHTS, details=details)

    # Load graph and params
    try:
        with open(graph_path) as f:
            graph_data = json.load(f)
        with open(params_path) as f:
            params = json.load(f)
    except Exception as e:
        feedback = f"Failed to load graph or params: {e}"
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback=feedback, subscores=subscores, weights=WEIGHTS, details=details)

    # 1) Validate params schema
    params_valid, params_err = validate_params_schema(params)
    if not params_valid:
        feedback_parts.append(params_err)
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)

    # 2) Validate graph schema
    graph_valid, graph_err = validate_graph_schema(graph_data)
    if not graph_valid:
        feedback_parts.append(graph_err)
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)

    # 3) Parse solution CSV
    row, err = _read_csv_single_row(sol_path)
    if err:
        feedback_parts.append(err)
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)

    # 4) Extract and validate values
    got_nodes = [x.strip() for x in row["path_nodes"].split(";") if x.strip()]
    got_modes = [x.strip() for x in row["path_modes"].split(";") if x.strip()]
    got_cost, err_cost = _safe_float(row["expected_cost"], "expected_cost")
    got_p, err_p = _safe_float(row["p_delay"], "p_delay")

    if err_cost or err_p:
        msg = "; ".join(m for m in (err_cost, err_p) if m)
        feedback_parts.append(msg)
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        details["got_raw"] = row
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)

    # 5) Validate path connectivity
    path_valid, edges, path_err = validate_path_in_graph(got_nodes, got_modes, graph_data)
    
    if not path_valid:
        feedback_parts.append(f"Invalid path: {path_err}")
        subscores["exact_matches"] = 0.0
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)

    # 6) Validate origin and destination
    if got_nodes[0] != params["origin"]:
        feedback_parts.append(f"Path must start at {params['origin']}, got {got_nodes[0]}")
        subscores["exact_matches"] = 0.0
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)
    
    if got_nodes[-1] != params["destination"]:
        feedback_parts.append(f"Path must end at {params['destination']}, got {got_nodes[-1]}")
        subscores["exact_matches"] = 0.0
        score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
        return GradingResult(score=score, feedback="; ".join(feedback_parts),
                             subscores=subscores, weights=WEIGHTS, details=details)

    # 7) Recompute expected cost and delay probability
    try:
        expected_cost, p_delay = compute_expected_cost_and_delay(edges, params)
        
        # Calculate relative errors as specified in task.yaml
        denom_cost = max(1.0, abs(expected_cost))
        denom_p = max(0.01, abs(p_delay)) if p_delay > 0 else 1.0
        
        rel_err_cost = abs(got_cost - expected_cost) / denom_cost
        rel_err_p = abs(got_p - p_delay) / denom_p
        
        # Check if both pass tolerances
        cost_pass = rel_err_cost <= REL_COST_TOL
        p_pass = rel_err_p <= REL_P_TOL
        
        details.update({
            "recomputed": {
                "expected_cost": expected_cost,
                "p_delay": p_delay
            },
            "submitted": {
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
            },
            "passes": {
                "path_valid": True,
                "origin_correct": True,
                "destination_correct": True,
                "cost_pass": cost_pass,
                "p_delay_pass": p_pass
            }
        })
        
        # All must pass for exact_matches = 1.0
        if cost_pass and p_pass:
            subscores["exact_matches"] = 1.0
            feedback_parts.append("All validations passed")
        else:
            subscores["exact_matches"] = 0.0
            if not cost_pass:
                feedback_parts.append(
                    f"Cost error {rel_err_cost:.2%} exceeds tolerance {REL_COST_TOL:.2%} "
                    f"(submitted: {got_cost:.2f}, expected: {expected_cost:.2f})"
                )
            if not p_pass:
                feedback_parts.append(
                    f"Delay probability error {rel_err_p:.2%} exceeds tolerance {REL_P_TOL:.2%} "
                    f"(submitted: {got_p:.4f}, expected: {p_delay:.4f})"
                )
                
    except Exception as e:
        feedback_parts.append(f"Error during validation: {e}")
        subscores["exact_matches"] = 0.0
        details["validation_error"] = str(e)

    # 8) Calculate final score
    score = sum(WEIGHTS[k] * subscores[k] for k in WEIGHTS)
    score = max(0.0, min(1.0, score))

    # 9) Generate feedback
    if not feedback_parts:
        feedback = "Solution validated successfully"
    else:
        feedback = "; ".join(feedback_parts)

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