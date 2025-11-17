import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
from apex_arena._types import GradingResult


def parse_pipeline_file(filepath: Path) -> Tuple[int, int, float, float,
                                                   Dict[str, float],
                                                   Dict[str, Tuple[float, float, float]],
                                                   Dict[Tuple[str, str], float]]:
    """
    Parse the pipeline.txt input file.
    
    File format:
        num_stages num_nodes throughput_req reliability_req
        <stage_id> <complexity>
        ...
        <node_id> <capacity> <reliability> <speed>
        ...
        [blank line]
        <from_node> <to_node> <latency>
        ...
    
    Args:
        filepath: Path to the pipeline.txt file
    
    Returns:
        A tuple containing:
        - num_stages: Number of pipeline stages
        - num_nodes: Number of available nodes
        - throughput_req: Required throughput (events/second)
        - reliability_req: Required end-to-end reliability
        - stages: Dict mapping stage_id to complexity
        - nodes: Dict mapping node_id to (capacity, reliability, speed)
        - network: Dict mapping (node_i, node_j) to latency (ms)
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        if len(parts) != 4:
            raise ValueError(f"First line must be 'num_stages num_nodes throughput reliability', got: {first_line}")
        num_stages, num_nodes, throughput_req, reliability_req = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
        
        # Parse stages
        stages = {}
        for _ in range(num_stages):
            line = f.readline().strip()
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid stage line: {line}")
            stage_id, complexity = parts[0], float(parts[1])
            stages[stage_id] = complexity
        
        # Parse nodes
        nodes = {}
        for _ in range(num_nodes):
            line = f.readline().strip()
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid node line: {line}")
            node_id = parts[0]
            capacity, reliability, speed = float(parts[1]), float(parts[2]), float(parts[3])
            nodes[node_id] = (capacity, reliability, speed)
        
        # Skip blank line
        f.readline()
        
        # Parse network latencies
        network = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid network line: {line}")
            from_node, to_node, latency = parts[0], parts[1], float(parts[2])
            # Store both directions (symmetric)
            network[(from_node, to_node)] = latency
            network[(to_node, from_node)] = latency
        
        return num_stages, num_nodes, throughput_req, reliability_req, stages, nodes, network


def parse_deployment_file(filepath: Path) -> Tuple[float, Dict[str, List[str]]]:
    """
    Parse the deployment.txt output file.
    
    Args:
        filepath: Path to deployment.txt
    
    Returns:
        Tuple of (claimed_latency, deployment_map)
        - claimed_latency: Claimed average latency (ms)
        - deployment_map: Dict mapping stage_id to list of node_ids
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
        try:
            claimed_latency = float(first_line)
        except ValueError:
            raise ValueError(f"First line must be a float (latency), got: '{first_line}'")
        
        deployment_map = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid deployment line (need at least stage and one node): {line}")
            stage_id = parts[0]
            node_ids = parts[1:]
            deployment_map[stage_id] = node_ids
        
        return claimed_latency, deployment_map


def compute_processing_time(stage_complexity: float, node_speed: float) -> float:
    """
    Compute processing time for a stage at a node.
    
    Formula: node_speed * stage_complexity / 10
    
    Args:
        stage_complexity: Stage complexity units
        node_speed: Node processing speed (ms)
    
    Returns:
        Processing time in milliseconds
    """
    return node_speed * stage_complexity / 10.0


def compute_stage_reliability(node_ids: List[str], nodes: Dict[str, Tuple[float, float, float]]) -> float:
    """
    Compute reliability of a stage deployed to one or more nodes.
    
    Single node: reliability = node_reliability
    Multiple nodes (redundancy): reliability = 1 - product(1 - r_i)
    
    Args:
        node_ids: List of nodes where stage is deployed
        nodes: Node specifications
    
    Returns:
        Stage reliability (0.0 to 1.0)
    """
    if not node_ids:
        return 0.0
    
    # Redundancy formula: 1 - product of failure probabilities
    failure_prob = 1.0
    for node_id in node_ids:
        if node_id not in nodes:
            raise ValueError(f"Unknown node: {node_id}")
        node_reliability = nodes[node_id][1]
        failure_prob *= (1.0 - node_reliability)
    
    return 1.0 - failure_prob


def compute_latency_and_validate(
    deployment_map: Dict[str, List[str]],
    stages: Dict[str, float],
    nodes: Dict[str, Tuple[float, float, float]],
    network: Dict[Tuple[str, str], float],
    throughput_req: float,
    reliability_req: float
) -> Tuple[float, List[str]]:
    """
    Compute actual latency and validate all constraints.
    
    This function validates:
    1. All stages are deployed
    2. All referenced nodes exist
    3. No node exceeds capacity
    4. End-to-end reliability meets requirement
    5. Computes actual average latency
    
    Latency calculation:
    - For each path through the pipeline (considering redundancy)
    - Sum processing times + network transfer times
    - Average across all possible paths
    
    Args:
        deployment_map: Stage to nodes mapping
        stages: Stage specifications
        nodes: Node specifications
        network: Network latency matrix
        throughput_req: Required throughput (events/sec)
        reliability_req: Required reliability
    
    Returns:
        Tuple of (actual_latency, errors)
        - actual_latency: Computed average latency in ms
        - errors: List of validation error messages
    """
    errors = []
    
    # Check all stages deployed
    stage_ids = sorted(stages.keys())
    deployed_stages = set(deployment_map.keys())
    missing_stages = set(stage_ids) - deployed_stages
    if missing_stages:
        errors.append(f"Missing deployments for stages: {sorted(missing_stages)}")
        return -1, errors
    
    # Check no extra stages
    extra_stages = deployed_stages - set(stage_ids)
    if extra_stages:
        errors.append(f"Unknown stages in deployment: {sorted(extra_stages)}")
        return -1, errors
    
    # Validate all nodes exist and stages have at least one node
    for stage_id, node_ids in deployment_map.items():
        if not node_ids:
            errors.append(f"Stage {stage_id} has no nodes assigned")
            return -1, errors
        for node_id in node_ids:
            if node_id not in nodes:
                errors.append(f"Unknown node {node_id} in deployment for stage {stage_id}")
                return -1, errors
    
    # Compute node loads and check capacity
    node_loads = {node_id: 0.0 for node_id in nodes}
    
    for stage_id in stage_ids:
        stage_complexity = stages[stage_id]
        node_ids = deployment_map[stage_id]
        
        # For redundant deployment, load is distributed (divide by number of replicas)
        load_per_node = 1.0 / len(node_ids)
        
        for node_id in node_ids:
            node_speed = nodes[node_id][2]
            processing_time = compute_processing_time(stage_complexity, node_speed)
            
            # Load = processing_time (ms) * throughput (events/sec) / 1000 (ms/sec)
            # This gives us concurrent events being processed
            stage_load = processing_time * throughput_req / 1000.0 * load_per_node
            node_loads[node_id] += stage_load
    
    # Check capacity constraints
    for node_id, load in node_loads.items():
        capacity = nodes[node_id][0]
        if load > capacity + 1e-6:  # Small epsilon for floating point
            errors.append(f"Node {node_id} exceeds capacity: load={load:.2f}, capacity={capacity}")
            return -1, errors
    
    # Compute end-to-end reliability
    pipeline_reliability = 1.0
    for stage_id in stage_ids:
        node_ids = deployment_map[stage_id]
        stage_rel = compute_stage_reliability(node_ids, nodes)
        pipeline_reliability *= stage_rel
    
    if pipeline_reliability < reliability_req - 1e-6:
        errors.append(f"Reliability requirement not met: {pipeline_reliability:.6f} < {reliability_req}")
        return -1, errors
    
    # Compute latency
    # For simplicity with redundancy, we compute average latency across all possible paths
    # Each stage has one or more nodes; we consider all combinations
    
    # Generate all possible paths (one node per stage)
    import itertools
    
    path_combinations = []
    for stage_id in stage_ids:
        path_combinations.append(deployment_map[stage_id])
    
    all_paths = list(itertools.product(*path_combinations))
    
    if not all_paths:
        errors.append("No valid paths through pipeline")
        return -1, errors
    
    # Compute latency for each path
    path_latencies = []
    for path in all_paths:
        latency = 0.0
        
        # Processing times
        for i, stage_id in enumerate(stage_ids):
            node_id = path[i]
            stage_complexity = stages[stage_id]
            node_speed = nodes[node_id][2]
            processing_time = compute_processing_time(stage_complexity, node_speed)
            latency += processing_time
        
        # Network transfer times
        for i in range(len(stage_ids) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            # If same node, no network transfer
            if from_node == to_node:
                continue
            
            # Look up network latency
            if (from_node, to_node) not in network:
                errors.append(f"No network latency defined between {from_node} and {to_node}")
                return -1, errors
            
            transfer_latency = network[(from_node, to_node)]
            latency += transfer_latency
        
        path_latencies.append(latency)
    
    # Average latency across all paths (assuming uniform distribution)
    avg_latency = sum(path_latencies) / len(path_latencies)
    
    return avg_latency, errors


def greedy_baseline(
    stages: Dict[str, float],
    nodes: Dict[str, Tuple[float, float, float]],
    network: Dict[Tuple[str, str], float],
    throughput_req: float,
    reliability_req: float
) -> float:
    """
    Sophisticated baseline algorithm for pipeline deployment.
    
    Strategy:
    1. Sort stages by complexity (descending)
    2. For each stage, find best node considering:
       - Available capacity
       - Network latency to previous stage
       - Node speed
       - Load balancing
    3. Add redundancy if needed for reliability
    4. Optimize for co-location when beneficial
    
    Returns:
        Baseline average latency (ms)
    """
    stage_ids = sorted(stages.keys())
    node_ids = sorted(nodes.keys())
    
    # Initialize deployment
    deployment = {stage_id: [] for stage_id in stage_ids}
    node_loads = {node_id: 0.0 for node_id in node_ids}
    
    # Greedy assignment: minimize latency while respecting capacity
    for i, stage_id in enumerate(stage_ids):
        stage_complexity = stages[stage_id]
        
        best_node = None
        best_score = float('inf')
        
        for node_id in node_ids:
            node_capacity, node_reliability, node_speed = nodes[node_id]
            
            # Compute processing time
            processing_time = compute_processing_time(stage_complexity, node_speed)
            
            # Compute load
            stage_load = processing_time * throughput_req / 1000.0
            
            # Check capacity
            if node_loads[node_id] + stage_load > node_capacity:
                continue
            
            # Compute score: processing time + network transfer
            score = processing_time
            
            # Add network transfer cost if not first stage
            if i > 0:
                prev_stage = stage_ids[i - 1]
                prev_nodes = deployment[prev_stage]
                
                if prev_nodes:
                    # If previous stage has nodes, compute min transfer latency
                    min_transfer = float('inf')
                    for prev_node in prev_nodes:
                        if prev_node == node_id:
                            min_transfer = 0  # Co-located
                            break
                        if (prev_node, node_id) in network:
                            min_transfer = min(min_transfer, network[(prev_node, node_id)])
                    
                    if min_transfer != float('inf'):
                        score += min_transfer
            
            # Penalize heavily loaded nodes (load balancing)
            load_penalty = node_loads[node_id] * 0.5
            score += load_penalty
            
            if score < best_score:
                best_score = score
                best_node = node_id
        
        # Assign stage to best node
        if best_node:
            deployment[stage_id].append(best_node)
            stage_load = compute_processing_time(stage_complexity, nodes[best_node][2]) * throughput_req / 1000.0
            node_loads[best_node] += stage_load
        else:
            # No valid node found - fallback to least loaded
            least_loaded = min(node_ids, key=lambda n: node_loads[n])
            deployment[stage_id].append(least_loaded)
            stage_load = compute_processing_time(stage_complexity, nodes[least_loaded][2]) * throughput_req / 1000.0
            node_loads[least_loaded] += stage_load
    
    # Check reliability and add redundancy if needed
    pipeline_reliability = 1.0
    for stage_id in stage_ids:
        stage_rel = compute_stage_reliability(deployment[stage_id], nodes)
        pipeline_reliability *= stage_rel
    
    # If reliability too low, add redundancy to weakest stages
    if pipeline_reliability < reliability_req:
        # Find stages with lowest reliability
        stage_rels = []
        for stage_id in stage_ids:
            stage_rel = compute_stage_reliability(deployment[stage_id], nodes)
            stage_rels.append((stage_rel, stage_id))
        
        stage_rels.sort()
        
        # Add redundancy to weakest stages
        for stage_rel, stage_id in stage_rels:
            if pipeline_reliability >= reliability_req:
                break
            
            # Find a reliable node with capacity
            stage_complexity = stages[stage_id]
            
            for node_id in sorted(node_ids, key=lambda n: nodes[n][1], reverse=True):
                if node_id in deployment[stage_id]:
                    continue
                
                node_capacity, node_reliability, node_speed = nodes[node_id]
                processing_time = compute_processing_time(stage_complexity, node_speed)
                
                # Load distributed across replicas
                current_replicas = len(deployment[stage_id])
                new_load_per_replica = processing_time * throughput_req / 1000.0 / (current_replicas + 1)
                
                # Check if we can add this replica
                can_add = True
                
                # Check new node capacity
                if node_loads[node_id] + new_load_per_replica > node_capacity:
                    can_add = False
                
                # Check existing nodes still have capacity after redistribution
                for existing_node in deployment[stage_id]:
                    old_load = processing_time * throughput_req / 1000.0 / current_replicas
                    if node_loads[existing_node] - old_load + new_load_per_replica > nodes[existing_node][0]:
                        can_add = False
                        break
                
                if can_add:
                    # Add redundancy
                    deployment[stage_id].append(node_id)
                    
                    # Update loads
                    node_loads[node_id] += new_load_per_replica
                    for existing_node in deployment[stage_id][:-1]:
                        old_load = processing_time * throughput_req / 1000.0 / current_replicas
                        node_loads[existing_node] = node_loads[existing_node] - old_load + new_load_per_replica
                    
                    # Recompute pipeline reliability
                    pipeline_reliability = 1.0
                    for sid in stage_ids:
                        stage_rel = compute_stage_reliability(deployment[sid], nodes)
                        pipeline_reliability *= stage_rel
                    
                    break
    
    # Compute latency
    latency, _ = compute_latency_and_validate(
        deployment, stages, nodes, network, throughput_req, reliability_req
    )
    
    return latency if latency != -1 else float('inf')


def grade(transcript: str) -> GradingResult:
    """
    Grade the stream processing pipeline deployment.
    
    GRADING CRITERIA (all must pass):
    1. Files exist: deployment.txt, data/pipeline.txt
    2. Deployment format is valid (first line is latency float)
    3. All stages are deployed to at least one node
    4. All nodes are valid (exist in input)
    5. No node exceeds capacity
    6. End-to-end reliability meets requirement
    7. Claimed latency matches computed latency
    8. Solution is near-optimal: latency ≤ 1.3 × baseline
    """
    feedback_messages: List[str] = []
    subscores = {"correct_answer": 0.0}
    weights = {"correct_answer": 1.0}

    try:
        deployment_path = Path("/workdir/deployment.txt")
        data_path = Path("/workdir/data/pipeline.txt")

        # ===== CHECK 1: Files exist =====
        for p, name in [(data_path, "Input"), (deployment_path, "Deployment")]:
            if not p.exists():
                feedback_messages.append(f"FAIL: {name} file {p} does not exist")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        # Parse input
        num_stages, num_nodes, throughput_req, reliability_req, stages, nodes, network = \
            parse_pipeline_file(data_path)
        
        feedback_messages.append(f"Problem: {num_stages} stages, {num_nodes} nodes, throughput={throughput_req} events/sec, reliability≥{reliability_req}")

        # ===== CHECK 2: Parse deployment =====
        try:
            claimed_latency, deployment_map = parse_deployment_file(deployment_path)
            feedback_messages.append(f"Claimed latency: {claimed_latency:.2f} ms")
        except ValueError as e:
            feedback_messages.append(f"FAIL: Invalid deployment format - {e}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        except Exception as e:
            feedback_messages.append(f"FAIL: Error parsing deployment.txt - {e}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # ===== CHECK 3: Validate deployment and compute latency =====
        actual_latency, errors = compute_latency_and_validate(
            deployment_map, stages, nodes, network, throughput_req, reliability_req
        )
        
        if errors:
            feedback_messages.append(f"FAIL: Deployment validation errors:")
            feedback_messages.extend(errors)
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        
        feedback_messages.append(f"Deployment valid: all constraints satisfied")
        feedback_messages.append(f"Actual computed latency: {actual_latency:.2f} ms")

        # ===== CHECK 4: Claimed vs actual match =====
        latency_diff = abs(claimed_latency - actual_latency)
        if latency_diff > 0.1:  # Allow 0.1ms tolerance for floating point
            feedback_messages.append(f"FAIL: Claimed latency ({claimed_latency:.2f}) != actual latency ({actual_latency:.2f}), diff={latency_diff:.2f}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        
        feedback_messages.append("Claimed latency matches computed latency")

        # ===== CHECK 5: Near-optimality check =====
        baseline_latency = greedy_baseline(
            stages, nodes, network, throughput_req, reliability_req
        )
        feedback_messages.append(f"Baseline latency: {baseline_latency:.2f} ms")

        # Tolerance: 1.3x
        try:
            tol_env = os.getenv("GRADER_NEAR_OPT_TOL")
            tolerance = float(tol_env) if tol_env is not None else 1.3
            if tolerance <= 1.0:
                tolerance = 1.3
        except Exception:
            tolerance = 1.3

        feedback_messages.append(f"Near-optimality tolerance: {tolerance:.2f}x baseline (solution must be ≤ {baseline_latency * tolerance:.1f} ms)")

        # Check near-optimality
        if baseline_latency == float('inf'):
            feedback_messages.append("WARNING: Baseline could not find valid solution")
            # If baseline fails but solution succeeds, accept it
        else:
            threshold = baseline_latency * tolerance
            if actual_latency > threshold:
                feedback_messages.append(
                    f"FAIL: Not near-optimal: {actual_latency:.2f} > {tolerance:.2f} × {baseline_latency:.2f} = {threshold:.1f}"
                )
                feedback_messages.append(f"Your solution is {(actual_latency/baseline_latency):.2f}x baseline (needs to be ≤ {tolerance:.2f}x)")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        feedback_messages.append(f"Solution is near-optimal ({(actual_latency/baseline_latency):.3f}x baseline ≤ {tolerance:.2f}x)")

        # ===== SUCCESS! =====
        subscores["correct_answer"] = 1.0
        feedback_messages.append("ALL CHECKS PASSED - Solution is valid and near-optimal!")
        return GradingResult(score=1.0, subscores=subscores, weights=weights,
                             feedback="; ".join(feedback_messages))

    except Exception as e:
        import traceback
        feedback_messages.append(f"UNEXPECTED ERROR: {e}")
        feedback_messages.append(traceback.format_exc())
        return GradingResult(score=0.0, subscores=subscores, weights=weights,
                             feedback="; ".join(feedback_messages))