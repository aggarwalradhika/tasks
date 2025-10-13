import os
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Set
import collections

from apex_arena._types import GradingResult


def parse_requests_file(filepath: Path) -> Tuple[int, int, int, Dict[str, List[Tuple[str, int, int]]], List[Tuple[str, str]]]:
    """
    Parse the requests.txt input file with dependencies.

    File format:
        n m d
        <request_id> <priority> <GPU_id> <execution_time>
        ...
        [blank line]
        <req_A> <req_B>  (A must complete before B)
        ...

    Returns:
        A tuple (n_gpus, n_requests, n_deps, requests, dependencies) where:
        - requests maps request_id -> list of (gpu_id, exec_time, priority)
        - dependencies is list of (prerequisite, dependent) tuples
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("Empty file")
        parts = first_line.split()
        if len(parts) != 3:
            raise ValueError(f"First line must be 'n m d', got: {first_line}")
        n, m, d = map(int, parts)
        
        requests = collections.defaultdict(list)
        
        # Parse request options
        for line in f:
            line = line.strip()
            if not line:
                break  # blank line separates requests from dependencies
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid line format: {line}")
            req_id, prio_s, gpu, t_s = parts
            try:
                prio = int(prio_s)
                t = int(t_s)
            except ValueError:
                raise ValueError(f"Invalid priority/time on line: {line}")
            
            # Validate constraints
            if prio not in {1, 2, 3, 4, 5}:
                raise ValueError(f"Priority must be 1-5, got {prio} for request {req_id}")
            if not (1 <= t <= 200):
                raise ValueError(f"Execution time must be 1-200, got {t} for request {req_id} on {gpu}")
            
            requests[req_id].append((gpu, t, prio))
        
        # Parse dependencies
        dependencies = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid dependency line: {line}")
            prereq, dependent = parts
            dependencies.append((prereq, dependent))
        
        # Validate priorities are consistent per-request
        for req_id, options in requests.items():
            priorities = {p for _, _, p in options}
            if len(priorities) != 1:
                raise ValueError(f"Inconsistent priorities for request {req_id}: {priorities}")
        
        # Validate dependency count
        if len(dependencies) != d:
            raise ValueError(f"Expected {d} dependencies, got {len(dependencies)}")
        
        return n, m, d, dict(requests), dependencies


def validate_dependencies_dag(dependencies: List[Tuple[str, str]], all_requests: Set[str]) -> Tuple[bool, str]:
    """
    Validate that dependencies form a DAG (no cycles).
    
    Returns:
        (is_valid, error_message)
    """
    # Check all dependency requests exist
    dep_requests = set()
    for prereq, dependent in dependencies:
        dep_requests.add(prereq)
        dep_requests.add(dependent)
    
    unknown = dep_requests - all_requests
    if unknown:
        return False, f"Unknown requests in dependencies: {sorted(unknown)}"
    
    # Build adjacency list
    graph = collections.defaultdict(list)
    for prereq, dependent in dependencies:
        graph[prereq].append(dependent)
    
    # Check for cycles using DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {req: WHITE for req in all_requests}
    
    def has_cycle(node):
        if color[node] == GRAY:
            return True  # back edge = cycle
        if color[node] == BLACK:
            return False
        
        color[node] = GRAY
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        color[node] = BLACK
        return False
    
    for req in all_requests:
        if color[req] == WHITE:
            if has_cycle(req):
                return False, "Dependencies contain a cycle"
    
    return True, ""


def compute_completion_times(schedule_lines: List[Tuple[str, str]],
                             requests: Dict[str, List[Tuple[str, int, int]]],
                             dependencies: List[Tuple[str, str]],
                             n_gpus: int) -> Dict[str, Tuple[int, int, int]]:
    """
    Compute completion times for each request respecting dependencies.
    
    Returns:
        Dict mapping request_id -> (completion_time, priority, weighted_completion_time)
    """
    # Build dependency graph
    prereqs = collections.defaultdict(set)  # dependent -> set of prerequisites
    for prereq, dependent in dependencies:
        prereqs[dependent].add(prereq)
    
    # Build per-GPU queues preserving order in schedule_lines
    gpu_queues: Dict[str, List[Tuple[str, int, int]]] = {f"GPU{i+1}": [] for i in range(n_gpus)}
    request_to_gpu = {}
    
    for req_id, assigned_gpu in schedule_lines:
        exec_time = None
        prio = None
        for gpu_id, et, p in requests[req_id]:
            if gpu_id == assigned_gpu:
                exec_time = et
                prio = p
                break
        if exec_time is None:
            raise ValueError(f"Cannot find execution time for {req_id} on {assigned_gpu}")
        gpu_queues[assigned_gpu].append((req_id, exec_time, prio))
        request_to_gpu[req_id] = assigned_gpu
    
    # Track GPU availability and request completion times
    gpu_available = {f"GPU{i+1}": 0 for i in range(n_gpus)}
    completion_times = {}
    
    # Process requests in schedule order, respecting dependencies
    for req_id, assigned_gpu in schedule_lines:
        # Find exec time and priority
        exec_time = None
        prio = None
        for gpu_id, et, p in requests[req_id]:
            if gpu_id == assigned_gpu:
                exec_time = et
                prio = p
                break
        
        # Compute earliest start time
        # Must wait for: (1) GPU to be available, (2) all prerequisites to complete
        earliest_start = gpu_available[assigned_gpu]
        
        if req_id in prereqs:
            for prereq_id in prereqs[req_id]:
                if prereq_id not in completion_times:
                    raise ValueError(f"Dependency violation: {prereq_id} must complete before {req_id}, but appears later in schedule")
                prereq_completion = completion_times[prereq_id][0]
                earliest_start = max(earliest_start, prereq_completion)
        
        # Schedule the request
        completion_time = earliest_start + exec_time
        weighted_ct = prio * completion_time
        
        completion_times[req_id] = (completion_time, prio, weighted_ct)
        gpu_available[assigned_gpu] = completion_time
    
    return completion_times


def compute_schedule_score(schedule_lines: List[Tuple[str, str]],
                          requests: Dict[str, List[Tuple[str, int, int]]],
                          dependencies: List[Tuple[str, str]],
                          n_gpus: int) -> int:
    """
    Compute maximum weighted completion time from schedule with dependencies.
    
    Returns:
        int: maximum weighted completion time
    """
    completion_times = compute_completion_times(schedule_lines, requests, dependencies, n_gpus)
    return max(wct for _, _, wct in completion_times.values())


def topological_sort(requests: Set[str], dependencies: List[Tuple[str, str]]) -> List[str]:
    """
    Return a topological ordering of requests respecting dependencies.
    Uses Kahn's algorithm.
    """
    # Build graph
    graph = collections.defaultdict(list)
    in_degree = {req: 0 for req in requests}
    
    for prereq, dependent in dependencies:
        graph[prereq].append(dependent)
        in_degree[dependent] += 1
    
    # Find all nodes with no incoming edges
    queue = [req for req in requests if in_degree[req] == 0]
    queue.sort()  # deterministic ordering
    
    result = []
    while queue:
        # Pick node with smallest ID for determinism
        node = queue.pop(0)
        result.append(node)
        
        # Remove edges from this node
        for neighbor in sorted(graph[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort()
    
    if len(result) != len(requests):
        raise ValueError("Cycle detected in dependencies during topological sort")
    
    return result


def sophisticated_baseline(requests: Dict[str, List[Tuple[str, int, int]]],
                          dependencies: List[Tuple[str, str]],
                          n_gpus: int) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Sophisticated baseline using:
    1. Topological ordering respecting dependencies
    2. Critical path analysis
    3. List scheduling with priority ratio heuristic
    
    This is much better than simple greedy and should be hard to beat.
    """
    all_requests = set(requests.keys())
    
    # Get topological order
    topo_order = topological_sort(all_requests, dependencies)
    
    # Compute priorities and min execution times
    req_info = {}
    for req_id in all_requests:
        priority = requests[req_id][0][2]
        min_time = min(et for _, et, _ in requests[req_id])
        avg_time = sum(et for _, et, _ in requests[req_id]) / len(requests[req_id])
        req_info[req_id] = (priority, min_time, avg_time)
    
    # Sort by: follow topo order, but within valid ordering use priority/min_time ratio
    # This is a sophisticated heuristic
    def priority_key(req_id):
        prio, min_t, avg_t = req_info[req_id]
        return (-prio / min_t, min_t, req_id)  # Higher priority/time ratio first
    
    # Build dependency graph for checking
    prereqs = collections.defaultdict(set)
    for prereq, dependent in dependencies:
        prereqs[dependent].add(prereq)
    
    # Schedule requests respecting topological constraints
    scheduled = set()
    schedule_lines = []
    gpu_available = {f"GPU{i+1}": 0 for i in range(n_gpus)}
    completion_times = {}
    
    # Use a priority queue of ready tasks
    ready = []
    for req_id in all_requests:
        if req_id not in prereqs or len(prereqs[req_id]) == 0:
            ready.append(req_id)
    
    ready.sort(key=priority_key)
    
    while ready:
        # Pick best ready task
        req_id = ready.pop(0)
        prio, min_time, avg_time = req_info[req_id]
        
        # Find best GPU considering dependencies
        best_gpu = None
        best_wct = float('inf')
        best_exec_time = None
        
        for i in range(n_gpus):
            gpu_id = f"GPU{i+1}"
            
            # Find exec time on this GPU
            exec_time = None
            for g, et, p in requests[req_id]:
                if g == gpu_id:
                    exec_time = et
                    break
            
            if exec_time is None:
                continue
            
            # Compute start time considering dependencies
            earliest_start = gpu_available[gpu_id]
            if req_id in prereqs:
                for prereq_id in prereqs[req_id]:
                    if prereq_id in completion_times:
                        earliest_start = max(earliest_start, completion_times[prereq_id])
            
            completion = earliest_start + exec_time
            wct = prio * completion
            
            if wct < best_wct:
                best_wct = wct
                best_gpu = gpu_id
                best_exec_time = exec_time
        
        if best_gpu is None:
            raise ValueError(f"No available GPU for {req_id}")
        
        # Schedule it
        earliest_start = gpu_available[best_gpu]
        if req_id in prereqs:
            for prereq_id in prereqs[req_id]:
                if prereq_id in completion_times:
                    earliest_start = max(earliest_start, completion_times[prereq_id])
        
        completion = earliest_start + best_exec_time
        schedule_lines.append((req_id, best_gpu))
        gpu_available[best_gpu] = completion
        completion_times[req_id] = completion
        scheduled.add(req_id)
        
        # Add newly ready tasks
        for other_req in all_requests:
            if other_req not in scheduled and other_req not in ready:
                if other_req in prereqs:
                    if all(p in scheduled for p in prereqs[other_req]):
                        ready.append(other_req)
        
        ready.sort(key=priority_key)
    
    score = compute_schedule_score(schedule_lines, requests, dependencies, n_gpus)
    return score, schedule_lines


def grade(transcript: str) -> GradingResult:
    """
    Grade the GPU scheduling submission with dependencies.
    
    This grader is significantly more sophisticated and uses a tighter tolerance (1.15x).
    """
    feedback_messages: List[str] = []
    subscores = {"correct_answer": 0.0}
    weights = {"correct_answer": 1.0}

    try:
        schedule_path = Path("/workdir/schedule.txt")
        answer_path = Path("/workdir/ans.txt")
        data_path = Path("/workdir/data/requests.txt")

        # Check files
        for p, name in [(data_path, "Input"), (schedule_path, "Schedule"), (answer_path, "Answer")]:
            if not p.exists():
                feedback_messages.append(f"{name} file {p} does not exist")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        # Parse input
        n_gpus, n_requests, n_deps, requests, dependencies = parse_requests_file(data_path)
        feedback_messages.append(f"Problem: {n_requests} requests, {n_gpus} GPUs, {n_deps} dependencies")

        # Validate dependencies form DAG
        is_dag, err = validate_dependencies_dag(dependencies, set(requests.keys()))
        if not is_dag:
            feedback_messages.append(f"Invalid dependencies: {err}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # Parse schedule
        schedule_lines = []
        seen = set()
        with open(schedule_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    feedback_messages.append(f"Invalid schedule line: {line}")
                    return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                         feedback="; ".join(feedback_messages))
                req_id, gpu_id = parts
                if req_id in seen:
                    feedback_messages.append(f"Duplicate request in schedule: {req_id}")
                    return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                         feedback="; ".join(feedback_messages))
                seen.add(req_id)
                schedule_lines.append((req_id, gpu_id))

        feedback_messages.append(f"Schedule contains {len(schedule_lines)} assignments")

        # Validate schedule
        if len(schedule_lines) != n_requests:
            feedback_messages.append(f"Schedule has {len(schedule_lines)} assignments, expected {n_requests}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # Check all requests scheduled
        scheduled_requests = {req_id for req_id, _ in schedule_lines}
        all_requests = set(requests.keys())
        if scheduled_requests != all_requests:
            missing = all_requests - scheduled_requests
            extra = scheduled_requests - all_requests
            msg = []
            if missing:
                msg.append(f"Missing: {sorted(missing)}")
            if extra:
                msg.append(f"Unknown: {sorted(extra)}")
            feedback_messages.append("; ".join(msg))
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # Validate GPU assignments
        valid_gpus = {f"GPU{i+1}" for i in range(n_gpus)}
        for req_id, assigned_gpu in schedule_lines:
            if assigned_gpu not in valid_gpus:
                feedback_messages.append(f"Invalid GPU '{assigned_gpu}' for {req_id}")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))
            available_gpus = {gpu for gpu, _, _ in requests[req_id]}
            if assigned_gpu not in available_gpus:
                feedback_messages.append(f"GPU '{assigned_gpu}' not available for {req_id}")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        feedback_messages.append("Schedule is feasible")

        # Parse claimed answer
        try:
            with open(answer_path, "r") as f:
                claimed_answer = int(f.read().strip())
            feedback_messages.append(f"Claimed answer: {claimed_answer}")
        except Exception as e:
            feedback_messages.append(f"Invalid ans.txt: {e}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # Compute actual score
        try:
            actual_score = compute_schedule_score(schedule_lines, requests, dependencies, n_gpus)
            feedback_messages.append(f"Actual computed score: {actual_score}")
        except ValueError as e:
            feedback_messages.append(f"Dependency violation in schedule: {e}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # Check claimed vs actual
        if claimed_answer != actual_score:
            feedback_messages.append(f"Claimed ({claimed_answer}) != actual ({actual_score})")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # Sophisticated baseline
        baseline_score, baseline_schedule = sophisticated_baseline(requests, dependencies, n_gpus)
        feedback_messages.append(f"Sophisticated baseline score: {baseline_score}")

        # Tolerance: TIGHTENED to 1.15
        try:
            tol_env = os.getenv("GRADER_NEAR_OPT_TOL")
            tolerance = float(tol_env) if tol_env is not None else 1.15
            if tolerance <= 1.0:
                tolerance = 1.15
        except Exception:
            tolerance = 1.15

        feedback_messages.append(f"Near-optimality tolerance: {tolerance:.2f}x (tighter than before!)")

        # Check near-optimality
        if baseline_score == 0:
            if actual_score != 0:
                feedback_messages.append("Baseline is 0 but actual > 0")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))
        else:
            if actual_score > baseline_score * tolerance:
                feedback_messages.append(
                    f"Not near-optimal: {actual_score} > {tolerance:.2f} × {baseline_score} = {baseline_score * tolerance:.1f}"
                )
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        # Success!
        subscores["correct_answer"] = 1.0
        feedback_messages.append("Solution is valid and near-optimal!")
        return GradingResult(score=1.0, subscores=subscores, weights=weights,
                             feedback="; ".join(feedback_messages))

    except Exception as e:
        import traceback
        feedback_messages.append(f"Error: {e}")
        feedback_messages.append(traceback.format_exc())
        return GradingResult(score=0.0, subscores=subscores, weights=weights,
                             feedback="; ".join(feedback_messages))