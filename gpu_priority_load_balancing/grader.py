import os
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
import collections

from apex_arena._types import GradingResult


def parse_requests_file(filepath: Path) -> Tuple[int, int, Dict[str, List[Tuple[str, int, int]]]]:
    """
    Parse the requests.txt input file.

    File format:
        n m
        <request_id> <priority> <GPU_id> <execution_time>
        ...

    Notes:
        - Each request appears exactly n times (one per GPU) in the canonical input,
          but the parser is tolerant as long as each request has >=1 option.
        - This function validates that priority is *per request* (the same across its options).
          If priorities differ between options for the same request, a ValueError is raised.

    Args:
        filepath: Path to the input file.

    Returns:
        A tuple (n_gpus, n_requests, requests) where `requests` maps request_id ->
        list of tuples (gpu_id, exec_time, priority).
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("Empty file")
        n, m = map(int, first_line.split())
        requests = collections.defaultdict(list)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid line format: {line}")
            req_id, prio_s, gpu, t_s = parts
            try:
                prio = int(prio_s)
                t = int(t_s)
            except ValueError:
                raise ValueError(f"Invalid priority/time on line: {line}")
            requests[req_id].append((gpu, t, prio))

    # Validate priorities are consistent per-request (priority is a request attribute)
    for req_id, options in requests.items():
        priorities = {p for _, _, p in options}
        if len(priorities) != 1:
            raise ValueError(f"Inconsistent priorities for request {req_id}: {priorities}")
    return n, m, dict(requests)


def parse_schedule_file(filepath: Path) -> List[Tuple[str, str]]:
    """
    Parse the schedule file.

    Expected line format:
        <request_id> <GPU_id>

    IMPORTANT: The grader *interprets the order of the lines in schedule.txt*
    as the execution order (first occurrence on a GPU executes first). This
    resolves the ambiguity of ordering: students must write schedule lines in
    the order they expect their tasks to execute on each GPU.

    Args:
        filepath: Path to the schedule file.

    Returns:
        List of (request_id, gpu_id) in the order appearing in the file.

    Raises:
        ValueError: on duplicate request lines (same request repeated).
    """
    schedule_lines: List[Tuple[str, str]] = []
    seen = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid schedule line format: {line}")
            req_id, gpu_id = parts
            if req_id in seen:
                raise ValueError(f"Duplicate request in schedule: {req_id}")
            seen.add(req_id)
            schedule_lines.append((req_id, gpu_id))
    return schedule_lines


def validate_schedule(schedule_lines: List[Tuple[str, str]],
                      requests: Dict[str, List[Tuple[str, int, int]]],
                      n_gpus: int) -> Tuple[bool, str]:
    """
    Validate schedule assignments.

    Checks:
      - Every request in `requests` appears exactly once in schedule_lines.
      - Assigned GPU is valid (GPU1..GPUn).
      - Assigned GPU is among the available options for that request.

    Args:
        schedule_lines: list of (request_id, assigned_gpu) in execution order.
        requests: mapping request_id -> [(gpu_id, exec_time, priority), ...]
        n_gpus: number of GPUs.

    Returns:
        (is_valid, error_message). If valid, error_message == "".
    """
    schedule_ids = [req_id for req_id, _ in schedule_lines]
    scheduled_requests = set(schedule_ids)
    all_requests = set(requests.keys())
    if scheduled_requests != all_requests:
        missing = all_requests - scheduled_requests
        extra = scheduled_requests - all_requests
        msg = []
        if missing:
            msg.append(f"Missing requests: {sorted(missing)}")
        if extra:
            msg.append(f"Unknown requests: {sorted(extra)}")
        return False, "; ".join(msg)

    valid_gpus = {f"GPU{i+1}" for i in range(n_gpus)}
    for req_id, assigned_gpu in schedule_lines:
        if assigned_gpu not in valid_gpus:
            return False, f"Invalid GPU '{assigned_gpu}' for request {req_id}"
        available_gpus = {gpu for gpu, _, _ in requests[req_id]}
        if assigned_gpu not in available_gpus:
            return False, f"GPU '{assigned_gpu}' is not available for request {req_id}"
    return True, ""


def compute_schedule_score_from_lines(schedule_lines: List[Tuple[str, str]],
                                      requests: Dict[str, List[Tuple[str, int, int]]],
                                      n_gpus: int) -> int:
    """
    Compute maximum weighted completion time from schedule lines that include order.

    Args:
        schedule_lines: list of (request_id, assigned_gpu) in execution order.
        requests: mapping request_id -> [(gpu_id, exec_time, priority), ...]
        n_gpus: number of GPUs.

    Returns:
        int: maximum weighted completion time.

    Raises:
        ValueError: if exec time for a (request, gpu) pair is missing.
    """
    # Build per-GPU queues preserving order in schedule_lines
    gpu_queues: Dict[str, List[Tuple[str, int, int]]] = {f"GPU{i+1}": [] for i in range(n_gpus)}
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

    # Simulate each GPU queue
    max_wct = 0
    for gpu_id, queue in gpu_queues.items():
        t = 0
        for _, et, prio in queue:
            t += et
            wct = prio * t
            if wct > max_wct:
                max_wct = wct
    return max_wct


def compute_schedule_score_from_assignment(assignment: Dict[str, str],
                                           requests: Dict[str, List[Tuple[str, int, int]]],
                                           n_gpus: int,
                                           tie_break: str = "priority_then_time") -> int:
    """
    Compute schedule score from an assignment mapping request->GPU.
    Because the assignment lacks order, we choose a deterministic ordering
    per GPU for evaluation. The ordering chosen is:

      - If tie_break == "priority_then_time": sort by priority (desc), then exec_time (asc), then request_id.
      - This is deterministic and reasonable for grading baseline/auto-generated schedules.

    Args:
        assignment: mapping request_id -> gpu_id
        requests: original options
        n_gpus: number of GPUs
        tie_break: ordering heuristic for tasks on a GPU

    Returns:
        int: computed maximum weighted completion time for the derived schedule.
    """
    # Build per-GPU lists (unordered)
    gpu_tasks: Dict[str, List[Tuple[str, int, int]]] = {f"GPU{i+1}": [] for i in range(n_gpus)}
    for req_id, gpu_id in assignment.items():
        found = False
        for g, et, p in requests[req_id]:
            if g == gpu_id:
                gpu_tasks[gpu_id].append((req_id, et, p))
                found = True
                break
        if not found:
            raise ValueError(f"Cannot find execution time for {req_id} on {gpu_id}")

    # Choose deterministic order per GPU
    ordered_lines: List[Tuple[str, str]] = []
    for gpu_id in sorted(gpu_tasks.keys(), key=lambda x: int(x[3:])):  # GPU1, GPU2...
        tasks = gpu_tasks[gpu_id]
        if tie_break == "priority_then_time":
            tasks_sorted = sorted(tasks, key=lambda x: (-x[2], x[1], x[0]))
        else:
            tasks_sorted = sorted(tasks, key=lambda x: (x[1], x[0]))
        for req_id, _, _ in tasks_sorted:
            ordered_lines.append((req_id, gpu_id))

    # Now compute using ordered_lines
    return compute_schedule_score_from_lines(ordered_lines, requests, n_gpus)


def greedy_baseline_score(requests: Dict[str, List[Tuple[str, int, int]]], n_gpus: int) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Build a deterministic greedy baseline schedule and return its score and lines.

    Algorithm (deterministic):
      - Sort requests by (priority desc, min_exec_time asc, request_id asc).
      - Maintain per-GPU current cumulative time.
      - For each request in that order, choose the GPU that leads to the smallest
        *resulting* weighted completion time for that request:
            candidate_score = (gpu_time[gpu] + exec_time_on_gpu) * priority
        Break ties deterministically by GPU id (GPU1 before GPU2 ...).

    Returns:
        (score, schedule_lines) where schedule_lines is list of (req_id, gpu_id) in the
        order they were assigned (used as execution order on each GPU).
    """
    # Precompute priority and minimum exec time for each request for sorting
    req_meta = []
    for req_id, options in requests.items():
        priority = options[0][2]  # validated to be same across options
        min_et = min(et for _, et, _ in options)
        req_meta.append((req_id, priority, min_et))

    # Sort requests: priority desc, min_et asc, req_id asc
    req_meta.sort(key=lambda x: (-x[1], x[2], x[0]))

    gpu_time = {f"GPU{i+1}": 0 for i in range(n_gpus)}
    schedule_lines: List[Tuple[str, str]] = []

    for req_id, prio, _ in req_meta:
        best_gpu = None
        best_candidate_val = None
        # Evaluate candidate GPU choices deterministically (GPU1..GPUn)
        for i in range(n_gpus):
            gpu_id = f"GPU{i+1}"
            # find exec time for this gpu
            found = False
            et = None
            for g, t, p in requests[req_id]:
                if g == gpu_id:
                    et = t
                    found = True
                    break
            if not found:
                # This GPU is not available for this request; skip
                continue
            candidate_val = (gpu_time[gpu_id] + et) * prio
            if best_candidate_val is None or candidate_val < best_candidate_val:
                best_candidate_val = candidate_val
                best_gpu = gpu_id
            # tie-breaking deterministic: prefer lower GPU id if equal
        if best_gpu is None:
            raise ValueError(f"No available GPU found for request {req_id} in greedy baseline")
        # assign
        schedule_lines.append((req_id, best_gpu))
        # update gpu_time
        gpu_time[best_gpu] += next(et for g, et, _ in requests[req_id] if g == best_gpu)

    score = compute_schedule_score_from_lines(schedule_lines, requests, n_gpus)
    return score, schedule_lines


def exact_optimal_score_if_small(requests: Dict[str, List[Tuple[str, int, int]]],
                                 n_gpus: int,
                                 max_requests_for_exact: int = 7,
                                 max_assignments_exhaustive: int = 100_000) -> Tuple[int, bool]:
    """
    Attempt to compute exact optimal score for very small instances.

    This function is intentionally limited to small inputs (default m <= 7)
    to avoid combinatorial explosion. It enumerates all assignments (n_gpus^m)
    and, for each assignment, chooses a deterministic ordering per GPU
    (we could enumerate orders per GPU but that is even more expensive).

    The function returns (best_score, used_exact_flag) where used_exact_flag
    indicates whether an exhaustive search was performed.

    Args:
        requests: mapping request_id -> options
        n_gpus: number of GPUs
        max_requests_for_exact: threshold for attempting exact search
        max_assignments_exhaustive: safety cap on number of assignments to enumerate

    Returns:
        (best_score, used_exact_flag)
    """
    req_ids = list(requests.keys())
    m = len(req_ids)
    if m > max_requests_for_exact:
        return 0, False

    total_assignments = n_gpus ** m
    if total_assignments > max_assignments_exhaustive:
        return 0, False

    best = None
    # enumerate assignments as tuples of gpu indices
    for tup in itertools.product(range(n_gpus), repeat=m):
        assignment = {}
        valid = True
        for req_idx, g_idx in enumerate(tup):
            req_id = req_ids[req_idx]
            gpu_id = f"GPU{g_idx+1}"
            # verify gpu_id is available for this request
            if all(g != gpu_id for g, _, _ in requests[req_id]):
                valid = False
                break
            assignment[req_id] = gpu_id
        if not valid:
            continue
        # compute score by deterministic ordering per GPU
        score = compute_schedule_score_from_assignment(assignment, requests, n_gpus)
        if best is None or score < best:
            best = score
    if best is None:
        return 0, False
    return best, True


def grade(transcript: str) -> GradingResult:
    """
    Grade the GPU scheduling submission.

    Steps:
      1. Verify required files exist:
           - /workdir/data/requests.txt
           - /workdir/schedule.txt
           - /workdir/ans.txt
      2. Parse input and schedule. Schedule file order is considered execution order.
      3. Validate schedule feasibility.
      4. Compute actual_score from schedule lines.
      5. Compute a greedy baseline score (deterministic).
      6. Optionally, for very small instances, compute exact optimal score.
      7. Check near-optimality: actual_score <= max(reference_score, exact_optimal)*tolerance.
         Default tolerance = 1.25 (25%). This tolerance is configurable with the
         environment variable `GRADER_NEAR_OPT_TOL` (float > 1.0).
      8. Verify claimed answer in ans.txt equals actual_score.

    Args:
        transcript: unused (kept for compatibility with the grader interface).

    Returns:
        GradingResult: structured result with score, subscores, weights, and feedback.
    """
    feedback_messages: List[str] = []
    subscores = {"correct_answer": 0.0}
    weights = {"correct_answer": 1.0}

    try:
        schedule_path = Path("/workdir/schedule.txt")
        answer_path = Path("/workdir/ans.txt")
        data_path = Path("/workdir/data/requests.txt")

        # check files
        for p, name in [(data_path, "Input"), (schedule_path, "Schedule"), (answer_path, "Answer")]:
            if not p.exists():
                feedback_messages.append(f"{name} file {p} does not exist")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        # parse input
        n_gpus, n_requests, requests = parse_requests_file(data_path)
        feedback_messages.append(f"Problem: {n_requests} requests, {n_gpus} GPUs")

        # parse schedule (ordered lines)
        schedule_lines = parse_schedule_file(schedule_path)
        feedback_messages.append(f"Schedule contains {len(schedule_lines)} unique assignments (order preserved)")

        # validate schedule
        is_valid, err = validate_schedule(schedule_lines, requests, n_gpus)
        if not is_valid:
            feedback_messages.append(f"Invalid schedule: {err}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        feedback_messages.append("Schedule is feasible")

        # claimed answer
        try:
            with open(answer_path, "r") as f:
                claimed_answer = int(f.read().strip())
            feedback_messages.append(f"Claimed answer: {claimed_answer}")
        except Exception as e:
            feedback_messages.append(f"Invalid ans.txt: {e}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # compute actual score based on provided order
        actual_score = compute_schedule_score_from_lines(schedule_lines, requests, n_gpus)
        feedback_messages.append(f"Actual computed score (respecting student's order): {actual_score}")

        # claimed vs computed
        if claimed_answer != actual_score:
            feedback_messages.append(f"Claimed answer ({claimed_answer}) does not match actual computed score ({actual_score})")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # greedy baseline
        baseline_score, baseline_schedule_lines = greedy_baseline_score(requests, n_gpus)
        feedback_messages.append(f"Deterministic greedy baseline score: {baseline_score}")

        # small-instance exact solver (best-effort)
        exact_score, used_exact = exact_optimal_score_if_small(requests, n_gpus)
        if used_exact:
            feedback_messages.append(f"Exact optimal score (computed exhaustively): {exact_score}")

        # reference score: prefer exact if available, else baseline
        reference_score = exact_score if used_exact else baseline_score

        # tolerance: default 1.25, configurable via environment variable
        try:
            tol_env = os.getenv("GRADER_NEAR_OPT_TOL")
            tolerance = float(tol_env) if tol_env is not None else 1.25
            if tolerance <= 1.0:
                feedback_messages.append(f"Ignoring invalid tolerance value {tolerance}; using 1.25")
                tolerance = 1.25
        except Exception:
            tolerance = 1.25

        feedback_messages.append(f"Near-optimality tolerance: {tolerance:.2f}x reference")

        # Enforce near-optimality: actual_score <= reference_score * tolerance
        # If reference_score is zero (degenerate), avoid division by zero by requiring actual_score == 0
        if reference_score == 0:
            if actual_score != 0:
                feedback_messages.append("Reference score is 0 but actual score is > 0 — failing near-optimality.")
                return GradingResult(score=0.0, subscores=subscores, weights=weights, feedback="; ".join(feedback_messages))
        else:
            if actual_score > reference_score * tolerance:
                feedback_messages.append(
                    f"Schedule is not near-optimal: actual {actual_score} > {tolerance:.2f} * reference ({reference_score})"
                )
                return GradingResult(score=0.0, subscores=subscores, weights=weights, feedback="; ".join(feedback_messages))

        # All good
        subscores["correct_answer"] = 1.0
        feedback_messages.append("Solution is valid and within near-optimal tolerance.")
        return GradingResult(score=1.0, subscores=subscores, weights=weights, feedback="; ".join(feedback_messages))

    except Exception as e:
        import traceback
        feedback_messages.append(f"Unexpected error during grading: {e}")
        feedback_messages.append(traceback.format_exc())
        return GradingResult(score=0.0, subscores=subscores, weights=weights, feedback="; ".join(feedback_messages))
