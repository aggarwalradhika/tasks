#!/bin/bash
set -e

cd /workdir/data

python3 - <<'PYCODE'
import collections
from typing import List, Tuple, Dict, Set

# Parse input
with open("requests.txt") as f:
    n, m, d = map(int, f.readline().split())
    
    # Parse request options
    lines = []
    for line in f:
        line = line.strip()
        if not line:
            break
        lines.append(line.split())
    
    # Group by request ID
    requests = collections.defaultdict(list)
    for req_id, prio, gpu, t in lines:
        requests[req_id].append((gpu, int(t), int(prio)))
    
    # Parse dependencies
    dependencies = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        prereq, dependent = line.split()
        dependencies.append((prereq, dependent))

print(f"Parsed: {n} GPUs, {m} requests, {d} dependencies")

# Build dependency graph
prereqs = collections.defaultdict(set)  # dependent -> set of prerequisites
successors = collections.defaultdict(set)  # prereq -> set of dependents

for prereq, dependent in dependencies:
    prereqs[dependent].add(prereq)
    successors[prereq].add(dependent)

def topological_sort(all_requests: Set[str]) -> List[str]:
    """Kahn's algorithm for topological sorting."""
    in_degree = {req: 0 for req in all_requests}
    for prereq, dependent in dependencies:
        in_degree[dependent] += 1
    
    queue = sorted([req for req in all_requests if in_degree[req] == 0])
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in sorted(successors[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort()
    
    return result

def compute_critical_path_priority(req_id: str, memo: Dict[str, float]) -> float:
    """
    Compute critical path length from this request to end.
    Requests on critical paths should be scheduled earlier.
    """
    if req_id in memo:
        return memo[req_id]
    
    if req_id not in successors or len(successors[req_id]) == 0:
        # Leaf node - just its own execution time
        min_time = min(et for _, et, _ in requests[req_id])
        memo[req_id] = min_time
        return min_time
    
    # Internal node - max of successor paths + own time
    min_time = min(et for _, et, _ in requests[req_id])
    max_successor_path = max(compute_critical_path_priority(succ, memo) 
                             for succ in successors[req_id])
    memo[req_id] = min_time + max_successor_path
    return memo[req_id]

def compute_schedule_with_deps(schedule_order: List[Tuple[str, str]]) -> int:
    """
    Simulate schedule respecting dependencies and compute max weighted CT.
    """
    gpu_available = {f"GPU{i+1}": 0 for i in range(n)}
    completion_times = {}
    
    for req_id, assigned_gpu in schedule_order:
        # Find exec time
        exec_time = None
        priority = None
        for gpu_id, et, p in requests[req_id]:
            if gpu_id == assigned_gpu:
                exec_time = et
                priority = p
                break
        
        # Compute earliest start
        earliest_start = gpu_available[assigned_gpu]
        if req_id in prereqs:
            for prereq_id in prereqs[req_id]:
                earliest_start = max(earliest_start, completion_times[prereq_id])
        
        completion = earliest_start + exec_time
        completion_times[req_id] = completion
        gpu_available[assigned_gpu] = completion
    
    # Compute max weighted completion time
    max_wct = 0
    for req_id, assigned_gpu in schedule_order:
        priority = requests[req_id][0][2]
        wct = priority * completion_times[req_id]
        max_wct = max(max_wct, wct)
    
    return max_wct

# Advanced scheduling algorithm
all_requests = set(requests.keys())
topo_order = topological_sort(all_requests)

print("Computing critical paths...")
critical_path_memo = {}
for req_id in all_requests:
    compute_critical_path_priority(req_id, critical_path_memo)

# Compute request priorities
req_priorities = {}
for req_id in all_requests:
    priority = requests[req_id][0][2]
    min_time = min(et for _, et, _ in requests[req_id])
    avg_time = sum(et for _, et, _ in requests[req_id]) / len(requests[req_id])
    critical_path = critical_path_memo.get(req_id, min_time)
    
    # Priority score combines:
    # 1. User priority (weight 5x)
    # 2. Critical path length (longer = more urgent)
    # 3. Efficiency ratio (priority/time)
    priority_score = (priority * 5.0) + (critical_path / 10.0) + (priority / min_time)
    
    req_priorities[req_id] = (priority_score, priority, min_time, critical_path, req_id)

print("Building initial schedule with critical path heuristic...")

# Sort by topo order, but use priority within constraints
scheduled = set()
schedule_lines = []
gpu_available = {f"GPU{i+1}": 0 for i in range(n)}
completion_times = {}

# Build ready queue
ready = []
for req_id in all_requests:
    if req_id not in prereqs or len(prereqs[req_id]) == 0:
        ready.append(req_id)

# Sort ready queue by priority
ready.sort(key=lambda r: req_priorities[r], reverse=True)

while ready:
    req_id = ready.pop(0)
    priority = requests[req_id][0][2]
    
    # Find best GPU using sophisticated heuristic
    best_gpu = None
    best_score = float('inf')
    best_exec_time = None
    
    for i in range(n):
        gpu_id = f"GPU{i+1}"
        
        # Find exec time on this GPU
        exec_time = None
        for g, et, p in requests[req_id]:
            if g == gpu_id:
                exec_time = et
                break
        
        if exec_time is None:
            continue
        
        # Compute start time with dependencies
        earliest_start = gpu_available[gpu_id]
        if req_id in prereqs:
            for prereq_id in prereqs[req_id]:
                if prereq_id in completion_times:
                    earliest_start = max(earliest_start, completion_times[prereq_id])
        
        completion = earliest_start + exec_time
        weighted_ct = priority * completion
        
        # Score considers both weighted CT and load balancing
        load_imbalance = abs(gpu_available[gpu_id] - sum(gpu_available.values())/n)
        score = weighted_ct + load_imbalance * 0.1
        
        if score < best_score:
            best_score = score
            best_gpu = gpu_id
            best_exec_time = exec_time
    
    if best_gpu is None:
        raise ValueError(f"No GPU available for {req_id}")
    
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
            else:
                if other_req not in scheduled:
                    ready.append(other_req)
    
    ready.sort(key=lambda r: req_priorities[r], reverse=True)

initial_score = compute_schedule_with_deps(schedule_lines)
print(f"Initial score: {initial_score}")

# Local search optimization
print("Applying local search...")
best_schedule = list(schedule_lines)
best_score = initial_score
improved = True
iterations = 0
max_iterations = 50

while improved and iterations < max_iterations:
    improved = False
    iterations += 1
    
    # Try swapping assignments
    for idx in range(len(best_schedule)):
        req_id, current_gpu = best_schedule[idx]
        
        # Try other GPUs
        for i in range(n):
            new_gpu = f"GPU{i+1}"
            if new_gpu == current_gpu:
                continue
            
            # Check if available on this GPU
            available = False
            for g, et, p in requests[req_id]:
                if g == new_gpu:
                    available = True
                    break
            
            if not available:
                continue
            
            # Try the swap
            test_schedule = list(best_schedule)
            test_schedule[idx] = (req_id, new_gpu)
            
            try:
                new_score = compute_schedule_with_deps(test_schedule)
                if new_score < best_score:
                    best_schedule = test_schedule
                    best_score = new_score
                    improved = True
                    print(f"  Improvement: {new_score} (moved {req_id} to {new_gpu})")
                    break
            except:
                continue
        
        if improved:
            break

print(f"Final score after local search: {best_score}")

# Write schedule in execution order
with open("/workdir/schedule.txt", "w") as f:
    for req_id, gpu_id in best_schedule:
        f.write(f"{req_id} {gpu_id}\n")

# Write answer
with open("/workdir/ans.txt", "w") as f:
    f.write(f"{best_score}\n")

print(f"\nMaximum weighted completion time: {best_score}")
print("Schedule written successfully")
PYCODE