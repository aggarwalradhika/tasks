#!/bin/bash
set -e

cd /workdir/data

python3 - <<'PYCODE'
import collections
from typing import List, Tuple, Dict, Set
import random

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
prereqs = collections.defaultdict(set)
successors = collections.defaultdict(set)

for prereq, dependent in dependencies:
    prereqs[dependent].add(prereq)
    successors[prereq].add(dependent)

def compute_critical_path_length(req_id: str, memo: Dict[str, float]) -> float:
    """Compute critical path using minimum execution times."""
    if req_id in memo:
        return memo[req_id]
    
    min_time = min(et for _, et, _ in requests[req_id])
    
    if req_id not in successors or len(successors[req_id]) == 0:
        memo[req_id] = min_time
        return min_time
    
    max_successor = max(compute_critical_path_length(s, memo) for s in successors[req_id])
    memo[req_id] = min_time + max_successor
    return memo[req_id]

def compute_schedule_score(schedule_order: List[Tuple[str, str]]) -> int:
    """Simulate schedule and compute max weighted CT."""
    gpu_available = {f"GPU{i+1}": 0 for i in range(n)}
    completion_times = {}
    
    for req_id, assigned_gpu in schedule_order:
        exec_time = None
        priority = None
        for gpu_id, et, p in requests[req_id]:
            if gpu_id == assigned_gpu:
                exec_time = et
                priority = p
                break
        
        earliest_start = gpu_available[assigned_gpu]
        if req_id in prereqs:
            for prereq_id in prereqs[req_id]:
                earliest_start = max(earliest_start, completion_times[prereq_id])
        
        completion = earliest_start + exec_time
        completion_times[req_id] = completion
        gpu_available[assigned_gpu] = completion
    
    max_wct = 0
    for req_id, _ in schedule_order:
        priority = requests[req_id][0][2]
        wct = priority * completion_times[req_id]
        max_wct = max(max_wct, wct)
    
    return max_wct

# Compute critical paths
all_requests = set(requests.keys())
critical_paths = {}
for req_id in all_requests:
    compute_critical_path_length(req_id, critical_paths)

def build_schedule_greedy(priority_weights: Tuple[float, float, float]) -> List[Tuple[str, str]]:
    """
    Build schedule using weighted priority heuristic.
    priority_weights = (w_priority, w_critical_path, w_efficiency)
    """
    w_prio, w_crit, w_eff = priority_weights
    
    # Compute request scores
    req_scores = {}
    for req_id in all_requests:
        priority = requests[req_id][0][2]
        min_time = min(et for _, et, _ in requests[req_id])
        crit_path = critical_paths[req_id]
        
        # Weighted combination
        score = (w_prio * priority * min_time + 
                 w_crit * crit_path + 
                 w_eff * priority / min_time)
        
        req_scores[req_id] = (score, priority, min_time, req_id)
    
    scheduled = set()
    schedule = []
    gpu_available = {f"GPU{i+1}": 0 for i in range(n)}
    completion_times = {}
    
    ready = [r for r in all_requests if len(prereqs[r]) == 0]
    ready.sort(key=lambda r: req_scores[r], reverse=True)
    
    while ready:
        req_id = ready.pop(0)
        priority = requests[req_id][0][2]
        
        # Find best GPU
        best_gpu = None
        best_metric = float('inf')
        best_exec = None
        
        for i in range(n):
            gpu_id = f"GPU{i+1}"
            
            exec_time = None
            for g, et, p in requests[req_id]:
                if g == gpu_id:
                    exec_time = et
                    break
            
            if exec_time is None:
                continue
            
            earliest_start = gpu_available[gpu_id]
            if req_id in prereqs:
                for prereq_id in prereqs[req_id]:
                    if prereq_id in completion_times:
                        earliest_start = max(earliest_start, completion_times[prereq_id])
            
            completion = earliest_start + exec_time
            wct = priority * completion
            
            # Prefer GPU with lowest weighted completion time
            if wct < best_metric:
                best_metric = wct
                best_gpu = gpu_id
                best_exec = exec_time
        
        if best_gpu is None:
            raise ValueError(f"No GPU for {req_id}")
        
        earliest_start = gpu_available[best_gpu]
        if req_id in prereqs:
            for prereq_id in prereqs[req_id]:
                if prereq_id in completion_times:
                    earliest_start = max(earliest_start, completion_times[prereq_id])
        
        completion = earliest_start + best_exec
        schedule.append((req_id, best_gpu))
        gpu_available[best_gpu] = completion
        completion_times[req_id] = completion
        scheduled.add(req_id)
        
        # Add newly ready tasks
        for other in all_requests:
            if other not in scheduled and other not in ready:
                if all(p in scheduled for p in prereqs[other]):
                    ready.append(other)
        
        ready.sort(key=lambda r: req_scores[r], reverse=True)
    
    return schedule

# Try multiple weight configurations to find best
print("Searching for optimal weight configuration...")
best_schedule = None
best_score = float('inf')

# Test different weight combinations
weight_configs = [
    (2.0, 1.0, 1.0),   # Emphasize priority
    (1.0, 2.0, 1.0),   # Emphasize critical path
    (1.0, 1.0, 2.0),   # Emphasize efficiency
    (2.0, 2.0, 1.0),   # Balance priority and critical path
    (3.0, 1.0, 0.5),   # Heavy priority focus
    (1.0, 3.0, 0.5),   # Heavy critical path focus
    (1.5, 1.5, 1.0),   # Balanced
]

for weights in weight_configs:
    schedule = build_schedule_greedy(weights)
    score = compute_schedule_score(schedule)
    print(f"  Weights {weights}: score = {score}")
    if score < best_score:
        best_score = score
        best_schedule = schedule

print(f"\nBest initial score: {best_score}")

# Local search with multiple strategies
print("Applying intensive local search...")
iterations = 0
max_iterations = 500
improved = True

while improved and iterations < max_iterations:
    improved = False
    iterations += 1
    
    # Strategy 1: Swap GPU assignments
    for idx in range(len(best_schedule)):
        req_id, current_gpu = best_schedule[idx]
        
        for i in range(n):
            new_gpu = f"GPU{i+1}"
            if new_gpu == current_gpu:
                continue
            
            available = any(g == new_gpu for g, _, _ in requests[req_id])
            if not available:
                continue
            
            test_schedule = list(best_schedule)
            test_schedule[idx] = (req_id, new_gpu)
            
            try:
                new_score = compute_schedule_score(test_schedule)
                if new_score < best_score:
                    best_schedule = test_schedule
                    best_score = new_score
                    improved = True
                    if iterations % 50 == 0:
                        print(f"  Iteration {iterations}: {best_score}")
                    break
            except:
                continue
        
        if improved:
            break
    
    # Strategy 2: Swap execution order (respecting dependencies)
    if not improved and iterations % 5 == 0:
        for i in range(len(best_schedule) - 1):
            req_a, gpu_a = best_schedule[i]
            req_b, gpu_b = best_schedule[i + 1]
            
            # Check if swap is valid (no dependency violation)
            if req_a in prereqs[req_b] or req_b in prereqs[req_a]:
                continue
            
            test_schedule = list(best_schedule)
            test_schedule[i], test_schedule[i + 1] = test_schedule[i + 1], test_schedule[i]
            
            try:
                new_score = compute_schedule_score(test_schedule)
                if new_score < best_score:
                    best_schedule = test_schedule
                    best_score = new_score
                    improved = True
                    break
            except:
                continue

print(f"Final score after {iterations} iterations: {best_score}")

# Write outputs
with open("/workdir/schedule.txt", "w") as f:
    for req_id, gpu_id in best_schedule:
        f.write(f"{req_id} {gpu_id}\n")

with open("/workdir/ans.txt", "w") as f:
    f.write(f"{best_score}\n")

print(f"\nMaximum weighted completion time: {best_score}")
print("Schedule written successfully")
PYCODE