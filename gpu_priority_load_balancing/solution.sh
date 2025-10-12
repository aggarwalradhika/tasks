#!/bin/bash
set -e

cd /workdir/data

python3 - <<'PYCODE'
import collections
import heapq

# Parse input
with open("requests.txt") as f:
    n, m = map(int, f.readline().split())
    lines = [line.strip().split() for line in f.readlines() if line.strip()]

# Group by request ID
requests = collections.defaultdict(list)
for req_id, prio, gpu, t in lines:
    requests[req_id].append((gpu, int(t), int(prio)))

# We need to minimize the maximum weighted completion time
# This is NP-hard, so we use a sophisticated greedy heuristic

# Priority queue approach: always assign the next request to minimize
# the increase in maximum weighted completion time

# Track GPU loads and assignments
gpu_times = {f"GPU{i+1}": 0 for i in range(n)}
schedule = {}  # request_id -> gpu_id

# Sort requests by priority (high first) then by minimum execution time
request_list = []
for req_id, options in requests.items():
    priority = options[0][2]
    min_time = min(et for _, et, _ in options)
    avg_time = sum(et for _, et, _ in options) / len(options)
    request_list.append((priority, min_time, avg_time, req_id, options))

# Sort by: priority (desc), then min_time (asc), then avg_time (asc)
request_list.sort(key=lambda x: (-x[0], x[1], x[2]))

# Greedy assignment with lookahead
for priority, min_time, avg_time, req_id, options in request_list:
    # For each GPU option, compute what the max weighted CT would be
    best_gpu = None
    best_increase = float('inf')
    
    for gpu_id, exec_time, _ in options:
        # If we assign this request to this GPU, what's the new weighted CT?
        new_completion = (gpu_times[gpu_id] + exec_time) * priority
        
        # Compute current max weighted completion time
        # (we need to track this properly)
        current_max = 0
        for other_gpu, other_time in gpu_times.items():
            if other_time > 0:
                # Approximate weighted CT (assuming priority 1 for existing)
                current_max = max(current_max, other_time)
        
        # How much would this assignment increase the max?
        increase = new_completion - current_max
        
        # Prefer GPU with smallest increase, with tie-breaking
        if increase < best_increase or (
            increase == best_increase and 
            (best_gpu is None or gpu_id < best_gpu)
        ):
            best_gpu = gpu_id
            best_increase = increase
            best_exec_time = exec_time
    
    # Assign request to best GPU
    schedule[req_id] = best_gpu
    gpu_times[best_gpu] += best_exec_time

# Now compute the actual maximum weighted completion time
# by simulating the schedule

gpu_queues = {f"GPU{i+1}": [] for i in range(n)}

for req_id, assigned_gpu in schedule.items():
    # Find the execution time and priority
    exec_time = None
    priority = None
    for gpu_id, et, p in requests[req_id]:
        if gpu_id == assigned_gpu:
            exec_time = et
            priority = p
            break
    
    gpu_queues[assigned_gpu].append((req_id, exec_time, priority))

# Simulate execution
max_weighted_ct = 0
for gpu_id, queue in gpu_queues.items():
    current_time = 0
    for req_id, exec_time, priority in queue:
        current_time += exec_time
        weighted_ct = priority * current_time
        max_weighted_ct = max(max_weighted_ct, weighted_ct)

# Write the schedule
with open("/workdir/schedule.txt", "w") as f:
    for req_id in sorted(schedule.keys()):
        f.write(f"{req_id} {schedule[req_id]}\n")

# Write the answer
with open("/workdir/ans.txt", "w") as f:
    f.write(f"{max_weighted_ct}\n")

print(f"Maximum weighted completion time: {max_weighted_ct}")
print(f"Schedule written to /workdir/schedule.txt")
PYCODE