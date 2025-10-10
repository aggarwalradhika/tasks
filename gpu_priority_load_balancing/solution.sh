#!/bin/bash
set -e

INPUT="requests.txt"
OUTPUT="ans.txt"

python3 - <<'PYCODE'
import collections

with open("requests.txt") as f:
    n, m = map(int, f.readline().split())
    lines = [line.strip().split() for line in f.readlines() if line.strip()]

# Group by request ID
requests = collections.defaultdict(list)
for req_id, prio, gpu, t in lines:
    requests[req_id].append((gpu, int(t), int(prio)))

# Track GPU loads
gpu_times = {f"GPU{i+1}": 0 for i in range(n)}
completion_times = []

# Sort by priority desc
for req_id, options in sorted(requests.items(), key=lambda x: -x[1][0][2]):
    # pick GPU where completion_time (load + exec) * priority is smallest
    best_gpu, best_finish = None, float("inf")
    prio = options[0][2]
    for gpu, exec_time, _ in options:
        finish = (gpu_times[gpu] + exec_time) * prio
        if finish < best_finish:
            best_gpu, best_finish = gpu, finish
    gpu_times[best_gpu] += [t for g,t,p in options if g == best_gpu][0]
    completion_times.append(best_finish)

max_weighted_ct = max(completion_times)
with open("ans.txt", "w") as f:
    f.write(str(max_weighted_ct) + "\n")
PYCODE
