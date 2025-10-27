#!/bin/bash
set -e

cd /workdir/data

python3 - <<'PYCODE'
import collections
import random
import math
from typing import List, Tuple, Dict, Set

# Parse complex network input
with open("network.txt") as f:
    header = f.readline().split()
    n, m, r, u, d = map(int, header[:5])
    alpha = list(map(float, header[5:]))
    
    regions = {}
    for _ in range(r):
        parts = f.readline().split()
        region_id, num_caches, bandwidth = parts[0], int(parts[1]), float(parts[2])
        regions[region_id] = {'bandwidth': bandwidth}
    
    caches = {}
    for _ in range(n):
        parts = f.readline().split()
        cache_id, region_id, capacity = parts[0], parts[1], int(parts[2])
        caches[cache_id] = {'region': region_id, 'capacity': capacity}
    
    objects = {}
    for _ in range(m):
        parts = f.readline().split()
        obj_id, pop, size = parts[0], int(parts[1]), int(parts[2])
        objects[obj_id] = {'popularity': pop, 'size': size, 'latencies': {}}
    
    for _ in range(n * m):
        parts = f.readline().split()
        obj_id, cache_id, latency = parts[0], parts[1], int(parts[2])
        objects[obj_id]['latencies'][cache_id] = latency
    
    user_groups = {}
    for _ in range(u):
        parts = f.readline().split()
        user_id, region_id, volume = parts[0], parts[1], int(parts[2])
        user_groups[user_id] = {'region': region_id, 'accesses': {}}
    
    for _ in range(u * m):
        parts = f.readline().split()
        user_id, obj_id, freq = parts[0], parts[1], int(parts[2])
        user_groups[user_id]['accesses'][obj_id] = freq
    
    transfer_costs = {}
    for _ in range(r * r):
        parts = f.readline().split()
        region_a, region_b, cost = parts[0], parts[1], float(parts[2])
        transfer_costs[(region_a, region_b)] = cost
    
    f.readline()  # blank line
    dependencies = []
    for line in f:
        line = line.strip()
        if line:
            prereq, dependent = line.split()
            dependencies.append((prereq, dependent))

print(f"Parsed: {n} caches, {m} objects, {r} regions, {u} user groups, {d} deps")
print(f"Cost weights: alpha = {alpha}")

# Build dependency graph
prereqs = collections.defaultdict(set)
successors = collections.defaultdict(set)
for prereq, dependent in dependencies:
    prereqs[dependent].add(prereq)
    successors[prereq].add(dependent)

def compute_total_cost(replication):
    """Compute full cost with all 4 components."""
    replication_map = collections.defaultdict(list)
    for obj_id, cache_id in replication:
        replication_map[obj_id].append(cache_id)
    
    # 1. Access Cost
    access_cost = 0.0
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        for obj_id, freq in user_data['accesses'].items():
            min_latency = float('inf')
            for cache_id in replication_map[obj_id]:
                latency = objects[obj_id]['latencies'][cache_id]
                cache_region = caches[cache_id]['region']
                if cache_region == user_region:
                    min_latency = min(min_latency, latency)
                else:
                    min_latency = min(min_latency, latency * 1.5)
            pop = objects[obj_id]['popularity']
            access_cost += pop * freq * min_latency
    
    # 2. Replication Cost
    replication_cost = 0.0
    for obj_id, cache_list in replication_map.items():
        num_replicas = len(cache_list)
        size = objects[obj_id]['size']
        penalty = 2 ** (num_replicas - 1)
        replication_cost += size * penalty * num_replicas
    
    # 3. Cross-Region Transfer Cost
    transfer_cost = 0.0
    for obj_id, cache_list in replication_map.items():
        if obj_id not in prereqs:
            continue
        obj_regions = {caches[c]['region'] for c in cache_list}
        for prereq_id in prereqs[obj_id]:
            prereq_regions = {caches[c]['region'] for c in replication_map[prereq_id]}
            if not obj_regions.intersection(prereq_regions):
                max_cost = 0
                for obj_reg in obj_regions:
                    for prereq_reg in prereq_regions:
                        cost = transfer_costs.get((prereq_reg, obj_reg), 0)
                        max_cost = max(max_cost, cost)
                size = objects[prereq_id]['size']
                transfer_cost += size * max_cost
    
    # 4. Bandwidth Penalty
    bandwidth_penalty = 0.0
    region_usage = {region_id: 0.0 for region_id in regions}
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        for obj_id, freq in user_data['accesses'].items():
            same_region = [c for c in replication_map[obj_id] 
                          if caches[c]['region'] == user_region]
            if not same_region:
                size_mb = objects[obj_id]['size']
                region_usage[user_region] += (size_mb * freq) / 1024.0
    
    for region_id, usage in region_usage.items():
        capacity = regions[region_id]['bandwidth']
        if usage > 0.8 * capacity:
            excess = usage - 0.8 * capacity
            bandwidth_penalty += excess * excess * 100
    
    total = (alpha[0] * access_cost + alpha[1] * replication_cost + 
             alpha[2] * transfer_cost + alpha[3] * bandwidth_penalty)
    return int(round(total))

# Calculate object importance
importance = {}
for obj_id in objects:
    pop = objects[obj_id]['popularity']
    total_accesses = sum(ug['accesses'][obj_id] for ug in user_groups.values())
    num_dependents = len(successors[obj_id])
    importance[obj_id] = pop * total_accesses * (1 + num_dependents * 0.5)

sorted_objects = sorted(objects.keys(), key=lambda x: -importance[x])

# Calculate region affinity for each object
region_affinity = {}
for obj_id in objects:
    affinity = collections.defaultdict(float)
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        freq = user_data['accesses'][obj_id]
        affinity[user_region] += freq
    region_affinity[obj_id] = affinity

def build_replication_strategy(config):
    """
    Build replication based on configuration.
    config = (top_k_pct_double, top_k_pct_triple, use_affinity_threshold)
    """
    top_double, top_triple, affinity_threshold = config
    
    replication = []
    cache_usage = {cache_id: 0 for cache_id in caches}
    
    for rank, obj_id in enumerate(sorted_objects):
        size = objects[obj_id]['size']
        
        # Determine number of replicas
        pct_rank = rank / len(sorted_objects)
        if pct_rank < top_triple:
            num_replicas = 3
        elif pct_rank < top_double:
            num_replicas = 2
        else:
            num_replicas = 1
        
        # Find best caches based on region affinity and latency
        affinity = region_affinity[obj_id]
        total_affinity = sum(affinity.values())
        
        # Calculate score for each cache
        cache_scores = []
        for cache_id, cache_data in caches.items():
            if cache_usage[cache_id] + size > cache_data['capacity']:
                continue
            
            cache_region = cache_data['region']
            region_freq = affinity[cache_region]
            
            # Score based on: access frequency + latency + capacity
            access_benefit = 0
            for user_id, user_data in user_groups.items():
                user_region = user_data['region']
                freq = user_data['accesses'][obj_id]
                latency = objects[obj_id]['latencies'][cache_id]
                if cache_region == user_region:
                    access_benefit += freq * (200 - latency)
                else:
                    access_benefit += freq * (200 - latency * 1.5) * 0.5
            
            utilization = cache_usage[cache_id] / cache_data['capacity']
            score = access_benefit / (1 + utilization * 2)
            cache_scores.append((score, cache_id, cache_region))
        
        cache_scores.sort(reverse=True)
        
        # Select caches, preferring different regions for replicas
        selected = []
        selected_regions = set()
        for score, cache_id, cache_region in cache_scores:
            if len(selected) >= num_replicas:
                break
            # For 2+ replicas, prefer different regions
            if num_replicas > 1 and len(selected) > 0:
                if cache_region in selected_regions and len(selected) < num_replicas - 1:
                    continue
            selected.append(cache_id)
            selected_regions.add(cache_region)
            cache_usage[cache_id] += size
        
        # Add remaining replicas if couldn't get enough
        if len(selected) < num_replicas:
            for _, cache_id, cache_region in cache_scores:
                if cache_id not in selected and cache_usage[cache_id] + size <= caches[cache_id]['capacity']:
                    selected.append(cache_id)
                    cache_usage[cache_id] += size
                    if len(selected) >= num_replicas:
                        break
        
        for cache_id in selected:
            replication.append((obj_id, cache_id))
    
    return replication

# Multi-start optimization with different strategies
print("\\nSearching across multiple strategies...")
best_replication = None
best_cost = float('inf')

strategies = [
    (0.15, 0.05, 0.3),  # Conservative replication
    (0.25, 0.10, 0.4),  # Moderate replication
    (0.35, 0.15, 0.5),  # Aggressive replication
    (0.20, 0.08, 0.35), # Balanced
    (0.30, 0.12, 0.45), # High coverage
]

for i, config in enumerate(strategies):
    try:
        replication = build_replication_strategy(config)
        cost = compute_total_cost(replication)
        print(f"  Strategy {i+1} (double={config[0]:.0%}, triple={config[1]:.0%}): cost = {cost}")
        if cost < best_cost:
            best_cost = cost
            best_replication = replication
    except Exception as e:
        print(f"  Strategy {i+1}: FAILED - {e}")

print(f"\\nBest initial cost: {best_cost}")

# Local search optimization
print("Applying local search optimization...")
iterations = 0
max_iterations = 200
improved = True

while improved and iterations < max_iterations:
    improved = False
    iterations += 1
    
    # Try adding a replica to high-value objects
    if iterations % 3 == 0:
        replication_map = collections.defaultdict(list)
        for obj_id, cache_id in best_replication:
            replication_map[obj_id].append(cache_id)
        
        for obj_id in sorted_objects[:len(sorted_objects)//3]:
            if len(replication_map[obj_id]) >= 5:
                continue
            
            size = objects[obj_id]['size']
            current_regions = {caches[c]['region'] for c in replication_map[obj_id]}
            
            # Try adding to an empty region
            for cache_id, cache_data in caches.items():
                if cache_id in replication_map[obj_id]:
                    continue
                cache_region = cache_data['region']
                if cache_region in current_regions:
                    continue
                
                # Check capacity
                used = sum(objects[oid]['size'] for oid, cid in best_replication if cid == cache_id)
                if used + size > cache_data['capacity']:
                    continue
                
                test_replication = list(best_replication) + [(obj_id, cache_id)]
                try:
                    new_cost = compute_total_cost(test_replication)
                    if new_cost < best_cost:
                        best_replication = test_replication
                        best_cost = new_cost
                        improved = True
                        print(f"  Iteration {iterations}: improved to {best_cost}")
                        break
                except:
                    continue
            if improved:
                break
    
    # Try removing a replica from low-value objects
    if not improved and iterations % 4 == 0:
        replication_map = collections.defaultdict(list)
        for obj_id, cache_id in best_replication:
            replication_map[obj_id].append(cache_id)
        
        for obj_id in sorted_objects[len(sorted_objects)//2:]:
            if len(replication_map[obj_id]) <= 1:
                continue
            
            # Try removing each replica
            for cache_id in replication_map[obj_id]:
                test_replication = [(o, c) for o, c in best_replication 
                                   if not (o == obj_id and c == cache_id)]
                try:
                    new_cost = compute_total_cost(test_replication)
                    if new_cost < best_cost:
                        best_replication = test_replication
                        best_cost = new_cost
                        improved = True
                        print(f"  Iteration {iterations}: improved to {best_cost}")
                        break
                except:
                    continue
            if improved:
                break

print(f"Final cost after {iterations} iterations: {best_cost}")

# Write outputs
with open("/workdir/replication.txt", "w") as f:
    for obj_id, cache_id in best_replication:
        f.write(f"{obj_id} {cache_id}\\n")

with open("/workdir/ans.txt", "w") as f:
    f.write(f"{best_cost}\\n")

print(f"\\nTotal cost: {best_cost}")
print("Replication strategy written successfully")
PYCODE