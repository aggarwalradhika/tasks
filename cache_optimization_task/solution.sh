#!/bin/bash
set -e

cd /workdir/data

python3 - <<'PYCODE'
import collections
import csv
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
    
    # 1. Access Cost with distance-aware routing
    access_cost = 0.0
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        for obj_id, freq in user_data['accesses'].items():
            min_latency = float('inf')
            for cache_id in replication_map[obj_id]:
                latency = objects[obj_id]['latencies'][cache_id]
                cache_region = caches[cache_id]['region']
                # Apply distance penalty
                if cache_region == user_region:
                    min_latency = min(min_latency, latency)
                else:
                    # Cross-region penalty varies by region pair
                    transfer_penalty = 1.0 + transfer_costs.get((user_region, cache_region), 0.5)
                    min_latency = min(min_latency, latency * transfer_penalty)
            pop = objects[obj_id]['popularity']
            access_cost += pop * freq * min_latency
    
    # 2. Replication Cost with non-linear scaling
    replication_cost = 0.0
    for obj_id, cache_list in replication_map.items():
        num_replicas = len(cache_list)
        size = objects[obj_id]['size']
        # Exponential penalty: 1,2,4,8,16 for 1-5 replicas
        penalty = 2 ** (num_replicas - 1)
        replication_cost += size * penalty * num_replicas
    
    # 3. Cross-Region Transfer Cost with hop counting
    transfer_cost = 0.0
    for obj_id, cache_list in replication_map.items():
        if obj_id not in prereqs:
            continue
        obj_regions = {caches[c]['region'] for c in cache_list}
        for prereq_id in prereqs[obj_id]:
            prereq_regions = {caches[c]['region'] for c in replication_map[prereq_id]}
            # If no overlap, incur transfer cost
            if not obj_regions.intersection(prereq_regions):
                # Use worst-case transfer cost
                max_cost = 0
                for obj_reg in obj_regions:
                    for prereq_reg in prereq_regions:
                        cost = transfer_costs.get((prereq_reg, obj_reg), 0)
                        max_cost = max(max_cost, cost)
                size = objects[prereq_id]['size']
                transfer_cost += size * max_cost * 50  # Amplify dependency cost
    
    # 4. Bandwidth Saturation with progressive penalty
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
        utilization = usage / capacity
        if utilization > 0.75:  # Lower threshold
            excess = usage - 0.75 * capacity
            bandwidth_penalty += excess * excess * 150  # Higher penalty
    
    total = (alpha[0] * access_cost + 
             alpha[1] * replication_cost + 
             alpha[2] * transfer_cost + 
             alpha[3] * bandwidth_penalty)
    
    return int(round(total))

# Calculate comprehensive object scoring
importance = {}
for obj_id in objects:
    pop = objects[obj_id]['popularity']
    total_accesses = sum(ug['accesses'][obj_id] for ug in user_groups.values())
    num_dependents = len(successors[obj_id])
    num_prereqs = len(prereqs[obj_id])
    # More complex importance considering dependencies both ways
    importance[obj_id] = pop * total_accesses * (1 + num_dependents * 0.5 + num_prereqs * 0.3)

sorted_objects = sorted(objects.keys(), key=lambda x: -importance[x])

# Calculate region affinity with weighted scoring
region_affinity = {}
for obj_id in objects:
    affinity = collections.defaultdict(float)
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        freq = user_data['accesses'][obj_id]
        pop = objects[obj_id]['popularity']
        # Weight by popularity
        affinity[user_region] += freq * pop
    region_affinity[obj_id] = affinity

def build_advanced_replication(top_pct_3rep, top_pct_2rep, bandwidth_factor):
    """Advanced replication with bandwidth awareness."""
    replication = []
    cache_usage = {cache_id: 0 for cache_id in caches}
    region_bw_estimate = {region_id: 0.0 for region_id in regions}
    
    for rank, obj_id in enumerate(sorted_objects):
        size = objects[obj_id]['size']
        pct_rank = rank / len(sorted_objects)
        
        # Determine replication level based on importance
        if pct_rank < top_pct_3rep:
            num_replicas = 3
        elif pct_rank < top_pct_2rep:
            num_replicas = 2
        else:
            num_replicas = 1
        
        # Calculate region demand
        affinity = region_affinity[obj_id]
        total_affinity = sum(affinity.values()) or 1
        
        # Score caches with complex heuristic
        cache_scores = []
        for cache_id, cache_data in caches.items():
            if cache_usage[cache_id] + size > cache_data['capacity']:
                continue
            
            cache_region = cache_data['region']
            region_demand = affinity.get(cache_region, 0)
            
            # Complex scoring: latency benefit + region demand + capacity headroom
            score = 0.0
            for user_id, user_data in user_groups.items():
                user_region = user_data['region']
                freq = user_data['accesses'][obj_id]
                latency = objects[obj_id]['latencies'][cache_id]
                
                if cache_region == user_region:
                    # Local access is best
                    score += freq * (250 - latency) * 2.0
                else:
                    # Cross-region access
                    penalty = 1.0 + transfer_costs.get((user_region, cache_region), 0.5)
                    score += freq * (250 - latency * penalty) * 0.4
            
            # Penalize high utilization
            utilization = cache_usage[cache_id] / cache_data['capacity']
            score *= (1.0 - utilization * 0.7)
            
            # Penalize regions near bandwidth limit
            bw_util = region_bw_estimate[cache_region] / regions[cache_region]['bandwidth']
            if bw_util > 0.6:
                score *= (1.0 - (bw_util - 0.6) * bandwidth_factor)
            
            cache_scores.append((score, cache_id, cache_region))
        
        if not cache_scores:
            continue
        
        cache_scores.sort(reverse=True)
        
        # Select caches preferring region diversity
        selected = []
        selected_regions = set()
        
        for score, cache_id, cache_region in cache_scores:
            if len(selected) >= num_replicas:
                break
            
            # For multiple replicas, prefer different regions
            if num_replicas > 1 and len(selected) > 0:
                if cache_region in selected_regions and len(cache_scores) > len(selected):
                    # Skip if same region and we have alternatives
                    if len([s for s in cache_scores if s[2] not in selected_regions]) > 0:
                        continue
            
            selected.append(cache_id)
            selected_regions.add(cache_region)
            cache_usage[cache_id] += size
            
            # Update bandwidth estimate
            for user_id, user_data in user_groups.items():
                user_region = user_data['region']
                if user_region != cache_region:
                    freq = user_data['accesses'][obj_id]
                    region_bw_estimate[user_region] += (size * freq) / 1024.0
        
        for cache_id in selected:
            replication.append((obj_id, cache_id))
    
    return replication

# Multi-strategy search with diverse configurations
print("\\nSearching with advanced strategies...")
best_replication = None
best_cost = float('inf')

strategies = [
    (0.12, 0.30, 2.0),  # Conservative, bandwidth-aware
    (0.18, 0.40, 1.5),  # Balanced
    (0.15, 0.35, 1.8),  # Moderate
    (0.20, 0.45, 1.2),  # Aggressive replication
    (0.10, 0.28, 2.5),  # Very conservative
    (0.25, 0.50, 1.0),  # High coverage
    (0.14, 0.32, 2.2),  # Bandwidth-optimized
]

for i, (top3, top2, bw_factor) in enumerate(strategies):
    try:
        replication = build_advanced_replication(top3, top2, bw_factor)
        if not replication:
            continue
        cost = compute_total_cost(replication)
        print(f"  Strategy {i+1}: {len(replication)} assignments, cost = {cost}")
        if cost < best_cost:
            best_cost = cost
            best_replication = replication
    except Exception as e:
        print(f"  Strategy {i+1}: FAILED - {e}")

print(f"\\nBest initial cost: {best_cost}")

# Intensive local search optimization
print("Applying multi-phase local search...")
max_iterations = 300
iteration = 0
no_improvement_count = 0

while iteration < max_iterations and no_improvement_count < 50:
    iteration += 1
    improved = False
    
    replication_map = collections.defaultdict(list)
    for obj_id, cache_id in best_replication:
        replication_map[obj_id].append(cache_id)
    
    # Phase 1: Try adding replicas to high-value objects
    if iteration % 3 == 0:
        for obj_id in sorted_objects[:max(3, len(sorted_objects)//4)]:
            if len(replication_map[obj_id]) >= 5:
                continue
            
            size = objects[obj_id]['size']
            current_regions = {caches[c]['region'] for c in replication_map[obj_id]}
            
            # Find best cache not in current regions
            best_add = None
            best_add_cost = best_cost
            
            for cache_id, cache_data in caches.items():
                if cache_id in replication_map[obj_id]:
                    continue
                
                cache_region = cache_data['region']
                # Prefer new regions
                if cache_region in current_regions and len(current_regions) < r:
                    continue
                
                used = sum(objects[oid]['size'] for oid, cid in best_replication if cid == cache_id)
                if used + size > cache_data['capacity']:
                    continue
                
                test_rep = list(best_replication) + [(obj_id, cache_id)]
                try:
                    new_cost = compute_total_cost(test_rep)
                    if new_cost < best_add_cost:
                        best_add_cost = new_cost
                        best_add = test_rep
                except:
                    continue
            
            if best_add and best_add_cost < best_cost:
                best_replication = best_add
                best_cost = best_add_cost
                improved = True
                print(f"  Iter {iteration}: Add replica -> {best_cost}")
                break
    
    # Phase 2: Try removing replicas from low-value objects
    if not improved and iteration % 4 == 0:
        for obj_id in reversed(sorted_objects[len(sorted_objects)//3:]):
            if len(replication_map[obj_id]) <= 1:
                continue
            
            # Try removing worst replica
            worst_remove = None
            worst_remove_cost = best_cost
            
            for cache_id in replication_map[obj_id]:
                test_rep = [(o, c) for o, c in best_replication 
                           if not (o == obj_id and c == cache_id)]
                try:
                    new_cost = compute_total_cost(test_rep)
                    if new_cost < worst_remove_cost:
                        worst_remove_cost = new_cost
                        worst_remove = test_rep
                except:
                    continue
            
            if worst_remove and worst_remove_cost < best_cost:
                best_replication = worst_remove
                best_cost = worst_remove_cost
                improved = True
                print(f"  Iter {iteration}: Remove replica -> {best_cost}")
                break
    
    # Phase 3: Try swapping caches
    if not improved and iteration % 5 == 0:
        for obj_id in sorted_objects[:len(sorted_objects)//2]:
            if len(replication_map[obj_id]) == 5:
                continue
            
            size = objects[obj_id]['size']
            for old_cache in replication_map[obj_id]:
                for new_cache, cache_data in caches.items():
                    if new_cache in replication_map[obj_id]:
                        continue
                    
                    used = sum(objects[oid]['size'] for oid, cid in best_replication if cid == new_cache)
                    if used + size > cache_data['capacity']:
                        continue
                    
                    test_rep = [(o, new_cache if (o == obj_id and c == old_cache) else c) 
                               for o, c in best_replication]
                    try:
                        new_cost = compute_total_cost(test_rep)
                        if new_cost < best_cost:
                            best_replication = test_rep
                            best_cost = new_cost
                            improved = True
                            print(f"  Iter {iteration}: Swap cache -> {best_cost}")
                            break
                    except:
                        continue
                if improved:
                    break
            if improved:
                break
    
    if improved:
        no_improvement_count = 0
    else:
        no_improvement_count += 1

print(f"Final cost after {iteration} iterations: {best_cost}")

# Write output as CSV with computed cost
with open("/workdir/sol.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['object_id', 'cache_id', 'total_cost'])
    
    # Sort for determinism
    best_replication.sort(key=lambda x: (x[0], x[1]))
    
    for obj_id, cache_id in best_replication:
        writer.writerow([obj_id, cache_id, best_cost])

print(f"\\nSolution written to sol.csv with total cost: {best_cost}")
PYCODE