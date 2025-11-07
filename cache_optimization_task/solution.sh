#!/bin/bash
set -e

cd /workdir/data

python3 - <<'PYCODE'
import collections
import random
import math
from typing import List, Tuple, Dict, Set

# Parse complex network input with robust error handling
try:
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
            # Clamp latency to valid range [10, 200] to handle invalid test data
            latency = max(10, min(200, latency))
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

except Exception as e:
    print(f"Error parsing input: {e}")
    raise

# Build dependency graph
prereqs = collections.defaultdict(set)
successors = collections.defaultdict(set)
for prereq, dependent in dependencies:
    prereqs[dependent].add(prereq)
    successors[prereq].add(dependent)


def validate_bandwidth_constraints(replication):
    """
    CRITICAL: Validate bandwidth as HARD constraint.
    Returns (is_valid, violations, region_usage).
    """
    replication_map = collections.defaultdict(list)
    for obj_id, cache_id in replication:
        replication_map[obj_id].append(cache_id)
    
    region_usage = {region_id: 0.0 for region_id in regions}
    
    # Calculate bandwidth usage per region
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        for obj_id, freq in user_data['accesses'].items():
            if obj_id not in replication_map:
                continue
            # Check if object is replicated in user's region
            same_region_replicas = [c for c in replication_map[obj_id] 
                                   if caches[c]['region'] == user_region]
            if not same_region_replicas:
                # Cross-region access requires bandwidth
                size_mb = objects[obj_id]['size']
                # Convert MB to GB (freq is accesses per hour)
                region_usage[user_region] += (size_mb * freq) / 1024.0
    
    violations = []
    for region_id, usage in region_usage.items():
        capacity = regions[region_id]['bandwidth']
        if usage > capacity:
            violations.append(
                f"Region {region_id}: {usage:.2f} GB/hr > {capacity:.2f} GB/hr capacity"
            )
    
    return len(violations) == 0, violations, region_usage


def compute_total_cost(replication):
    """Compute full cost with all 4 components."""
    replication_map = collections.defaultdict(list)
    for obj_id, cache_id in replication:
        replication_map[obj_id].append(cache_id)
    
    # 1. Access Cost (with 1.5× cross-region penalty)
    access_cost = 0.0
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        for obj_id, freq in user_data['accesses'].items():
            if not replication_map[obj_id]:
                continue
            min_latency = float('inf')
            for cache_id in replication_map[obj_id]:
                latency = objects[obj_id]['latencies'][cache_id]
                cache_region = caches[cache_id]['region']
                # Apply 1.5× cross-region penalty
                if cache_region == user_region:
                    effective_latency = latency
                else:
                    effective_latency = latency * 1.5
                min_latency = min(min_latency, effective_latency)
            if min_latency != float('inf'):
                pop = objects[obj_id]['popularity']
                access_cost += pop * freq * min_latency
    
    # 2. Replication Cost (exponential penalty: 2^(n-1))
    replication_cost = 0.0
    for obj_id, cache_list in replication_map.items():
        num_replicas = len(cache_list)
        if num_replicas > 0:
            size = objects[obj_id]['size']
            penalty = 2 ** (num_replicas - 1)
            replication_cost += size * penalty * num_replicas
    
    # 3. Cross-Region Transfer Cost
    transfer_cost = 0.0
    for obj_id, cache_list in replication_map.items():
        if obj_id not in prereqs or not cache_list:
            continue
        obj_regions = {caches[c]['region'] for c in cache_list}
        for prereq_id in prereqs[obj_id]:
            if prereq_id not in replication_map:
                continue
            prereq_regions = {caches[c]['region'] for c in replication_map[prereq_id]}
            if not obj_regions.intersection(prereq_regions):
                max_cost = 0
                for obj_reg in obj_regions:
                    for prereq_reg in prereq_regions:
                        cost = transfer_costs.get((prereq_reg, obj_reg), 0)
                        max_cost = max(max_cost, cost)
                size = objects[prereq_id]['size']
                transfer_cost += size * max_cost
    
    # 4. Bandwidth Penalty (quadratic for usage > 80%)
    bandwidth_penalty = 0.0
    region_usage = {region_id: 0.0 for region_id in regions}
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        for obj_id, freq in user_data['accesses'].items():
            if obj_id not in replication_map:
                continue
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
# Formula derived from specification: popularity × total_accesses × (1 + dependency_factor)
importance = {}
for obj_id in objects:
    pop = objects[obj_id]['popularity']
    size = objects[obj_id]['size']
    total_accesses = sum(ug['accesses'].get(obj_id, 0) for ug in user_groups.values())
    num_dependents = len(successors[obj_id])
    num_prereqs = len(prereqs[obj_id])
    
    # Weight by: popularity (given), access frequency (observable), 
    # and dependency importance (derived from dependency graph)
    importance[obj_id] = (pop * total_accesses * 
                         (1 + num_dependents * 0.3) * (1 + num_prereqs * 0.1))

sorted_objects = sorted(objects.keys(), key=lambda x: -importance[x])

# Calculate region affinity for each object (from access patterns)
region_affinity = {}
for obj_id in objects:
    affinity = collections.defaultdict(float)
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        freq = user_data['accesses'].get(obj_id, 0)
        affinity[user_region] += freq
    region_affinity[obj_id] = affinity


def build_replication_strategy(config):
    """
    Build replication strategy that ENFORCES bandwidth constraints.
    
    config = (top_k_pct_double, top_k_pct_triple, diversity_weight)
    
    Strategy:
    1. Determine replica count based on importance percentile
    2. Select caches considering: region affinity, latency, capacity, and cross-region penalty
    3. VALIDATE bandwidth constraints - if violated, reduce cross-region replication
    """
    top_double, top_triple, diversity_weight = config
    
    replication = []
    cache_usage = {cache_id: 0 for cache_id in caches}
    
    for rank, obj_id in enumerate(sorted_objects):
        size = objects[obj_id]['size']
        
        # Determine number of replicas based on importance percentile
        pct_rank = rank / len(sorted_objects)
        if pct_rank < top_triple:
            num_replicas = 3
        elif pct_rank < top_double:
            num_replicas = 2
        else:
            num_replicas = 1
        
        # Get region affinity (which regions access this object most)
        affinity = region_affinity[obj_id]
        
        # Calculate score for each cache
        cache_scores = []
        for cache_id, cache_data in caches.items():
            if cache_usage[cache_id] + size > cache_data['capacity']:
                continue
            
            cache_region = cache_data['region']
            
            # Score based on: access frequency, latency (with 1.5× penalty), and utilization
            access_benefit = 0.0
            for user_id, user_data in user_groups.items():
                user_region = user_data['region']
                freq = user_data['accesses'].get(obj_id, 0)
                latency = objects[obj_id]['latencies'][cache_id]
                
                # Consider 1.5× cross-region penalty in scoring
                if cache_region == user_region:
                    effective_latency = latency
                    benefit_multiplier = 1.0
                else:
                    effective_latency = latency * 1.5
                    benefit_multiplier = 0.4  # Reduce benefit for cross-region
                
                access_benefit += freq * (250 - effective_latency) * benefit_multiplier
            
            # Prefer less utilized caches
            utilization = cache_usage[cache_id] / cache_data['capacity']
            utilization_penalty = 1 + utilization * 3
            
            score = access_benefit / utilization_penalty
            cache_scores.append((score, cache_id, cache_region))
        
        if not cache_scores:
            continue
        
        cache_scores.sort(reverse=True)
        
        # Select caches, preferring different regions for replicas
        selected = []
        selected_regions = set()
        
        # First pass: try to get diverse regions
        for score, cache_id, cache_region in cache_scores:
            if len(selected) >= num_replicas:
                break
            # For multiple replicas, strongly prefer different regions
            if num_replicas > 1 and cache_region in selected_regions and len(selected) < num_replicas:
                continue
            selected.append(cache_id)
            selected_regions.add(cache_region)
            cache_usage[cache_id] += size
        
        # Second pass: fill remaining slots if needed
        if len(selected) < num_replicas:
            for score, cache_id, cache_region in cache_scores:
                if cache_id in selected:
                    continue
                if cache_usage[cache_id] + size > caches[cache_id]['capacity']:
                    continue
                selected.append(cache_id)
                cache_usage[cache_id] += size
                if len(selected) >= num_replicas:
                    break
        
        for cache_id in selected:
            replication.append((obj_id, cache_id))
    
    return replication


def enforce_bandwidth_constraints(replication):
    """
    CRITICAL: Ensure replication satisfies bandwidth constraints.
    If violated, iteratively remove cross-region replicas until valid.
    """
    max_iterations = 100
    iteration = 0
    
    while iteration < max_iterations:
        is_valid, violations, usage = validate_bandwidth_constraints(replication)
        
        if is_valid:
            return replication  # Success!
        
        # Find regions exceeding bandwidth
        over_capacity_regions = set()
        for region_id, bw_usage in usage.items():
            if bw_usage > regions[region_id]['bandwidth']:
                over_capacity_regions.add(region_id)
        
        if not over_capacity_regions:
            break
        
        # Build replication map
        replication_map = collections.defaultdict(list)
        for obj_id, cache_id in replication:
            replication_map[obj_id].append(cache_id)
        
        # Find candidates to remove: objects with replicas in over-capacity regions
        # that have other replicas elsewhere
        candidates = []
        for obj_id, cache_list in replication_map.items():
            if len(cache_list) <= 1:
                continue  # Must keep at least one replica
            
            for cache_id in cache_list:
                cache_region = caches[cache_id]['region']
                if cache_region not in over_capacity_regions:
                    continue
                
                # Calculate bandwidth reduction if we remove this replica
                reduction = 0.0
                for user_id, user_data in user_groups.items():
                    user_region = user_data['region']
                    if user_region != cache_region:
                        continue
                    
                    freq = user_data['accesses'].get(obj_id, 0)
                    # Check if removing this replica would force cross-region access
                    other_replicas = [c for c in cache_list if c != cache_id]
                    same_region_others = [c for c in other_replicas 
                                         if caches[c]['region'] == cache_region]
                    
                    if not same_region_others:
                        # This removal would create cross-region access
                        size_mb = objects[obj_id]['size']
                        reduction += (size_mb * freq) / 1024.0
                
                if reduction > 0:
                    # Score: how much bandwidth we save per unit of cost increase
                    candidates.append((reduction, obj_id, cache_id))
        
        if not candidates:
            # Can't fix bandwidth constraint by removing replicas
            print(f"WARNING: Cannot satisfy bandwidth constraints")
            break
        
        # Remove the replica that gives best bandwidth reduction
        candidates.sort(reverse=True)
        _, obj_id, cache_id = candidates[0]
        replication = [(o, c) for o, c in replication if not (o == obj_id and c == cache_id)]
        
        iteration += 1
        print(f"  Bandwidth fix iteration {iteration}: removed {obj_id} from {cache_id}")
    
    return replication


# Multi-start optimization with diverse strategies
print("\nSearching across multiple strategies...")
best_replication = None
best_cost = float('inf')

# Parameterization based on observable problem characteristics:
# - top_double: fraction of objects to replicate twice
# - top_triple: fraction of objects to replicate three times
# - diversity_weight: preference for region diversity
strategies = [
    (0.10, 0.03, 0.8),  # Very conservative replication
    (0.20, 0.08, 0.7),  # Conservative
    (0.30, 0.12, 0.6),  # Moderate
    (0.40, 0.18, 0.5),  # Aggressive
    (0.25, 0.10, 0.65), # Balanced
    (0.35, 0.15, 0.55), # High coverage
    (0.15, 0.05, 0.75), # Low replication
]

for i, config in enumerate(strategies):
    try:
        replication = build_replication_strategy(config)
        if not replication:
            print(f"  Strategy {i+1}: No valid replication found")
            continue
        
        # CRITICAL: Enforce bandwidth constraints
        replication = enforce_bandwidth_constraints(replication)
        
        # Validate constraints are satisfied
        is_valid, violations, _ = validate_bandwidth_constraints(replication)
        if not is_valid:
            print(f"  Strategy {i+1}: FAILED bandwidth validation")
            continue
        
        cost = compute_total_cost(replication)
        print(f"  Strategy {i+1} (double={config[0]:.0%}, triple={config[1]:.0%}): cost = {cost}")
        if cost < best_cost:
            best_cost = cost
            best_replication = replication
    except Exception as e:
        print(f"  Strategy {i+1}: FAILED - {e}")

if best_replication is None:
    print("ERROR: No valid initial strategy found")
    # Create minimal valid solution
    best_replication = []
    cache_usage = {cache_id: 0 for cache_id in caches}
    for obj_id in objects:
        size = objects[obj_id]['size']
        for cache_id in caches:
            if cache_usage[cache_id] + size <= caches[cache_id]['capacity']:
                best_replication.append((obj_id, cache_id))
                cache_usage[cache_id] += size
                break
    best_cost = compute_total_cost(best_replication)

print(f"\nBest initial cost: {best_cost}")

# Validate bandwidth before local search
is_valid, violations, _ = validate_bandwidth_constraints(best_replication)
if not is_valid:
    print("Enforcing bandwidth constraints on initial solution...")
    best_replication = enforce_bandwidth_constraints(best_replication)
    best_cost = compute_total_cost(best_replication)
    print(f"After enforcement: {best_cost}")

# Local search optimization
print("Applying local search optimization...")
iterations = 0
max_iterations = 300
no_improvement_count = 0

while iterations < max_iterations and no_improvement_count < 50:
    improved = False
    iterations += 1
    
    replication_map = collections.defaultdict(list)
    for obj_id, cache_id in best_replication:
        replication_map[obj_id].append(cache_id)
    
    # Strategy 1: Add replicas to high-value objects
    if iterations % 3 == 0:
        for obj_id in sorted_objects[:max(1, len(sorted_objects)//4)]:
            if len(replication_map[obj_id]) >= 5:
                continue
            
            size = objects[obj_id]['size']
            current_regions = {caches[c]['region'] for c in replication_map[obj_id]}
            
            for cache_id, cache_data in sorted(caches.items()):
                if cache_id in replication_map[obj_id]:
                    continue
                cache_region = cache_data['region']
                
                if cache_region in current_regions and len(replication_map[obj_id]) >= 2:
                    continue
                
                used = sum(objects[oid]['size'] for oid, cid in best_replication if cid == cache_id)
                if used + size > cache_data['capacity']:
                    continue
                
                test_replication = list(best_replication) + [(obj_id, cache_id)]
                
                # CRITICAL: Check bandwidth constraints
                is_valid, _, _ = validate_bandwidth_constraints(test_replication)
                if not is_valid:
                    continue
                
                try:
                    new_cost = compute_total_cost(test_replication)
                    if new_cost < best_cost:
                        best_replication = test_replication
                        best_cost = new_cost
                        improved = True
                        print(f"  Iteration {iterations}: improved to {best_cost} (added replica)")
                        break
                except:
                    continue
            if improved:
                break
    
    # Strategy 2: Remove replicas from low-value objects
    if not improved and iterations % 4 == 0:
        for obj_id in reversed(sorted_objects[max(1, len(sorted_objects)//3):]):
            if len(replication_map[obj_id]) <= 1:
                continue
            
            for cache_id in list(replication_map[obj_id]):
                test_replication = [(o, c) for o, c in best_replication 
                                   if not (o == obj_id and c == cache_id)]
                try:
                    new_cost = compute_total_cost(test_replication)
                    if new_cost < best_cost:
                        best_replication = test_replication
                        best_cost = new_cost
                        improved = True
                        print(f"  Iteration {iterations}: improved to {best_cost} (removed replica)")
                        break
                except:
                    continue
            if improved:
                break
    
    # Strategy 3: Swap cache locations
    if not improved and iterations % 5 == 0:
        for obj_id in sorted_objects[:max(1, len(sorted_objects)//2)]:
            if len(replication_map[obj_id]) == 0:
                continue
            
            size = objects[obj_id]['size']
            current_cache = replication_map[obj_id][0]
            
            for new_cache_id in caches:
                if new_cache_id in replication_map[obj_id]:
                    continue
                
                used = sum(objects[oid]['size'] for oid, cid in best_replication if cid == new_cache_id)
                if used + size > caches[new_cache_id]['capacity']:
                    continue
                
                test_replication = [(o, c if not (o == obj_id and c == current_cache) else new_cache_id) 
                                   for o, c in best_replication]
                
                # CRITICAL: Check bandwidth constraints
                is_valid, _, _ = validate_bandwidth_constraints(test_replication)
                if not is_valid:
                    continue
                
                try:
                    new_cost = compute_total_cost(test_replication)
                    if new_cost < best_cost:
                        best_replication = test_replication
                        best_cost = new_cost
                        improved = True
                        print(f"  Iteration {iterations}: improved to {best_cost} (swapped cache)")
                        break
                except:
                    continue
            if improved:
                break
    
    if improved:
        no_improvement_count = 0
    else:
        no_improvement_count += 1

print(f"Final cost after {iterations} iterations: {best_cost}")

# FINAL VALIDATION: Ensure bandwidth constraints are satisfied
is_valid, violations, usage = validate_bandwidth_constraints(best_replication)
if not is_valid:
    print("CRITICAL: Final solution violates bandwidth constraints!")
    print("Violations:")
    for v in violations:
        print(f"  {v}")
    print("Enforcing constraints...")
    best_replication = enforce_bandwidth_constraints(best_replication)
    best_cost = compute_total_cost(best_replication)
    print(f"Final cost after enforcement: {best_cost}")
    
    # Re-validate
    is_valid, violations, usage = validate_bandwidth_constraints(best_replication)
    if not is_valid:
        print("ERROR: Could not satisfy bandwidth constraints!")
        for v in violations:
            print(f"  {v}")
else:
    print("✓ Bandwidth constraints satisfied")
    for region_id, bw_usage in usage.items():
        capacity = regions[region_id]['bandwidth']
        pct = (bw_usage / capacity * 100) if capacity > 0 else 0
        print(f"  Region {region_id}: {bw_usage:.2f}/{capacity:.2f} GB/hr ({pct:.1f}%)")

# Write outputs
with open("/workdir/replication.txt", "w") as f:
    for obj_id, cache_id in best_replication:
        f.write(f"{obj_id} {cache_id}\n")

with open("/workdir/ans.txt", "w") as f:
    f.write(f"{best_cost}\n")

print(f"\nTotal cost: {best_cost}")
print(f"Total assignments: {len(best_replication)}")
print("Replication strategy written successfully")
PYCODE