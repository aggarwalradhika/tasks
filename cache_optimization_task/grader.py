import os
import collections
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set

from apex_arena._types import GradingResult


def parse_network_file(filepath: Path):
    """
    Parse the complex network.txt input file.
    
    Returns:
        Tuple containing all parsed data structures
    """
    with open(filepath) as f:
        # Header
        header = f.readline().strip().split()
        if len(header) != 9:
            raise ValueError(f"Header must have 9 values, got {len(header)}")
        n, m, r, u, d = map(int, header[:5])
        alpha = list(map(float, header[5:]))
        
        # Regions
        regions = {}
        for _ in range(r):
            parts = f.readline().strip().split()
            if len(parts) != 3:
                raise ValueError(f"Region line must have 3 values")
            region_id, num_caches, bandwidth = parts[0], int(parts[1]), float(parts[2])
            regions[region_id] = {'num_caches': num_caches, 'bandwidth': bandwidth}
        
        # Caches
        caches = {}
        for _ in range(n):
            parts = f.readline().strip().split()
            if len(parts) != 3:
                raise ValueError(f"Cache line must have 3 values")
            cache_id, region_id, capacity = parts[0], parts[1], int(parts[2])
            if region_id not in regions:
                raise ValueError(f"Unknown region {region_id} for cache {cache_id}")
            caches[cache_id] = {'region': region_id, 'capacity': capacity}
        
        # Objects
        objects = {}
        for _ in range(m):
            parts = f.readline().strip().split()
            if len(parts) != 3:
                raise ValueError(f"Object line must have 3 values")
            obj_id, pop, size = parts[0], int(parts[1]), int(parts[2])
            if pop not in {1, 2, 3, 4, 5}:
                raise ValueError(f"Invalid popularity {pop} for {obj_id}")
            if not (5 <= size <= 100):
                raise ValueError(f"Invalid size {size} for {obj_id}")
            objects[obj_id] = {'popularity': pop, 'size': size, 'latencies': {}}
        
        # Object-Cache Latencies
        for _ in range(n * m):
            parts = f.readline().strip().split()
            if len(parts) != 3:
                raise ValueError(f"Latency line must have 3 values")
            obj_id, cache_id, latency = parts[0], parts[1], int(parts[2])
            if obj_id not in objects:
                raise ValueError(f"Unknown object {obj_id} in latency spec")
            if cache_id not in caches:
                raise ValueError(f"Unknown cache {cache_id} in latency spec")
            if not (10 <= latency <= 200):
                raise ValueError(f"Invalid latency {latency} for {obj_id} on {cache_id}")
            objects[obj_id]['latencies'][cache_id] = latency
        
        # User Groups
        user_groups = {}
        for _ in range(u):
            parts = f.readline().strip().split()
            if len(parts) != 3:
                raise ValueError(f"User group line must have 3 values")
            user_id, region_id, volume = parts[0], parts[1], int(parts[2])
            if region_id not in regions:
                raise ValueError(f"Unknown region {region_id} for user {user_id}")
            user_groups[user_id] = {'region': region_id, 'volume': volume, 'accesses': {}}
        
        # Access Patterns
        total_accesses = u * m
        for _ in range(total_accesses):
            parts = f.readline().strip().split()
            if len(parts) != 3:
                raise ValueError(f"Access pattern line must have 3 values")
            user_id, obj_id, freq = parts[0], parts[1], int(parts[2])
            if user_id not in user_groups:
                raise ValueError(f"Unknown user {user_id} in access pattern")
            if obj_id not in objects:
                raise ValueError(f"Unknown object {obj_id} in access pattern")
            user_groups[user_id]['accesses'][obj_id] = freq
        
        # Transfer Costs
        transfer_costs = {}
        for _ in range(r * r):
            parts = f.readline().strip().split()
            if len(parts) != 3:
                raise ValueError(f"Transfer cost line must have 3 values")
            region_a, region_b, cost = parts[0], parts[1], float(parts[2])
            if region_a not in regions or region_b not in regions:
                raise ValueError(f"Unknown regions in transfer cost: {region_a}, {region_b}")
            transfer_costs[(region_a, region_b)] = cost
        
        # Dependencies (after blank line)
        line = f.readline()
        if line.strip():
            raise ValueError("Expected blank line before dependencies")
        
        dependencies = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid dependency line: {line}")
            prereq, dependent = parts
            if prereq not in objects or dependent not in objects:
                raise ValueError(f"Unknown objects in dependency: {prereq}, {dependent}")
            dependencies.append((prereq, dependent))
        
        if len(dependencies) != d:
            raise ValueError(f"Expected {d} dependencies, got {len(dependencies)}")
        
        return n, m, r, u, d, alpha, regions, caches, objects, user_groups, transfer_costs, dependencies


def validate_dependencies_dag(dependencies: List[Tuple[str, str]], all_objects: Set[str]) -> Tuple[bool, str]:
    """Validate DAG property."""
    graph = collections.defaultdict(list)
    for prereq, dependent in dependencies:
        graph[prereq].append(dependent)
    
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {obj: WHITE for obj in all_objects}
    
    def has_cycle(node):
        if color[node] == GRAY:
            return True
        if color[node] == BLACK:
            return False
        color[node] = GRAY
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        color[node] = BLACK
        return False
    
    for obj in all_objects:
        if color[obj] == WHITE:
            if has_cycle(obj):
                return False, "Dependencies contain a cycle"
    return True, ""


def compute_total_cost(replication, n, m, r, u, alpha, regions, caches, objects, 
                      user_groups, transfer_costs, dependencies):
    """
    Compute total cost with all four components.
    """
    # Build replication map: obj_id -> list of cache_ids
    replication_map = collections.defaultdict(list)
    for obj_id, cache_id in replication:
        replication_map[obj_id].append(cache_id)
    
    # Build dependency graph
    prereqs = collections.defaultdict(set)
    for prereq, dependent in dependencies:
        prereqs[dependent].add(prereq)
    
    # 1. ACCESS LATENCY COST
    access_cost = 0.0
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        for obj_id, freq in user_data['accesses'].items():
            if obj_id not in replication_map:
                raise ValueError(f"Object {obj_id} not replicated anywhere")
            
            # Find nearest replica
            min_latency = float('inf')
            for cache_id in replication_map[obj_id]:
                latency = objects[obj_id]['latencies'][cache_id]
                cache_region = caches[cache_id]['region']
                # Prefer same region (no cross-region penalty)
                if cache_region == user_region:
                    min_latency = min(min_latency, latency)
                else:
                    # Add cross-region penalty
                    min_latency = min(min_latency, latency * 1.5)
            
            pop = objects[obj_id]['popularity']
            access_cost += pop * freq * min_latency
    
    # 2. REPLICATION COST
    replication_cost = 0.0
    for obj_id, cache_list in replication_map.items():
        num_replicas = len(cache_list)
        size = objects[obj_id]['size']
        penalty = 2 ** (num_replicas - 1)
        replication_cost += size * penalty * num_replicas
    
    # 3. CROSS-REGION TRANSFER COST
    transfer_cost = 0.0
    for obj_id, cache_list in replication_map.items():
        if obj_id not in prereqs:
            continue
        # For each dependency, check if it's in a different region
        obj_regions = {caches[c]['region'] for c in cache_list}
        for prereq_id in prereqs[obj_id]:
            prereq_regions = {caches[c]['region'] for c in replication_map[prereq_id]}
            # If no overlap in regions, incur transfer cost
            if not obj_regions.intersection(prereq_regions):
                # Pick worst case: max cost
                max_cost = 0
                for obj_reg in obj_regions:
                    for prereq_reg in prereq_regions:
                        cost = transfer_costs.get((prereq_reg, obj_reg), 0)
                        max_cost = max(max_cost, cost)
                size = objects[prereq_id]['size']
                transfer_cost += size * max_cost
    
    # 4. BANDWIDTH SATURATION PENALTY
    bandwidth_penalty = 0.0
    region_usage = {region_id: 0.0 for region_id in regions}
    
    # Calculate bandwidth usage per region (simplified: size * access frequency)
    for user_id, user_data in user_groups.items():
        user_region = user_data['region']
        for obj_id, freq in user_data['accesses'].items():
            # Find if object is in same region
            same_region_replicas = [c for c in replication_map[obj_id] 
                                   if caches[c]['region'] == user_region]
            if not same_region_replicas:
                # Cross-region access, add to bandwidth
                size_mb = objects[obj_id]['size']
                # Convert to GB (freq is per hour)
                region_usage[user_region] += (size_mb * freq) / 1024.0
    
    for region_id, usage in region_usage.items():
        capacity = regions[region_id]['bandwidth']
        if usage > 0.8 * capacity:
            excess = usage - 0.8 * capacity
            bandwidth_penalty += excess * excess * 100
    
    # Total cost with weights
    total = (alpha[0] * access_cost + 
             alpha[1] * replication_cost + 
             alpha[2] * transfer_cost + 
             alpha[3] * bandwidth_penalty)
    
    return int(round(total))


def sophisticated_baseline(n, m, r, u, alpha, regions, caches, objects, 
                          user_groups, transfer_costs, dependencies):
    """
    Sophisticated baseline for multi-region replication.
    
    Strategy:
    1. Identify high-impact objects (popularity × total access volume)
    2. Replicate top-k% to multiple regions
    3. Single-replicate everything else to best location
    4. Use greedy capacity-aware placement
    """
    # Calculate importance score for each object
    importance = {}
    for obj_id, obj_data in objects.items():
        pop = obj_data['popularity']
        total_accesses = sum(ug['accesses'].get(obj_id, 0) for ug in user_groups.values())
        importance[obj_id] = pop * total_accesses
    
    # Sort by importance
    sorted_objects = sorted(objects.keys(), key=lambda x: -importance[x])
    
    # Determine replication strategy
    # Top 20% get 2 replicas, rest get 1
    top_20_percent = max(1, m // 5)
    
    replication = []
    cache_usage = {cache_id: 0 for cache_id in caches}
    
    for rank, obj_id in enumerate(sorted_objects):
        size = objects[obj_id]['size']
        num_replicas = 2 if rank < top_20_percent else 1
        
        # Find best caches for this object
        # Score each cache by: latency cost to users + current usage
        cache_scores = []
        for cache_id, cache_data in caches.items():
            if cache_usage[cache_id] + size > cache_data['capacity']:
                continue
            
            # Calculate expected access cost if placed here
            cost = 0
            cache_region = cache_data['region']
            for user_id, user_data in user_groups.items():
                user_region = user_data['region']
                freq = user_data['accesses'].get(obj_id, 0)
                if freq == 0:
                    continue
                latency = objects[obj_id]['latencies'][cache_id]
                if cache_region != user_region:
                    latency *= 1.5
                cost += latency * freq
            
            # Add penalty for high utilization
            utilization = cache_usage[cache_id] / cache_data['capacity']
            score = cost * (1 + utilization)
            cache_scores.append((score, cache_id))
        
        cache_scores.sort()
        selected = []
        for _, cache_id in cache_scores[:num_replicas]:
            replication.append((obj_id, cache_id))
            cache_usage[cache_id] += size
            selected.append(cache_id)
            if len(selected) >= num_replicas:
                break
        
        if len(selected) < num_replicas:
            # Couldn't fit all replicas, just place what we could
            pass
    
    # Compute score
    score = compute_total_cost(replication, n, m, r, u, alpha, regions, caches, 
                              objects, user_groups, transfer_costs, dependencies)
    return score, replication


def grade(transcript: str) -> GradingResult:
    """
    Grade the multi-region cache replication submission.
    """
    feedback_messages: List[str] = []
    subscores = {"correct_answer": 0.0}
    weights = {"correct_answer": 1.0}

    try:
        replication_path = Path("/workdir/replication.txt")
        answer_path = Path("/workdir/ans.txt")
        data_path = Path("/workdir/data/network.txt")

        # CHECK 1: Files exist
        for p, name in [(data_path, "Input"), (replication_path, "Replication"), (answer_path, "Answer")]:
            if not p.exists():
                feedback_messages.append(f"FAIL: {name} file {p} does not exist")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        # Parse input
        n, m, r, u, d, alpha, regions, caches, objects, user_groups, transfer_costs, dependencies = parse_network_file(data_path)
        feedback_messages.append(f"Problem: {m} objects, {n} caches, {r} regions, {u} user groups, {d} dependencies")

        # CHECK 2: Dependencies form DAG
        is_dag, err = validate_dependencies_dag(dependencies, set(objects.keys()))
        if not is_dag:
            feedback_messages.append(f"FAIL: Invalid dependencies - {err}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        feedback_messages.append("Dependencies form a valid DAG")

        # CHECK 3: Parse replication file
        replication = []
        with open(replication_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    feedback_messages.append(f"FAIL: Invalid replication format at line {line_num}")
                    return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                         feedback="; ".join(feedback_messages))
                obj_id, cache_id = parts
                if obj_id not in objects:
                    feedback_messages.append(f"FAIL: Unknown object {obj_id}")
                    return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                         feedback="; ".join(feedback_messages))
                if cache_id not in caches:
                    feedback_messages.append(f"FAIL: Unknown cache {cache_id}")
                    return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                         feedback="; ".join(feedback_messages))
                replication.append((obj_id, cache_id))

        feedback_messages.append(f"Replication format valid: {len(replication)} assignments")

        # CHECK 4: Every object replicated at least once, at most 5 times
        replication_counts = collections.Counter(obj_id for obj_id, _ in replication)
        for obj_id in objects:
            count = replication_counts.get(obj_id, 0)
            if count == 0:
                feedback_messages.append(f"FAIL: Object {obj_id} not replicated anywhere")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))
            if count > 5:
                feedback_messages.append(f"FAIL: Object {obj_id} replicated {count} times (max 5)")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))
        
        feedback_messages.append("All objects replicated (1-5 times each)")

        # CHECK 5: Capacity constraints
        cache_usage = {cache_id: 0 for cache_id in caches}
        for obj_id, cache_id in replication:
            size = objects[obj_id]['size']
            cache_usage[cache_id] += size
        
        violations = []
        for cache_id, usage in cache_usage.items():
            capacity = caches[cache_id]['capacity']
            if usage > capacity:
                violations.append(f"{cache_id}: {usage}MB > {capacity}MB")
        
        if violations:
            feedback_messages.append(f"FAIL: Capacity violations: {'; '.join(violations)}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        
        feedback_messages.append("Capacity constraints satisfied")

        # CHECK 6: Parse claimed answer
        try:
            with open(answer_path, "r") as f:
                claimed_answer = int(f.read().strip())
            feedback_messages.append(f"Claimed answer: {claimed_answer}")
        except Exception as e:
            feedback_messages.append(f"FAIL: Invalid ans.txt format - {e}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # CHECK 7: Compute actual cost
        try:
            actual_cost = compute_total_cost(replication, n, m, r, u, alpha, regions, 
                                            caches, objects, user_groups, transfer_costs, dependencies)
            feedback_messages.append(f"Actual computed cost: {actual_cost}")
        except Exception as e:
            feedback_messages.append(f"FAIL: Cost computation error - {e}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # CHECK 8: Claimed vs actual
        if claimed_answer != actual_cost:
            feedback_messages.append(f"FAIL: Claimed ({claimed_answer}) != actual ({actual_cost})")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        
        feedback_messages.append("Claimed answer matches computed cost")

        # CHECK 9: Near-optimality
        baseline_cost, baseline_replication = sophisticated_baseline(
            n, m, r, u, alpha, regions, caches, objects, user_groups, transfer_costs, dependencies)
        feedback_messages.append(f"Baseline cost: {baseline_cost}")

        tolerance = float(os.getenv("GRADER_NEAR_OPT_TOL", "1.30"))
        if tolerance <= 1.0:
            tolerance = 1.30

        feedback_messages.append(f"Near-optimality tolerance: {tolerance:.2f}x baseline")

        if baseline_cost == 0:
            if actual_cost != 0:
                feedback_messages.append("FAIL: Baseline is 0 but solution > 0")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))
        else:
            threshold = baseline_cost * tolerance
            if actual_cost > threshold:
                feedback_messages.append(
                    f"FAIL: Not near-optimal: {actual_cost} > {tolerance:.2f} × {baseline_cost} = {threshold:.1f}"
                )
                feedback_messages.append(f"Your solution is {(actual_cost/baseline_cost):.2f}x baseline")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        feedback_messages.append(f"Solution is near-optimal ({(actual_cost/baseline_cost):.3f}x baseline)")

        # SUCCESS
        subscores["correct_answer"] = 1.0
        feedback_messages.append("ALL CHECKS PASSED!")
        return GradingResult(score=1.0, subscores=subscores, weights=weights,
                             feedback="; ".join(feedback_messages))

    except Exception as e:
        import traceback
        feedback_messages.append(f"UNEXPECTED ERROR: {e}")
        feedback_messages.append(traceback.format_exc())
        return GradingResult(score=0.0, subscores=subscores, weights=weights,
                             feedback="; ".join(feedback_messages))