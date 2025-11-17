#!/bin/bash
set -e

cd /workdir/data

python3 - <<'PYCODE'
import itertools
import math
from typing import Dict, List, Tuple, Set

# Parse input
with open("pipeline.txt") as f:
    first_line = f.readline().strip().split()
    num_stages, num_nodes, throughput_req, reliability_req = int(first_line[0]), int(first_line[1]), float(first_line[2]), float(first_line[3])
    
    stages = {}
    for _ in range(num_stages):
        parts = f.readline().split()
        stage_id, complexity = parts[0], float(parts[1])
        stages[stage_id] = complexity
    
    nodes = {}
    for _ in range(num_nodes):
        parts = f.readline().split()
        node_id = parts[0]
        capacity, reliability, speed = float(parts[1]), float(parts[2]), float(parts[3])
        nodes[node_id] = (capacity, reliability, speed)
    
    f.readline()  # blank line
    
    network = {}
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        from_node, to_node, latency = parts[0], parts[1], float(parts[2])
        network[(from_node, to_node)] = latency
        network[(to_node, from_node)] = latency

print(f"Parsed: {num_stages} stages, {num_nodes} nodes")
print(f"Requirements: throughput={throughput_req} events/sec, reliability≥{reliability_req}")

stage_ids = sorted(stages.keys())
node_ids = sorted(nodes.keys())

def compute_processing_time(stage_complexity: float, node_speed: float) -> float:
    """Processing time = node_speed * stage_complexity / 10"""
    return node_speed * stage_complexity / 10.0

def compute_stage_reliability(node_list: List[str]) -> float:
    """Compute reliability with redundancy: 1 - product(1 - r_i)"""
    if not node_list:
        return 0.0
    failure_prob = 1.0
    for node_id in node_list:
        node_reliability = nodes[node_id][1]
        failure_prob *= (1.0 - node_reliability)
    return 1.0 - failure_prob

def compute_latency(deployment: Dict[str, List[str]]) -> float:
    """Compute average latency across all paths through pipeline"""
    # Generate all possible paths
    path_combinations = [deployment[stage_id] for stage_id in stage_ids]
    all_paths = list(itertools.product(*path_combinations))
    
    if not all_paths:
        return float('inf')
    
    path_latencies = []
    for path in all_paths:
        latency = 0.0
        
        # Processing times
        for i, stage_id in enumerate(stage_ids):
            node_id = path[i]
            stage_complexity = stages[stage_id]
            node_speed = nodes[node_id][2]
            latency += compute_processing_time(stage_complexity, node_speed)
        
        # Network transfers
        for i in range(len(stage_ids) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            if from_node != to_node:
                latency += network.get((from_node, to_node), 0)
        
        path_latencies.append(latency)
    
    return sum(path_latencies) / len(path_latencies)

def check_capacity(deployment: Dict[str, List[str]]) -> bool:
    """Check if deployment satisfies capacity constraints"""
    node_loads = {node_id: 0.0 for node_id in node_ids}
    
    for stage_id in stage_ids:
        stage_complexity = stages[stage_id]
        node_list = deployment[stage_id]
        
        if not node_list:
            return False
        
        load_per_node = 1.0 / len(node_list)
        
        for node_id in node_list:
            node_speed = nodes[node_id][2]
            processing_time = compute_processing_time(stage_complexity, node_speed)
            stage_load = processing_time * throughput_req / 1000.0 * load_per_node
            node_loads[node_id] += stage_load
    
    for node_id, load in node_loads.items():
        if load > nodes[node_id][0] + 1e-6:
            return False
    
    return True

def check_reliability(deployment: Dict[str, List[str]]) -> bool:
    """Check if deployment meets reliability requirement"""
    pipeline_reliability = 1.0
    for stage_id in stage_ids:
        stage_rel = compute_stage_reliability(deployment[stage_id])
        pipeline_reliability *= stage_rel
    return pipeline_reliability >= reliability_req - 1e-6

print("Starting greedy deployment with optimization...")

# PHASE 1: Greedy initial assignment
deployment = {stage_id: [] for stage_id in stage_ids}
node_loads = {node_id: 0.0 for node_id in node_ids}

for i, stage_id in enumerate(stage_ids):
    stage_complexity = stages[stage_id]
    
    best_node = None
    best_score = float('inf')
    
    for node_id in node_ids:
        node_capacity, node_reliability, node_speed = nodes[node_id]
        processing_time = compute_processing_time(stage_complexity, node_speed)
        stage_load = processing_time * throughput_req / 1000.0
        
        # Check capacity
        if node_loads[node_id] + stage_load > node_capacity:
            continue
        
        # Score: processing time + network transfer + load balancing penalty
        score = processing_time
        
        # Network transfer cost
        if i > 0:
            prev_stage = stage_ids[i - 1]
            prev_nodes = deployment[prev_stage]
            
            if prev_nodes:
                min_transfer = float('inf')
                for prev_node in prev_nodes:
                    if prev_node == node_id:
                        min_transfer = 0
                        break
                    min_transfer = min(min_transfer, network.get((prev_node, node_id), 1000))
                score += min_transfer
        
        # Load balancing penalty
        score += node_loads[node_id] * 0.3
        
        if score < best_score:
            best_score = score
            best_node = node_id
    
    if best_node:
        deployment[stage_id].append(best_node)
        stage_load = compute_processing_time(stage_complexity, nodes[best_node][2]) * throughput_req / 1000.0
        node_loads[best_node] += stage_load
    else:
        # Fallback: least loaded node
        least_loaded = min(node_ids, key=lambda n: node_loads[n])
        deployment[stage_id].append(least_loaded)

print(f"Initial assignment: {[(sid, nds) for sid, nds in deployment.items()]}")

# PHASE 2: Add redundancy for reliability
current_reliability = 1.0
for stage_id in stage_ids:
    stage_rel = compute_stage_reliability(deployment[stage_id])
    current_reliability *= stage_rel

print(f"Initial reliability: {current_reliability:.6f} (requirement: {reliability_req})")

if current_reliability < reliability_req:
    print("Adding redundancy to meet reliability requirement...")
    
    # Sort stages by reliability (weakest first)
    stage_rels = []
    for stage_id in stage_ids:
        stage_rel = compute_stage_reliability(deployment[stage_id])
        stage_rels.append((stage_rel, stage_id))
    stage_rels.sort()
    
    for stage_rel, stage_id in stage_rels:
        if check_reliability(deployment):
            break
        
        stage_complexity = stages[stage_id]
        
        # Try adding most reliable available node
        candidates = []
        for node_id in node_ids:
            if node_id not in deployment[stage_id]:
                candidates.append((nodes[node_id][1], node_id))
        
        candidates.sort(reverse=True)
        
        for _, node_id in candidates:
            # Test if we can add this node
            test_deployment = {k: list(v) for k, v in deployment.items()}
            test_deployment[stage_id].append(node_id)
            
            if check_capacity(test_deployment):
                deployment[stage_id].append(node_id)
                print(f"  Added {node_id} to {stage_id}")
                
                # Update loads
                node_loads = {node_id: 0.0 for node_id in node_ids}
                for sid in stage_ids:
                    sc = stages[sid]
                    nl = deployment[sid]
                    load_per = 1.0 / len(nl)
                    for nid in nl:
                        ns = nodes[nid][2]
                        pt = compute_processing_time(sc, ns)
                        node_loads[nid] += pt * throughput_req / 1000.0 * load_per
                
                if check_reliability(deployment):
                    break
                break

# PHASE 3: Local optimization (try co-location)
print("Optimizing for co-location to reduce network latency...")

for attempt in range(3):
    improved = False
    
    for i in range(len(stage_ids) - 1):
        stage_id = stage_ids[i]
        next_stage_id = stage_ids[i + 1]
        
        current_nodes = deployment[stage_id]
        next_nodes = deployment[next_stage_id]
        
        # Check if they share any nodes
        shared = set(current_nodes) & set(next_nodes)
        if shared:
            continue
        
        # Try moving next_stage to a node of current_stage
        for node_id in current_nodes:
            if node_id in next_nodes:
                continue
            
            # Test moving next_stage to this node
            test_deployment = {k: list(v) for k, v in deployment.items()}
            
            # Try replacing first node of next_stage with this node
            if len(test_deployment[next_stage_id]) == 1:
                test_deployment[next_stage_id] = [node_id]
            else:
                # Keep redundancy, just swap first node
                test_deployment[next_stage_id][0] = node_id
            
            if check_capacity(test_deployment) and check_reliability(test_deployment):
                new_latency = compute_latency(test_deployment)
                old_latency = compute_latency(deployment)
                
                if new_latency < old_latency - 0.1:
                    print(f"  Co-locating {next_stage_id} with {stage_id} on {node_id}: {old_latency:.2f} -> {new_latency:.2f} ms")
                    deployment = test_deployment
                    improved = True
                    break
        
        if improved:
            break
    
    if not improved:
        break

# Final validation
if not check_capacity(deployment):
    print("ERROR: Final deployment violates capacity!")
    
if not check_reliability(deployment):
    print("ERROR: Final deployment does not meet reliability!")

final_latency = compute_latency(deployment)
print(f"Final latency: {final_latency:.2f} ms")

# PHASE 4: Write output
with open("/workdir/deployment.txt", "w") as f:
    f.write(f"{final_latency:.6f}\n")
    for stage_id in stage_ids:
        node_list = deployment[stage_id]
        f.write(f"{stage_id} {' '.join(node_list)}\n")

print("Deployment written successfully!")

# Debug info
for stage_id in stage_ids:
    stage_rel = compute_stage_reliability(deployment[stage_id])
    print(f"  {stage_id}: nodes={deployment[stage_id]}, reliability={stage_rel:.6f}")

pipeline_rel = 1.0
for stage_id in stage_ids:
    pipeline_rel *= compute_stage_reliability(deployment[stage_id])
print(f"Pipeline reliability: {pipeline_rel:.6f}")
PYCODE