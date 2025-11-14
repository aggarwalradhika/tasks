#!/bin/bash
set -e

cd /workdir/data

python3 - <<'PYCODE'
import collections
import math
from typing import List, Tuple, Dict, Set, Optional
import heapq

# Parse input
with open("warehouse.txt") as f:
    rows, cols, r, n, pick_time, battery_capacity = map(int, f.readline().split())
    
    robot_starts = {}
    for _ in range(r):
        parts = f.readline().split()
        robot_id, row, col = parts[0], int(parts[1]), int(parts[2])
        robot_starts[robot_id] = (row, col)
    
    orders = {}
    for _ in range(n):
        parts = f.readline().split()
        order_id, num_items = parts[0], int(parts[1])
        items = []
        for _ in range(num_items):
            item_parts = f.readline().split()
            items.append((int(item_parts[0]), int(item_parts[1])))
        orders[order_id] = items
    
    f.readline()  # blank line
    
    obstacles = set()
    for line in f:
        line = line.strip()
        if not line:
            break
        parts = line.split()
        obstacles.add((int(parts[0]), int(parts[1])))
    
    charging_stations = set()
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        charging_stations.add((int(parts[0]), int(parts[1])))

print(f"Parsed: {rows}×{cols} grid, {r} robots, {n} orders")

def manhattan_distance(pos1, pos2):
    """Chebyshev distance for 8-directional movement."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

def a_star_path_with_collision_avoidance(start, goal, obstacles, rows, cols, reserved_positions, start_time=0):
    """
    A* pathfinding WITH collision avoidance.
    reserved_positions: dict of time -> set of (robot_id, position) tuples
    Returns: list of positions (path) or None if no path found
    """
    if start == goal:
        return [start]
    
    open_set = []
    heapq.heappush(open_set, (0, start_time, start))
    came_from = {}
    g_score = {(start, start_time): 0}
    
    max_time = start_time + 1000  # Increased search depth
    
    while open_set:
        _, time, current = heapq.heappop(open_set)
        
        if time > max_time:
            break
        
        if current == goal:
            # Reconstruct path
            path = []
            current_time = time
            current_pos = current
            while (current_pos, current_time) in came_from:
                path.append(current_pos)
                current_pos, current_time = came_from[(current_pos, current_time)]
            path.reverse()
            return path
        
        r, c = current
        next_time = time + 1
        
        # Try 8 directions + wait in place
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in obstacles:
                    continue
                
                neighbor = (nr, nc)
                
                # Check vertex collision (position occupied at next_time)
                occupied = False
                for robot_id, pos in reserved_positions.get(next_time, set()):
                    if pos == neighbor:
                        occupied = True
                        break
                
                if occupied:
                    continue
                
                # Check edge collision (swapping positions)
                if dr != 0 or dc != 0:  # If moving (not waiting)
                    # Check if another robot is at neighbor at current time and at current at next_time
                    for robot_id_curr, pos_curr in reserved_positions.get(time, set()):
                        if pos_curr == neighbor:
                            # Someone is at neighbor now, check if they'll be at our current position next
                            for robot_id_next, pos_next in reserved_positions.get(next_time, set()):
                                if robot_id_next == robot_id_curr and pos_next == current:
                                    occupied = True
                                    break
                        if occupied:
                            break
                
                if occupied:
                    continue
                
                tentative_g = g_score.get((current, time), float('inf')) + 1
                
                if (neighbor, next_time) not in g_score or tentative_g < g_score[(neighbor, next_time)]:
                    came_from[(neighbor, next_time)] = (current, time)
                    g_score[(neighbor, next_time)] = tentative_g
                    f_score = tentative_g + manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, next_time, neighbor))
    
    return None

# Greedy order assignment
robot_assignments = {rid: [] for rid in robot_starts}
robot_positions = dict(robot_starts)
robot_times = {rid: 0 for rid in robot_starts}

for order_id in sorted(orders.keys()):
    items = orders[order_id]
    if not items:
        continue
    
    best_robot = None
    best_dist = float('inf')
    
    for robot_id in robot_starts:
        dist = manhattan_distance(robot_positions[robot_id], items[0])
        adjusted = robot_times[robot_id] + dist
        
        if adjusted < best_dist:
            best_dist = adjusted
            best_robot = robot_id
    
    if best_robot:
        robot_assignments[best_robot].append(order_id)
        total_dist = sum(manhattan_distance(items[i], items[i+1]) for i in range(len(items)-1))
        total_dist += manhattan_distance(robot_positions[best_robot], items[0])
        robot_times[best_robot] += total_dist + len(items) * pick_time
        robot_positions[best_robot] = items[-1]

print(f"Assigned orders: {[(rid, len(ol)) for rid, ol in robot_assignments.items()]}")

# PRIORITIZED PLANNING with collision avoidance
robot_priority = sorted(robot_starts.keys(), 
                       key=lambda rid: len(robot_assignments[rid]), 
                       reverse=True)

reserved_positions = collections.defaultdict(set)
all_plans = {}
robot_finish_times = {}

for robot_id in robot_priority:
    print(f"Planning for {robot_id}...")
    plan = []
    current_pos = robot_starts[robot_id]
    current_battery = battery_capacity
    current_time = 0
    
    # Reserve starting position
    reserved_positions[current_time].add((robot_id, current_pos))
    
    assigned_orders = robot_assignments[robot_id]
    
    for order_id in assigned_orders:
        items = orders[order_id]
        
        for item_pos in items:
            # Check if need charge before going to item
            dist = manhattan_distance(current_pos, item_pos)
            
            if dist + 20 > current_battery:  # Safety margin
                # Go to nearest station
                nearest_station = min(charging_stations, 
                                    key=lambda s: manhattan_distance(current_pos, s))
                
                # Try multiple times with increasing wait if path blocked
                path = None
                wait_attempts = 0
                max_wait_attempts = 3
                
                while path is None and wait_attempts < max_wait_attempts:
                    if wait_attempts > 0:
                        print(f"  {robot_id}: Path blocked, waiting {wait_attempts * 10} time units...")
                        for _ in range(wait_attempts * 10):
                            current_time += 1
                            reserved_positions[current_time].add((robot_id, current_pos))
                    
                    path = a_star_path_with_collision_avoidance(
                        current_pos, nearest_station, obstacles, rows, cols, 
                        reserved_positions, current_time
                    )
                    
                    wait_attempts += 1
                
                if path is None:
                    print(f"  ERROR: {robot_id} cannot find collision-free path to charging station from {current_pos} to {nearest_station}")
                    # This is a critical failure - we cannot proceed safely
                    raise Exception(f"Cannot find safe path for {robot_id}")
                
                # Execute move to charging station
                for pos in path:
                    current_time += 1
                    plan.append(f"MOVE {pos[0]} {pos[1]}")
                    reserved_positions[current_time].add((robot_id, pos))
                    current_battery -= 1
                current_pos = path[-1]
                
                # Charge
                plan.append("CHARGE")
                battery_used = battery_capacity - current_battery
                charge_time = int(math.ceil(math.sqrt(battery_used))) if battery_used > 0 else 1
                for t in range(charge_time):
                    current_time += 1
                    reserved_positions[current_time].add((robot_id, current_pos))
                current_battery = battery_capacity
            
            # Path to item WITH collision avoidance
            path = None
            wait_attempts = 0
            max_wait_attempts = 3
            
            while path is None and wait_attempts < max_wait_attempts:
                if wait_attempts > 0:
                    print(f"  {robot_id}: Path to item blocked, waiting {wait_attempts * 10} time units...")
                    for _ in range(wait_attempts * 10):
                        current_time += 1
                        reserved_positions[current_time].add((robot_id, current_pos))
                
                path = a_star_path_with_collision_avoidance(
                    current_pos, item_pos, obstacles, rows, cols,
                    reserved_positions, current_time
                )
                
                wait_attempts += 1
            
            if path is None:
                print(f"  ERROR: {robot_id} cannot find collision-free path to item at {item_pos} from {current_pos}")
                raise Exception(f"Cannot find safe path for {robot_id}")
            
            # Execute move to item
            for pos in path:
                current_time += 1
                plan.append(f"MOVE {pos[0]} {pos[1]}")
                reserved_positions[current_time].add((robot_id, pos))
                current_battery -= 1
            current_pos = path[-1]
            
            # Verify we're at the item location
            if current_pos != item_pos:
                print(f"  ERROR: {robot_id} not at item location {item_pos}, currently at {current_pos}")
                raise Exception(f"Robot {robot_id} positioning error")
            
            # Pick at item location
            plan.append(f"PICK {order_id}")
            for t in range(pick_time):
                current_time += 1
                reserved_positions[current_time].add((robot_id, current_pos))
    
    # Return to nearest station
    nearest_station = min(charging_stations, 
                         key=lambda s: manhattan_distance(current_pos, s))
    
    path = None
    wait_attempts = 0
    max_wait_attempts = 3
    
    while path is None and wait_attempts < max_wait_attempts:
        if wait_attempts > 0:
            print(f"  {robot_id}: Return path blocked, waiting {wait_attempts * 10} time units...")
            for _ in range(wait_attempts * 10):
                current_time += 1
                reserved_positions[current_time].add((robot_id, current_pos))
        
        path = a_star_path_with_collision_avoidance(
            current_pos, nearest_station, obstacles, rows, cols,
            reserved_positions, current_time
        )
        
        wait_attempts += 1
    
    if path is None:
        print(f"  ERROR: {robot_id} cannot return to charging station")
        raise Exception(f"Cannot find safe return path for {robot_id}")
    
    for pos in path:
        current_time += 1
        plan.append(f"MOVE {pos[0]} {pos[1]}")
        reserved_positions[current_time].add((robot_id, pos))
    
    plan.append("RETURN")
    all_plans[robot_id] = plan
    robot_finish_times[robot_id] = current_time
    print(f"  {robot_id} completes at time {current_time}")

makespan = max(robot_finish_times.values()) if robot_finish_times else 0

print(f"Computed makespan: {makespan}")

# Write schedule
with open("/workdir/schedule.txt", "w") as f:
    f.write(f"{makespan}\n")
    for robot_id in sorted(all_plans.keys()):
        f.write(f"{robot_id}\n")
        for cmd in all_plans[robot_id]:
            f.write(f"{cmd}\n")

print("Schedule generated successfully with full collision avoidance")
PYCODE