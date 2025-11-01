import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
import collections

from apex_arena._types import GradingResult


def parse_warehouse_file(filepath: Path) -> Tuple[int, int, int, int, int, int, 
                                                    Dict[str, Tuple[int, int]],
                                                    Dict[str, List[Tuple[int, int]]],
                                                    Set[Tuple[int, int]],
                                                    Set[Tuple[int, int]]]:
    """
    Parse the warehouse.txt input file.
    
    File format:
        rows cols num_robots num_orders pick_time battery_capacity
        <robot_id> <start_row> <start_col>
        ...
        <order_id> <num_items>
        <item_row> <item_col>
        ...
        [blank line]
        <obstacle_row> <obstacle_col>
        ...
        [blank line]
        <charging_row> <charging_col>
        ...
    
    Args:
        filepath: Path to the warehouse.txt file
    
    Returns:
        A tuple containing:
        - rows: Number of rows in the grid
        - cols: Number of columns in the grid
        - num_robots: Number of robots
        - num_orders: Number of orders
        - pick_time: Time units required to pick an item
        - battery_capacity: Maximum battery capacity
        - robot_starts: Dict mapping robot_id to (row, col) starting position
        - orders: Dict mapping order_id to list of (row, col) item locations
        - obstacles: Set of (row, col) obstacle positions
        - charging_stations: Set of (row, col) charging station positions
    
    Raises:
        ValueError: If file format is invalid
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("Empty file")
        parts = first_line.split()
        if len(parts) != 6:
            raise ValueError(f"First line must be 'rows cols r n p battery_capacity', got: {first_line}")
        rows, cols, r, n, p, battery = map(int, parts)
        
        # Parse robot starts
        robot_starts = {}
        for _ in range(r):
            line = f.readline().strip()
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid robot line: {line}")
            robot_id, row, col = parts[0], int(parts[1]), int(parts[2])
            robot_starts[robot_id] = (row, col)
        
        # Parse orders
        orders = {}
        for _ in range(n):
            line = f.readline().strip()
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid order header: {line}")
            order_id, num_items = parts[0], int(parts[1])
            items = []
            for _ in range(num_items):
                item_line = f.readline().strip()
                item_parts = item_line.split()
                if len(item_parts) != 2:
                    raise ValueError(f"Invalid item location: {item_line}")
                items.append((int(item_parts[0]), int(item_parts[1])))
            orders[order_id] = items
        
        # Skip blank line
        f.readline()
        
        # Parse obstacles
        obstacles = set()
        for line in f:
            line = line.strip()
            if not line:
                break
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid obstacle line: {line}")
            obstacles.add((int(parts[0]), int(parts[1])))
        
        # Parse charging stations
        charging_stations = set()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid charging station line: {line}")
            charging_stations.add((int(parts[0]), int(parts[1])))
        
        return rows, cols, r, n, p, battery, robot_starts, orders, obstacles, charging_stations


def is_adjacent(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
    """
    Check if two positions are adjacent (including diagonals).
    
    In 8-directional movement, positions are adjacent if they differ
    by at most 1 in both row and column, and are not the same position.
    
    Args:
        pos1: First position (row, col)
        pos2: Second position (row, col)
    
    Returns:
        True if positions are adjacent, False otherwise
    
    Examples:
        >>> is_adjacent((0, 0), (0, 1))  # horizontal
        True
        >>> is_adjacent((0, 0), (1, 1))  # diagonal
        True
        >>> is_adjacent((0, 0), (2, 0))  # too far
        False
        >>> is_adjacent((0, 0), (0, 0))  # same position
        False
    """
    r1, c1 = pos1
    r2, c2 = pos2
    return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 and (r1, c1) != (r2, c2)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Chebyshev distance for 8-directional movement.
    
    This is the correct distance metric for grids allowing diagonal movement.
    It equals the minimum number of moves needed to reach pos2 from pos1.
    
    Args:
        pos1: First position (row, col)
        pos2: Second position (row, col)
    
    Returns:
        Minimum number of moves between positions
    
    Examples:
        >>> manhattan_distance((0, 0), (3, 4))
        4  # max(3, 4) = 4 moves (e.g., 3 diagonal + 1 vertical)
        >>> manhattan_distance((0, 0), (2, 2))
        2  # 2 diagonal moves
    """
    r1, c1 = pos1
    r2, c2 = pos2
    return max(abs(r1 - r2), abs(c1 - c2))


def simulate_plan(plan_path: Path, 
                 rows: int, cols: int,
                 robot_starts: Dict[str, Tuple[int, int]],
                 orders: Dict[str, List[Tuple[int, int]]],
                 obstacles: Set[Tuple[int, int]],
                 charging_stations: Set[Tuple[int, int]],
                 pick_time: int,
                 battery_capacity: int) -> Tuple[int, List[str]]:
    """
    Simulate the execution of the robot plan and validate all constraints.
    
    This function performs step-by-step simulation of the plan, tracking:
    - Robot positions over time
    - Battery levels
    - Order completion status
    - Collision detection (no two robots in same cell at same time)
    - Movement validity (adjacency, bounds, obstacles)
    - Battery constraints (never negative)
    - Charging station access
    
    Args:
        plan_path: Path to the plan.txt file
        rows: Number of rows in grid
        cols: Number of columns in grid
        robot_starts: Robot starting positions
        orders: Order definitions (item locations)
        obstacles: Obstacle positions
        charging_stations: Charging station positions
        pick_time: Time units to pick an item
        battery_capacity: Maximum battery capacity
    
    Returns:
        A tuple (makespan, errors) where:
        - makespan: The maximum completion time across all robots, or -1 if errors
        - errors: List of error messages (empty if valid)
    
    Validation checks performed:
    1. Plan format is correct (robot headers, valid commands)
    2. All robots have plans
    3. Movements are valid (adjacent cells, within bounds, no obstacles)
    4. Battery never goes negative
    5. Charging only at charging stations
    6. No collisions (two robots in same cell at same time)
    7. All orders completed exactly once
    8. All items in an order are picked before order completion
    9. Return command only at charging stations
    """
    errors = []
    
    # Parse plan
    robot_plans = {}
    current_robot = None
    
    with open(plan_path) as f:
        # Skip first line (makespan)
        f.readline()
        
        for line_num, line in enumerate(f, 2):  # Start at line 2 since we skipped line 1
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a robot header
            if not line.startswith('MOVE') and not line.startswith('PICK') and \
               not line.startswith('CHARGE') and not line.startswith('RETURN'):
                current_robot = line
                if current_robot not in robot_starts:
                    errors.append(f"Line {line_num}: Unknown robot '{current_robot}'")
                    return -1, errors
                if current_robot in robot_plans:
                    errors.append(f"Line {line_num}: Duplicate plan for robot '{current_robot}'")
                    return -1, errors
                robot_plans[current_robot] = []
                continue
            
            if current_robot is None:
                errors.append(f"Line {line_num}: Command before robot header")
                return -1, errors
            
            robot_plans[current_robot].append((line_num, line))
    
    # Check all robots have plans
    if set(robot_plans.keys()) != set(robot_starts.keys()):
        missing = set(robot_starts.keys()) - set(robot_plans.keys())
        errors.append(f"Missing plans for robots: {sorted(missing)}")
        return -1, errors
    
    # Simulate execution
    robot_state = {}
    for robot_id, start_pos in robot_starts.items():
        robot_state[robot_id] = {
            'position': start_pos,
            'battery': battery_capacity,
            'time': 0,
            'plan_index': 0,
            'completed_orders': set(),
            'picked_items': collections.defaultdict(set),  # order_id -> set of picked locations
            'finished': False
        }
    
    # Track order assignments
    order_assignments = {}
    
    # Track positions over time for collision detection
    position_timeline = collections.defaultdict(set)  # time -> set of (robot_id, position)
    
    # Execute plans
    all_finished = False
    max_time = 0
    
    while not all_finished:
        all_finished = True
        
        for robot_id in sorted(robot_plans.keys()):
            state = robot_state[robot_id]
            
            if state['finished']:
                continue
            
            all_finished = False
            plan = robot_plans[robot_id]
            
            if state['plan_index'] >= len(plan):
                errors.append(f"Robot {robot_id}: Plan exhausted without RETURN")
                return -1, errors
            
            line_num, command = plan[state['plan_index']]
            parts = command.split()
            cmd_type = parts[0]
            
            if cmd_type == 'MOVE':
                if len(parts) != 3:
                    errors.append(f"Line {line_num}: Invalid MOVE command format")
                    return -1, errors
                
                try:
                    target_row, target_col = int(parts[1]), int(parts[2])
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid coordinates in MOVE")
                    return -1, errors
                
                # Validate bounds
                if not (0 <= target_row < rows and 0 <= target_col < cols):
                    errors.append(f"Line {line_num}: MOVE out of bounds: ({target_row}, {target_col})")
                    return -1, errors
                
                # Check obstacle
                if (target_row, target_col) in obstacles:
                    errors.append(f"Line {line_num}: MOVE to obstacle at ({target_row}, {target_col})")
                    return -1, errors
                
                # Check adjacency
                if not is_adjacent(state['position'], (target_row, target_col)):
                    errors.append(f"Line {line_num}: MOVE not adjacent from {state['position']} to ({target_row}, {target_col})")
                    return -1, errors
                
                # Check battery
                if state['battery'] < 1:
                    errors.append(f"Line {line_num}: Insufficient battery for MOVE (battery={state['battery']})")
                    return -1, errors
                
                # Store previous position for edge collision check
                prev_pos = state['position']
                
                # Execute move
                state['battery'] -= 1
                state['position'] = (target_row, target_col)
                state['time'] += 1
                
                # Check vertex collision (two robots at same position at same time)
                for other_robot, other_pos in position_timeline[state['time']]:
                    if other_pos == state['position'] and other_robot != robot_id:
                        errors.append(f"Line {line_num}: Vertex collision at {state['position']} at time {state['time']} between {robot_id} and {other_robot}")
                        return -1, errors
                
                # Check edge collision (two robots swapping positions)
                # Robot A: pos1 -> pos2 at time t
                # Robot B: pos2 -> pos1 at time t (COLLISION!)
                if state['time'] - 1 in position_timeline:
                    for other_robot, other_pos in position_timeline[state['time']]:
                        if other_robot != robot_id:
                            # Find other robot's previous position
                            for check_robot, check_pos in position_timeline[state['time'] - 1]:
                                if check_robot == other_robot:
                                    other_prev_pos = check_pos
                                    # Check if robots swapped positions
                                    if other_prev_pos == state['position'] and prev_pos == other_pos:
                                        errors.append(f"Line {line_num}: Edge collision (position swap) between {robot_id} and {other_robot} at time {state['time']}")
                                        return -1, errors
                                    break
                
                position_timeline[state['time']].add((robot_id, state['position']))
                state['plan_index'] += 1
            
            elif cmd_type == 'PICK':
                if len(parts) != 2:
                    errors.append(f"Line {line_num}: Invalid PICK command format")
                    return -1, errors
                
                order_id = parts[1]
                
                if order_id not in orders:
                    errors.append(f"Line {line_num}: Unknown order '{order_id}'")
                    return -1, errors
                
                # Check if order already assigned
                if order_id in order_assignments and order_assignments[order_id] != robot_id:
                    errors.append(f"Line {line_num}: Order '{order_id}' already assigned to {order_assignments[order_id]}")
                    return -1, errors
                
                # Record assignment
                if order_id not in order_assignments:
                    order_assignments[order_id] = robot_id
                
                # Check if at correct location for one of the items
                if state['position'] not in orders[order_id]:
                    errors.append(f"Line {line_num}: PICK '{order_id}' at {state['position']}, but no item there")
                    return -1, errors
                
                # Mark item as picked
                state['picked_items'][order_id].add(state['position'])
                
                # Check if all items picked
                if state['picked_items'][order_id] == set(orders[order_id]):
                    state['completed_orders'].add(order_id)
                
                # Execute pick
                state['time'] += pick_time
                state['plan_index'] += 1
            
            elif cmd_type == 'CHARGE':
                if len(parts) != 1:
                    errors.append(f"Line {line_num}: Invalid CHARGE command format")
                    return -1, errors
                
                # Check at charging station
                if state['position'] not in charging_stations:
                    errors.append(f"Line {line_num}: CHARGE at {state['position']}, but no charging station there")
                    return -1, errors
                
                # Calculate charge time
                battery_used = battery_capacity - state['battery']
                charge_time = int(math.ceil(math.sqrt(battery_used)))
                
                state['battery'] = battery_capacity
                state['time'] += charge_time
                state['plan_index'] += 1
            
            elif cmd_type == 'RETURN':
                if len(parts) != 1:
                    errors.append(f"Line {line_num}: Invalid RETURN command format")
                    return -1, errors
                
                # Check at charging station
                if state['position'] not in charging_stations:
                    errors.append(f"Line {line_num}: RETURN at {state['position']}, but no charging station there")
                    return -1, errors
                
                # Mark as finished
                state['finished'] = True
                max_time = max(max_time, state['time'])
                state['plan_index'] += 1
            
            else:
                errors.append(f"Line {line_num}: Unknown command '{cmd_type}'")
                return -1, errors
    
    # Validate all orders completed
    all_orders = set(orders.keys())
    completed_orders = set()
    for robot_id, state in robot_state.items():
        completed_orders.update(state['completed_orders'])
    
    if completed_orders != all_orders:
        missing = all_orders - completed_orders
        errors.append(f"Incomplete orders: {sorted(missing)}")
        return -1, errors
    
    # Check each order completed by exactly one robot
    if set(order_assignments.keys()) != all_orders:
        missing = all_orders - set(order_assignments.keys())
        errors.append(f"Orders not assigned: {sorted(missing)}")
        return -1, errors
    
    return max_time, errors


def greedy_baseline(rows: int, cols: int,
                   robot_starts: Dict[str, Tuple[int, int]],
                   orders: Dict[str, List[Tuple[int, int]]],
                   obstacles: Set[Tuple[int, int]],
                   charging_stations: Set[Tuple[int, int]],
                   pick_time: int,
                   battery_capacity: int) -> int:
    """
    Sophisticated baseline using:
    1. Smart order assignment considering distance and workload balance
    2. Proper battery management with charging penalties
    3. Accurate time estimation including all operations
    4. Load balancing to minimize makespan
    
    This baseline should be reasonably good and hard to beat significantly.
    """
    robot_times = {rid: 0 for rid in robot_starts}
    robot_positions = dict(robot_starts)
    robot_batteries = {rid: battery_capacity for rid in robot_starts}
    robot_assignments = {rid: [] for rid in robot_starts}
    
    # Calculate order characteristics for better assignment
    order_info = {}
    for order_id, items in orders.items():
        if not items:
            continue
        
        # Centroid of order items
        centroid = (sum(r for r, c in items) / len(items), 
                   sum(c for r, c in items) / len(items))
        
        # Internal travel distance
        internal_dist = sum(manhattan_distance(items[i], items[i+1]) 
                           for i in range(len(items) - 1))
        
        # Total items
        num_items = len(items)
        
        order_info[order_id] = {
            'centroid': centroid,
            'internal_dist': internal_dist,
            'num_items': num_items,
            'items': items
        }
    
    # Sort orders by a composite metric (internal distance, then number of items)
    # This helps group nearby items together
    sorted_orders = sorted(order_info.keys(), 
                          key=lambda oid: (order_info[oid]['internal_dist'], 
                                          order_info[oid]['num_items']))
    
    # Assign orders using a more sophisticated approach
    for order_id in sorted_orders:
        info = order_info[order_id]
        items = info['items']
        
        best_robot = None
        best_score = float('inf')
        
        for robot_id in robot_starts:
            current_pos = robot_positions[robot_id]
            current_battery = robot_batteries[robot_id]
            current_time = robot_times[robot_id]
            
            # Calculate actual cost considering current state
            # Distance to first item
            dist_to_first = manhattan_distance(current_pos, items[0])
            
            # Distance between items
            internal_dist = info['internal_dist']
            
            # Total travel distance
            total_travel = dist_to_first + internal_dist
            
            # Pick time
            pick_cost = info['num_items'] * pick_time
            
            # Check if charging is needed
            charge_penalty = 0
            if total_travel > current_battery:
                # Need to charge - add cost to go to station, charge, and return
                nearest_station = min(charging_stations, 
                                    key=lambda s: manhattan_distance(current_pos, s))
                station_dist = manhattan_distance(current_pos, nearest_station)
                
                # Charging time based on battery used
                battery_used = battery_capacity - current_battery
                charge_time = int(math.ceil(math.sqrt(battery_used))) if battery_used > 0 else 0
                
                charge_penalty = station_dist + charge_time
            
            # Return cost to nearest station
            final_pos = items[-1] if items else current_pos
            return_dist = min(manhattan_distance(final_pos, station) 
                            for station in charging_stations)
            
            # Total estimated time for this order
            order_time = total_travel + pick_cost + charge_penalty
            
            # Score combines completion time with load balancing
            # Penalize robots that are already busy to balance workload
            completion_time = current_time + order_time
            workload_penalty = len(robot_assignments[robot_id]) * 20
            
            score = completion_time + workload_penalty
            
            if score < best_score:
                best_score = score
                best_robot = robot_id
        
        # Assign order to best robot
        if best_robot:
            info_best = order_info[order_id]
            items_best = info_best['items']
            
            current_pos = robot_positions[best_robot]
            
            # Update robot state
            dist_to_first = manhattan_distance(current_pos, items_best[0])
            internal_dist = info_best['internal_dist']
            total_travel = dist_to_first + internal_dist
            
            # Handle charging
            if total_travel > robot_batteries[best_robot]:
                nearest_station = min(charging_stations, 
                                    key=lambda s: manhattan_distance(current_pos, s))
                station_dist = manhattan_distance(current_pos, nearest_station)
                battery_used = battery_capacity - robot_batteries[best_robot]
                charge_time = int(math.ceil(math.sqrt(battery_used))) if battery_used > 0 else 0
                
                robot_times[best_robot] += station_dist + charge_time
                robot_batteries[best_robot] = battery_capacity
                current_pos = nearest_station
            
            # Execute order
            pick_cost = info_best['num_items'] * pick_time
            robot_times[best_robot] += total_travel + pick_cost
            robot_batteries[best_robot] -= total_travel
            robot_positions[best_robot] = items_best[-1] if items_best else current_pos
            robot_assignments[best_robot].append(order_id)
    
    # Add return time for all robots
    for robot_id in robot_starts:
        current_pos = robot_positions[robot_id]
        return_dist = min(manhattan_distance(current_pos, station) 
                         for station in charging_stations)
        robot_times[robot_id] += return_dist
    
    # Makespan is the maximum completion time
    makespan = max(robot_times.values()) if robot_times else 0
    
    return makespan


def grade(transcript: str) -> GradingResult:
    """
    Grade the warehouse robot planning submission.
    
    GRADING CRITERIA (all must pass):
    1. Files exist: schedule.txt, data/warehouse.txt
    2. Schedule format is valid (first line is makespan integer)
    3. All robots have valid plans ending with RETURN
    4. All movements are valid (adjacent, within bounds, no obstacles)
    5. No battery violations
    6. No collision violations
    7. All orders completed exactly once
    8. Claimed makespan (first line) matches computed makespan
    9. Solution is near-optimal: makespan ≤ 1.25 × baseline
    """
    feedback_messages: List[str] = []
    subscores = {"correct_answer": 0.0}
    weights = {"correct_answer": 1.0}

    try:
        schedule_path = Path("/workdir/schedule.txt")
        data_path = Path("/workdir/data/warehouse.txt")

        # ===== CHECK 1: Files exist =====
        for p, name in [(data_path, "Input"), (schedule_path, "Schedule")]:
            if not p.exists():
                feedback_messages.append(f"FAIL: {name} file {p} does not exist")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        # Parse input
        rows, cols, r, n, pick_time, battery, robot_starts, orders, obstacles, charging_stations = \
            parse_warehouse_file(data_path)
        
        feedback_messages.append(f"Problem: {rows}×{cols} grid, {r} robots, {n} orders, pick_time={pick_time}, battery={battery}")

        # ===== CHECK 2: Parse claimed makespan from first line =====
        try:
            with open(schedule_path, "r") as f:
                first_line = f.readline().strip()
                claimed_makespan = int(first_line)
            feedback_messages.append(f"Claimed makespan (from schedule.txt line 1): {claimed_makespan}")
        except ValueError:
            feedback_messages.append(f"FAIL: First line of schedule.txt must be an integer (makespan), got: '{first_line}'")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        except Exception as e:
            feedback_messages.append(f"FAIL: Error reading makespan from schedule.txt - {e}")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))

        # ===== CHECK 3: Simulate plan and validate =====
        actual_makespan, errors = simulate_plan(
            schedule_path, rows, cols, robot_starts, orders, obstacles, 
            charging_stations, pick_time, battery
        )
        
        if errors:
            feedback_messages.append(f"FAIL: Plan validation errors:")
            feedback_messages.extend(errors)
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        
        feedback_messages.append(f"Plan valid: all constraints satisfied")
        feedback_messages.append(f"Actual computed makespan: {actual_makespan}")

        # ===== CHECK 4: Claimed vs actual match =====
        if claimed_makespan != actual_makespan:
            feedback_messages.append(f"FAIL: Claimed makespan ({claimed_makespan}) != actual makespan ({actual_makespan})")
            return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                 feedback="; ".join(feedback_messages))
        
        feedback_messages.append("Claimed makespan matches computed makespan")

        # ===== CHECK 5: Near-optimality check =====
        baseline_makespan = greedy_baseline(
            rows, cols, robot_starts, orders, obstacles, 
            charging_stations, pick_time, battery
        )
        feedback_messages.append(f"Baseline makespan: {baseline_makespan}")

        # Tolerance: 1.25x
        try:
            tol_env = os.getenv("GRADER_NEAR_OPT_TOL")
            tolerance = float(tol_env) if tol_env is not None else 1.25
            if tolerance <= 1.0:
                tolerance = 1.25
        except Exception:
            tolerance = 1.25

        feedback_messages.append(f"Near-optimality tolerance: {tolerance:.2f}x baseline (solution must be ≤ {baseline_makespan * tolerance:.1f})")

        # Check near-optimality
        if baseline_makespan == 0:
            if actual_makespan != 0:
                feedback_messages.append("FAIL: Baseline is 0 but your solution has makespan > 0")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))
        else:
            threshold = baseline_makespan * tolerance
            if actual_makespan > threshold:
                feedback_messages.append(
                    f"FAIL: Not near-optimal: {actual_makespan} > {tolerance:.2f} × {baseline_makespan} = {threshold:.1f}"
                )
                feedback_messages.append(f"Your solution is {(actual_makespan/baseline_makespan):.2f}x baseline (needs to be ≤ {tolerance:.2f}x)")
                return GradingResult(score=0.0, subscores=subscores, weights=weights,
                                     feedback="; ".join(feedback_messages))

        feedback_messages.append(f"Solution is near-optimal ({(actual_makespan/baseline_makespan):.3f}x baseline ≤ {tolerance:.2f}x)")

        # ===== SUCCESS! =====
        subscores["correct_answer"] = 1.0
        feedback_messages.append("ALL CHECKS PASSED - Solution is valid and near-optimal!")
        return GradingResult(score=1.0, subscores=subscores, weights=weights,
                             feedback="; ".join(feedback_messages))

    except Exception as e:
        import traceback
        feedback_messages.append(f"UNEXPECTED ERROR: {e}")
        feedback_messages.append(traceback.format_exc())
        return GradingResult(score=0.0, subscores=subscores, weights=weights,
                             feedback="; ".join(feedback_messages))