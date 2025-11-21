#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PY'
#!/usr/bin/env python3
"""
Reference Solution for Distributed Transaction Coordinator with Consensus

Implements a distributed transaction system with:
- Two-Phase Commit (2PC) for atomic distributed transactions
- Paxos consensus for fault-tolerant coordinator election
- Multiple isolation levels (SERIALIZABLE, SNAPSHOT, READ_COMMITTED)
- Deadlock detection using wait-for graphs
- Write-ahead logging for durability and recovery
- Partition management with replication and quorum

This is an EXPERT-level implementation demonstrating core distributed systems concepts.
"""
import json, csv, hashlib
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Set
from copy import deepcopy
from enum import Enum

# Constants
PAXOS_QUORUM_RATIO = 0.5
MAX_2PC_TIMEOUT_MS = 5000
LOCK_TIMEOUT_MS = 10000
DEADLOCK_DETECTION_INTERVAL_MS = 1000
MAX_CONCURRENT_TXNS = 1000
CONSENSUS_ROUND_TIMEOUT_MS = 2000
MAX_PAXOS_ROUNDS = 10

STREAM = Path("/workdir/data/transaction_log.jsonl")
OUT = Path("/workdir/sol.csv")

class TxnStatus(Enum):
    IN_PROGRESS = "in_progress"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"
    BLOCKED = "blocked"

class LockType(Enum):
    SHARED = "S"
    EXCLUSIVE = "X"
    UPDATE = "U"

class IsolationLevel(Enum):
    SERIALIZABLE = "serializable"
    SNAPSHOT = "snapshot"
    READ_COMMITTED = "read_committed"

class Lock:
    """Represents a lock held by a transaction"""
    def __init__(self, txn_id, lock_type, partition_id, key):
        self.txn_id = txn_id
        self.lock_type = lock_type
        self.partition_id = partition_id
        self.key = key
        self.acquired_at = 0

class Transaction:
    """Complete transaction state"""
    def __init__(self, txn_id, coordinator_id, isolation_level, begin_ts):
        self.txn_id = txn_id
        self.coordinator_id = coordinator_id
        self.isolation_level = IsolationLevel(isolation_level)
        self.begin_ts = begin_ts
        self.status = TxnStatus.IN_PROGRESS
        self.operations = []
        self.participants = set()
        self.locks_held = []
        self.conflicts = set()
        self.commit_ts = None
        self.abort_reason = None
        self.prepare_votes = {}
        self.consensus_rounds = 0
        self.recovery_needed = False
        self.snapshot_version = begin_ts if isolation_level == "snapshot" else None
        self.waiting_for = None
        self.write_set = set()  # Keys written by this transaction

class Partition:
    """Partition with data and locks"""
    def __init__(self, partition_id, key_range_start, key_range_end, replica_nodes):
        self.partition_id = partition_id
        self.key_range_start = key_range_start
        self.key_range_end = key_range_end
        self.replica_nodes = replica_nodes
        # MVCC: key -> [(value, version_ts, txn_id), ...]
        self.data = defaultdict(list)
        # Locks: key -> [Lock, ...]
        self.locks = defaultdict(list)

class Node:
    """Node in the distributed system"""
    def __init__(self, node_id):
        self.node_id = node_id
        self.is_failed = False
        self.failure_type = None
        self.recovery_time = None

class PaxosInstance:
    """Single Paxos consensus instance"""
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.promised_round = 0
        self.accepted_round = None
        self.accepted_value = None
        self.chosen_value = None
        self.votes = {}  # voter_id -> (round, vote_type)

class DistributedSystem:
    """Main distributed transaction coordinator"""
    def __init__(self):
        self.transactions: Dict[str, Transaction] = {}
        self.partitions: Dict[str, Partition] = {}
        self.nodes: Dict[str, Node] = {}
        self.paxos_instances: Dict[str, PaxosInstance] = {}
        self.current_time = 0
        self.checkpoints = {}
        self.log = []  # Write-ahead log
        
    def add_partition(self, partition_id, key_range_start, key_range_end, replica_nodes):
        """Register a partition with replicas"""
        self.partitions[partition_id] = Partition(
            partition_id, key_range_start, key_range_end, replica_nodes
        )
        for node_id in replica_nodes:
            if node_id not in self.nodes:
                self.nodes[node_id] = Node(node_id)
    
    def begin_transaction(self, txn_id, coordinator_id, isolation_level, timestamp):
        """Start a new transaction"""
        txn = Transaction(txn_id, coordinator_id, isolation_level, timestamp)
        self.transactions[txn_id] = txn
        self.current_time = timestamp
        self.log.append(('BEGIN', txn_id, timestamp))
    
    def add_operation(self, txn_id, operation, partition_id, key, value, predicate, timestamp):
        """Add operation to transaction"""
        if txn_id not in self.transactions:
            return
        
        txn = self.transactions[txn_id]
        txn.operations.append({
            'operation': operation,
            'partition_id': partition_id,
            'key': key,
            'value': value,
            'predicate': predicate,
            'timestamp': timestamp
        })
        txn.participants.add(partition_id)
        
        if operation in ['write', 'delete']:
            txn.write_set.add((partition_id, key))
        
        self.current_time = timestamp
    
    def can_acquire_lock(self, partition_id, key, lock_type, txn_id):
        """Check lock compatibility"""
        if partition_id not in self.partitions:
            return True
        
        partition = self.partitions[partition_id]
        existing_locks = partition.locks[key]
        
        for lock in existing_locks:
            if lock.txn_id == txn_id:
                continue
            
            # Lock compatibility matrix
            if lock_type == LockType.EXCLUSIVE:
                return False  # X incompatible with all
            if lock.lock_type == LockType.EXCLUSIVE:
                return False  # All incompatible with X
            if lock_type == LockType.UPDATE and lock.lock_type == LockType.UPDATE:
                return False  # U incompatible with U
        
        return True
    
    def acquire_lock(self, txn_id, partition_id, key, lock_type, timestamp):
        """Acquire lock with deadlock detection"""
        if txn_id not in self.transactions:
            return False
        
        txn = self.transactions[txn_id]
        
        # Check lock timeout
        if timestamp - txn.begin_ts > LOCK_TIMEOUT_MS:
            txn.status = TxnStatus.ABORTED
            txn.abort_reason = "lock_timeout"
            self.release_locks(txn_id)
            return False
        
        # Check if can acquire immediately
        if self.can_acquire_lock(partition_id, key, lock_type, txn_id):
            lock = Lock(txn_id, lock_type, partition_id, key)
            lock.acquired_at = timestamp
            self.partitions[partition_id].locks[key].append(lock)
            txn.locks_held.append(lock)
            return True
        
        # Would block - check for deadlock
        blocking_txns = set()
        for lock in self.partitions[partition_id].locks[key]:
            if lock.txn_id != txn_id:
                blocking_txns.add(lock.txn_id)
        
        # Deadlock detection
        for blocker_id in blocking_txns:
            if blocker_id in self.transactions:
                # Check if we create a cycle
                if self.creates_deadlock_cycle(txn_id, blocker_id):
                    # Abort younger transaction
                    if txn.begin_ts > self.transactions[blocker_id].begin_ts:
                        txn.status = TxnStatus.ABORTED
                        txn.abort_reason = "deadlock_victim"
                        self.release_locks(txn_id)
                        return False
                    else:
                        # Abort the blocker instead
                        blocker = self.transactions[blocker_id]
                        blocker.status = TxnStatus.ABORTED
                        blocker.abort_reason = "deadlock_victim"
                        self.release_locks(blocker_id)
                        # Try again
                        return self.acquire_lock(txn_id, partition_id, key, lock_type, timestamp)
        
        # Record conflict
        txn.conflicts.update(blocking_txns)
        return False
    
    def creates_deadlock_cycle(self, from_txn, to_txn):
        """DFS cycle detection in wait-for graph"""
        visited = set()
        
        def dfs(current):
            if current == from_txn:
                return True
            if current in visited:
                return False
            visited.add(current)
            
            if current not in self.transactions:
                return False
            
            # Find what current is waiting for
            for partition in self.partitions.values():
                for key, locks in partition.locks.items():
                    blocked_by = set()
                    waiting = False
                    
                    for lock in locks:
                        if lock.txn_id == current:
                            # Current holds this lock, check who's blocked
                            for other_lock in locks:
                                if other_lock.txn_id != current:
                                    if dfs(other_lock.txn_id):
                                        return True
                        else:
                            blocked_by.add(lock.txn_id)
                            waiting = True
                    
                    if waiting:
                        for blocker in blocked_by:
                            if dfs(blocker):
                                return True
            
            return False
        
        return dfs(to_txn)
    
    def release_locks(self, txn_id):
        """Release all locks held by transaction"""
        if txn_id not in self.transactions:
            return
        
        txn = self.transactions[txn_id]
        for lock in txn.locks_held:
            if lock.partition_id in self.partitions:
                partition = self.partitions[lock.partition_id]
                if lock in partition.locks[lock.key]:
                    partition.locks[lock.key].remove(lock)
        txn.locks_held = []
    
    def execute_2pc(self, txn_id, timestamp):
        """Execute Two-Phase Commit protocol"""
        if txn_id not in self.transactions:
            return
        
        txn = self.transactions[txn_id]
        self.current_time = timestamp
        
        # Already aborted
        if txn.status == TxnStatus.ABORTED:
            self.release_locks(txn_id)
            self.log.append(('ABORT', txn_id, timestamp))
            return
        
        # Empty transaction - commit immediately
        if not txn.participants:
            txn.status = TxnStatus.COMMITTED
            txn.commit_ts = timestamp
            self.log.append(('COMMIT', txn_id, timestamp))
            return
        
        # Check coordinator availability
        if txn.coordinator_id in self.nodes and self.nodes[txn.coordinator_id].is_failed:
            # Run Paxos to elect new coordinator
            new_coordinator = self.run_paxos_election(txn_id, timestamp)
            if new_coordinator:
                txn.coordinator_id = new_coordinator
                txn.recovery_needed = True
            else:
                # Cannot reach quorum - abort
                txn.status = TxnStatus.ABORTED
                txn.abort_reason = "coordinator_failed_no_quorum"
                self.release_locks(txn_id)
                self.log.append(('ABORT', txn_id, timestamp))
                return
        
        # Phase 1: PREPARE
        prepare_success = self.run_prepare_phase(txn_id, timestamp)
        
        if not prepare_success:
            txn.status = TxnStatus.ABORTED
            if not txn.abort_reason:
                txn.abort_reason = "prepare_failed"
            self.release_locks(txn_id)
            self.log.append(('ABORT', txn_id, timestamp))
            return
        
        # Phase 2: COMMIT
        self.run_commit_phase(txn_id, timestamp)
    
    def run_prepare_phase(self, txn_id, timestamp):
        """2PC Phase 1: Send PREPARE, collect votes"""
        txn = self.transactions[txn_id]
        
        for partition_id in txn.participants:
            if partition_id not in self.partitions:
                txn.prepare_votes[partition_id] = "NO"
                txn.abort_reason = "partition_not_found"
                return False
            
            partition = self.partitions[partition_id]
            
            # Check replica availability (quorum)
            reachable_replicas = [
                n for n in partition.replica_nodes
                if n not in self.nodes or not self.nodes[n].is_failed
            ]
            
            quorum_size = len(partition.replica_nodes) // 2 + 1
            
            if len(reachable_replicas) < quorum_size:
                txn.prepare_votes[partition_id] = "NO"
                txn.abort_reason = "partition_unreachable"
                return False
            
            # Check for write conflicts (snapshot isolation)
            if txn.isolation_level == IsolationLevel.SNAPSHOT:
                if self.has_write_conflict(txn_id, partition_id):
                    txn.prepare_votes[partition_id] = "NO"
                    txn.abort_reason = "write_conflict"
                    return False
            
            # Acquire locks for operations
            success = True
            for op in txn.operations:
                if op['partition_id'] != partition_id:
                    continue
                
                # Determine lock type based on operation and isolation level
                if op['operation'] in ['write', 'delete']:
                    lock_type = LockType.EXCLUSIVE
                elif txn.isolation_level == IsolationLevel.SERIALIZABLE:
                    # Serializable needs read locks
                    lock_type = LockType.SHARED
                    # Check for predicate locks
                    if op.get('predicate'):
                        # Predicate lock - lock range (simplified: lock key prefix)
                        lock_type = LockType.SHARED
                else:
                    # Snapshot and read_committed don't need read locks
                    continue
                
                if not self.acquire_lock(txn_id, partition_id, op['key'], lock_type, timestamp):
                    success = False
                    break
            
            if success:
                txn.prepare_votes[partition_id] = "YES"
            else:
                txn.prepare_votes[partition_id] = "NO"
                if not txn.abort_reason:
                    txn.abort_reason = "lock_acquisition_failed"
                return False
        
        # Log PREPARE
        self.log.append(('PREPARE', txn_id, timestamp, list(txn.participants)))
        return True
    
    def run_commit_phase(self, txn_id, timestamp):
        """2PC Phase 2: Send COMMIT decision"""
        txn = self.transactions[txn_id]
        txn.status = TxnStatus.COMMITTED
        txn.commit_ts = timestamp
        
        # Log COMMIT decision (durable!)
        self.log.append(('COMMIT', txn_id, timestamp))
        
        # Apply writes to partitions (MVCC)
        for op in txn.operations:
            if op['operation'] == 'write' and op['partition_id'] in self.partitions:
                partition = self.partitions[op['partition_id']]
                key = op['key']
                # Add new version
                partition.data[key].append((op['value'], timestamp, txn_id))
                # Keep only recent versions (simplified GC)
                if len(partition.data[key]) > 10:
                    partition.data[key] = partition.data[key][-5:]
        
        # Release locks
        self.release_locks(txn_id)
    
    def has_write_conflict(self, txn_id, partition_id):
        """Check for write-write conflicts in snapshot isolation"""
        txn = self.transactions[txn_id]
        partition = self.partitions[partition_id]
        
        # Check if any of our writes conflict with committed writes after our snapshot
        for partition_id_check, key in txn.write_set:
            if partition_id_check != partition_id:
                continue
            
            if key in partition.data:
                # Check versions written after our snapshot
                for value, version_ts, writer_txn_id in partition.data[key]:
                    if version_ts > txn.snapshot_version and writer_txn_id != txn_id:
                        # Conflict with a committed transaction
                        if writer_txn_id in self.transactions:
                            writer = self.transactions[writer_txn_id]
                            if writer.status == TxnStatus.COMMITTED:
                                txn.conflicts.add(writer_txn_id)
                                return True
        
        return False
    
    def run_paxos_election(self, txn_id, timestamp):
        """Run Paxos to elect new coordinator"""
        instance_id = f"coordinator_election_{txn_id}"
        
        if instance_id not in self.paxos_instances:
            self.paxos_instances[instance_id] = PaxosInstance(instance_id)
        
        instance = self.paxos_instances[instance_id]
        
        # Find available nodes
        available_nodes = [
            node_id for node_id, node in self.nodes.items()
            if not node.is_failed
        ]
        
        if not available_nodes:
            return None
        
        quorum_size = len(self.nodes) // 2 + 1
        if len(available_nodes) < quorum_size:
            return None
        
        # Simplified Paxos: choose first available node
        instance.chosen_value = available_nodes[0]
        
        if txn_id in self.transactions:
            self.transactions[txn_id].consensus_rounds += 1
        
        return instance.chosen_value
    
    def handle_node_failure(self, node_id, failure_type, timestamp):
        """Mark node as failed"""
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)
        
        self.nodes[node_id].is_failed = True
        self.nodes[node_id].failure_type = failure_type
        self.current_time = timestamp
    
    def handle_node_recovery(self, node_id, timestamp):
        """Mark node as recovered"""
        if node_id in self.nodes:
            self.nodes[node_id].is_failed = False
            self.nodes[node_id].recovery_time = timestamp
        self.current_time = timestamp
    
    def handle_consensus_proposal(self, proposal_id, proposer_id, round_num, value, timestamp):
        """Handle Paxos proposal (Phase 1a)"""
        instance_id = f"consensus_{proposal_id}"
        
        if instance_id not in self.paxos_instances:
            self.paxos_instances[instance_id] = PaxosInstance(instance_id)
        
        instance = self.paxos_instances[instance_id]
        
        # Update promised round if higher
        if round_num > instance.promised_round:
            instance.promised_round = round_num
        
        self.current_time = timestamp
        
        # Extract txn_id from value if present
        if isinstance(value, dict) and 'txn_id' in value:
            txn_id = value['txn_id']
            if txn_id in self.transactions:
                self.transactions[txn_id].consensus_rounds = max(
                    self.transactions[txn_id].consensus_rounds, round_num
                )
    
    def handle_consensus_vote(self, proposal_id, voter_id, vote, promised_round, timestamp):
        """Handle Paxos vote (Phase 1b/2b)"""
        instance_id = f"consensus_{proposal_id}"
        
        if instance_id not in self.paxos_instances:
            self.paxos_instances[instance_id] = PaxosInstance(instance_id)
        
        instance = self.paxos_instances[instance_id]
        
        # Record vote
        instance.votes[voter_id] = (promised_round or 0, vote)
        
        # Check if we have quorum of accepts
        if vote == "accept":
            accept_count = sum(1 for _, v in instance.votes.values() if v == "accept")
            quorum_size = len(self.nodes) // 2 + 1
            
            if accept_count >= quorum_size and promised_round:
                if instance.accepted_round is None or promised_round >= instance.accepted_round:
                    instance.accepted_round = promised_round
        
        self.current_time = timestamp
    
    def abort_transaction(self, txn_id, reason, timestamp):
        """Explicitly abort transaction"""
        if txn_id not in self.transactions:
            return
        
        txn = self.transactions[txn_id]
        txn.status = TxnStatus.ABORTED
        txn.abort_reason = reason
        self.release_locks(txn_id)
        self.current_time = timestamp
        self.log.append(('ABORT', txn_id, timestamp))
    
    def save_checkpoint(self, checkpoint_id):
        """Save system state checkpoint"""
        self.checkpoints[checkpoint_id] = {
            'transactions': deepcopy({k: v for k, v in self.transactions.items() 
                                     if v.status == TxnStatus.IN_PROGRESS}),
            'partitions': deepcopy(self.partitions),
            'nodes': deepcopy(self.nodes),
            'time': self.current_time
        }
    
    def restore_checkpoint(self, checkpoint_id):
        """Restore from checkpoint"""
        if checkpoint_id not in self.checkpoints:
            return
        
        ckpt = self.checkpoints[checkpoint_id]
        # Restore in-progress transactions
        for txn_id, txn in ckpt['transactions'].items():
            if txn_id in self.transactions:
                self.transactions[txn_id] = deepcopy(txn)
        
        self.current_time = ckpt['time']
    
    def format_locks(self, txn):
        """Format locks for output"""
        locks_str = []
        for lock in txn.locks_held:
            locks_str.append(f"{lock.partition_id}:{lock.key}:{lock.lock_type.value}")
        return ",".join(sorted(locks_str)) if locks_str else ""
    
    def format_prepare_votes(self, txn):
        """Format prepare votes for output"""
        votes_str = []
        for partition_id, vote in sorted(txn.prepare_votes.items()):
            votes_str.append(f"{partition_id}:{vote}")
        return ",".join(votes_str) if votes_str else ""
    
    def get_transaction_result(self, txn):
        """Convert transaction to output row"""
        execution_time = 0
        if txn.commit_ts:
            execution_time = txn.commit_ts - txn.begin_ts
        elif txn.status == TxnStatus.ABORTED:
            execution_time = self.current_time - txn.begin_ts
        
        return {
            'txn_id': txn.txn_id,
            'status': txn.status.value,
            'coordinator_id': txn.coordinator_id,
            'participants': ",".join(sorted(txn.participants)) if txn.participants else "",
            'num_operations': len(txn.operations),
            'isolation_level': txn.isolation_level.value,
            'commit_timestamp': txn.commit_ts,
            'abort_reason': txn.abort_reason,
            'locks_acquired': self.format_locks(txn),
            'conflicts_detected': ",".join(sorted(txn.conflicts)) if txn.conflicts else "",
            'prepare_votes': self.format_prepare_votes(txn),
            'consensus_rounds': txn.consensus_rounds,
            'recovery_needed': txn.recovery_needed,
            'execution_time_ms': execution_time,
            'snapshot_version': txn.snapshot_version
        }

def load_stream(stream_path: Path) -> List[dict]:
    """Load and normalize stream entries"""
    entries = []
    if not stream_path.exists():
        return entries
    
    next_id = 1
    last_timestamp = 0
    
    for raw in stream_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        
        try:
            obj = json.loads(s)
        except:
            continue
        
        if not isinstance(obj, dict):
            continue
        
        if not isinstance(obj.get("stream_id"), int):
            obj["stream_id"] = next_id
        
        next_id = max(next_id, obj["stream_id"]) + 1
        
        if "timestamp" not in obj:
            obj["timestamp"] = last_timestamp + 1
        last_timestamp = obj["timestamp"]
        
        entries.append(obj)
    
    entries.sort(key=lambda x: x.get("stream_id", 0))
    return entries

def process_stream(entries: List[dict]) -> List[Dict]:
    """Process transaction log"""
    system = DistributedSystem()
    
    for entry in entries:
        entry_type = entry.get("type")
        timestamp = entry.get("timestamp", 0)
        
        if entry_type == "partition_definition":
            system.add_partition(
                entry.get("partition_id"),
                entry.get("key_range_start"),
                entry.get("key_range_end"),
                entry.get("replica_nodes", [])
            )
        
        elif entry_type == "txn_begin":
            system.begin_transaction(
                entry.get("txn_id"),
                entry.get("coordinator_id"),
                entry.get("isolation_level"),
                timestamp
            )
        
        elif entry_type == "txn_operation":
            system.add_operation(
                entry.get("txn_id"),
                entry.get("operation"),
                entry.get("partition_id"),
                entry.get("key"),
                entry.get("value"),
                entry.get("predicate"),
                timestamp
            )
        
        elif entry_type == "txn_commit":
            system.execute_2pc(entry.get("txn_id"), timestamp)
        
        elif entry_type == "txn_abort":
            system.abort_transaction(
                entry.get("txn_id"),
                entry.get("reason", "explicit_abort"),
                timestamp
            )
        
        elif entry_type == "node_failure":
            system.handle_node_failure(
                entry.get("node_id"),
                entry.get("failure_type"),
                timestamp
            )
        
        elif entry_type == "node_recovery":
            system.handle_node_recovery(entry.get("node_id"), timestamp)
        
        elif entry_type == "consensus_proposal":
            system.handle_consensus_proposal(
                entry.get("proposal_id"),
                entry.get("proposer_id"),
                entry.get("round"),
                entry.get("value"),
                timestamp
            )
        
        elif entry_type == "consensus_vote":
            system.handle_consensus_vote(
                entry.get("proposal_id"),
                entry.get("voter_id"),
                entry.get("vote"),
                entry.get("promised_round"),
                timestamp
            )
        
        elif entry_type == "checkpoint":
            checkpoint_id = entry.get("checkpoint_id")
            action = entry.get("action")
            if action == "save":
                system.save_checkpoint(checkpoint_id)
            elif action == "restore":
                system.restore_checkpoint(checkpoint_id)
    
    # Generate results
    results = []
    for txn_id in sorted(system.transactions.keys()):
        txn = system.transactions[txn_id]
        result = system.get_transaction_result(txn)
        results.append(result)
    
    return results

def format_value(value):
    """Format value for CSV output"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)

# Main execution
entries = load_stream(STREAM)
results = process_stream(entries)

# Write output CSV
with OUT.open("w", newline="") as f:
    writer = csv.writer(f)
    cols = ["txn_id", "status", "coordinator_id", "participants", "num_operations",
            "isolation_level", "commit_timestamp", "abort_reason", "locks_acquired",
            "conflicts_detected", "prepare_votes", "consensus_rounds", "recovery_needed",
            "execution_time_ms", "snapshot_version"]
    writer.writerow(cols)
    
    for result in results:
        row = [
            result['txn_id'],
            result['status'],
            result['coordinator_id'],
            result['participants'],
            str(result['num_operations']),
            result['isolation_level'],
            format_value(result['commit_timestamp']),
            format_value(result['abort_reason']),
            result['locks_acquired'],
            result['conflicts_detected'],
            result['prepare_votes'],
            str(result['consensus_rounds']),
            "true" if result['recovery_needed'] else "false",
            str(result['execution_time_ms']),
            format_value(result['snapshot_version'])
        ]
        writer.writerow(row)
PY