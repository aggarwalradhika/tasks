#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PY'
#!/usr/bin/env python3
"""
Reference Solution for Distributed Transaction Coordinator with Consensus

This solution exactly matches the grader's logic to ensure 100% correctness.
"""
import json, csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set
from enum import Enum

# Constants matching grader
PAXOS_QUORUM_RATIO = 0.5
LOCK_TIMEOUT_MS = 10000

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
    def __init__(self, txn_id, lock_type, partition_id, key):
        self.txn_id = txn_id
        self.lock_type = lock_type
        self.partition_id = partition_id
        self.key = key
        self.acquired_at = 0

class Transaction:
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

class Partition:
    def __init__(self, partition_id, key_range_start, key_range_end, replica_nodes):
        self.partition_id = partition_id
        self.key_range_start = key_range_start
        self.key_range_end = key_range_end
        self.replica_nodes = replica_nodes
        self.data = {}
        self.locks = defaultdict(list)

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.is_failed = False
        self.failure_type = None

class PaxosInstance:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.promised_round = 0
        self.accepted_round = None
        self.accepted_value = None
        self.chosen_value = None

class DistributedSystem:
    def __init__(self):
        self.transactions: Dict[str, Transaction] = {}
        self.partitions: Dict[str, Partition] = {}
        self.nodes: Dict[str, Node] = {}
        self.paxos_instances: Dict[str, PaxosInstance] = {}
        self.current_time = 0
        self.checkpoints = {}
        
    def add_partition(self, partition_id, key_range_start, key_range_end, replica_nodes):
        self.partitions[partition_id] = Partition(
            partition_id, key_range_start, key_range_end, replica_nodes
        )
        for node_id in replica_nodes:
            if node_id not in self.nodes:
                self.nodes[node_id] = Node(node_id)
    
    def begin_transaction(self, txn_id, coordinator_id, isolation_level, timestamp):
        txn = Transaction(txn_id, coordinator_id, isolation_level, timestamp)
        self.transactions[txn_id] = txn
        self.current_time = timestamp
    
    def add_operation(self, txn_id, operation, partition_id, key, value, predicate, timestamp):
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
        self.current_time = timestamp
    
    def can_acquire_lock(self, partition_id, key, lock_type, txn_id):
        if partition_id not in self.partitions:
            return True
        
        partition = self.partitions[partition_id]
        existing_locks = partition.locks[key]
        
        for lock in existing_locks:
            if lock.txn_id == txn_id:
                continue
            if lock_type == LockType.EXCLUSIVE or lock.lock_type == LockType.EXCLUSIVE:
                return False
            if lock_type == LockType.UPDATE and lock.lock_type == LockType.UPDATE:
                return False
        
        return True
    
    def acquire_lock(self, txn_id, partition_id, key, lock_type, timestamp):
        if txn_id not in self.transactions:
            return False
        
        txn = self.transactions[txn_id]
        
        if self.can_acquire_lock(partition_id, key, lock_type, txn_id):
            lock = Lock(txn_id, lock_type, partition_id, key)
            lock.acquired_at = timestamp
            self.partitions[partition_id].locks[key].append(lock)
            txn.locks_held.append(lock)
            return True
        
        blocking_txns = set()
        for lock in self.partitions[partition_id].locks[key]:
            if lock.txn_id != txn_id:
                blocking_txns.add(lock.txn_id)
        
        for blocker_id in blocking_txns:
            if blocker_id in self.transactions:
                if self.creates_cycle(blocker_id, txn_id):
                    if txn.begin_ts > self.transactions[blocker_id].begin_ts:
                        txn.status = TxnStatus.ABORTED
                        txn.abort_reason = "deadlock_victim"
                        self.release_locks(txn_id)
                        return False
        
        if timestamp - txn.begin_ts > LOCK_TIMEOUT_MS:
            txn.status = TxnStatus.ABORTED
            txn.abort_reason = "lock_timeout"
            self.release_locks(txn_id)
            return False
        
        txn.conflicts.update(blocking_txns)
        return False
    
    def creates_cycle(self, from_txn, to_txn):
        visited = set()
        
        def dfs(current):
            if current == to_txn:
                return True
            if current in visited:
                return False
            visited.add(current)
            
            if current not in self.transactions:
                return False
            
            txn = self.transactions[current]
            if txn.waiting_for:
                return dfs(txn.waiting_for)
            return False
        
        return dfs(from_txn)
    
    def release_locks(self, txn_id):
        if txn_id not in self.transactions:
            return
        
        txn = self.transactions[txn_id]
        for lock in txn.locks_held:
            partition = self.partitions[lock.partition_id]
            if lock in partition.locks[lock.key]:
                partition.locks[lock.key].remove(lock)
        txn.locks_held = []
    
    def execute_2pc(self, txn_id, timestamp):
        if txn_id not in self.transactions:
            return
        
        txn = self.transactions[txn_id]
        self.current_time = timestamp
        
        if txn.status == TxnStatus.ABORTED:
            self.release_locks(txn_id)
            return
        
        if not txn.participants:
            txn.status = TxnStatus.COMMITTED
            txn.commit_ts = timestamp
            return
        
        if txn.coordinator_id in self.nodes and self.nodes[txn.coordinator_id].is_failed:
            new_coordinator = self.run_paxos_election(txn_id, timestamp)
            if new_coordinator:
                txn.coordinator_id = new_coordinator
                txn.recovery_needed = True
            else:
                txn.status = TxnStatus.ABORTED
                txn.abort_reason = "coordinator_failed_no_quorum"
                self.release_locks(txn_id)
                return
        
        prepare_success = self.run_prepare_phase(txn_id, timestamp)
        
        if not prepare_success:
            txn.status = TxnStatus.ABORTED
            if not txn.abort_reason:
                txn.abort_reason = "prepare_failed"
            self.release_locks(txn_id)
            return
        
        self.run_commit_phase(txn_id, timestamp)
    
    def run_prepare_phase(self, txn_id, timestamp):
        txn = self.transactions[txn_id]
        
        for partition_id in txn.participants:
            partition = self.partitions[partition_id]
            
            reachable_replicas = [
                n for n in partition.replica_nodes
                if n not in self.nodes or not self.nodes[n].is_failed
            ]
            
            quorum_size = len(partition.replica_nodes) // 2 + 1
            
            if len(reachable_replicas) < quorum_size:
                txn.prepare_votes[partition_id] = "NO"
                txn.abort_reason = "partition_unreachable"
                return False
            
            if txn.isolation_level == IsolationLevel.SNAPSHOT:
                if self.has_write_conflict(txn_id, partition_id):
                    txn.prepare_votes[partition_id] = "NO"
                    txn.abort_reason = "write_conflict"
                    return False
            
            success = True
            for op in txn.operations:
                if op['partition_id'] != partition_id:
                    continue
                
                lock_type = LockType.EXCLUSIVE if op['operation'] in ['write', 'delete'] else LockType.SHARED
                
                if txn.isolation_level == IsolationLevel.SNAPSHOT and op['operation'] == 'read':
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
        
        return True
    
    def run_commit_phase(self, txn_id, timestamp):
        txn = self.transactions[txn_id]
        txn.status = TxnStatus.COMMITTED
        txn.commit_ts = timestamp
        
        for op in txn.operations:
            if op['operation'] == 'write':
                partition = self.partitions[op['partition_id']]
                key = op['key']
                if key not in partition.data:
                    partition.data[key] = []
                partition.data[key].append((op['value'], timestamp, txn_id))
        
        self.release_locks(txn_id)
    
    def has_write_conflict(self, txn_id, partition_id):
        txn = self.transactions[txn_id]
        partition = self.partitions[partition_id]
        
        for op in txn.operations:
            if op['partition_id'] != partition_id or op['operation'] != 'write':
                continue
            
            key = op['key']
            if key in partition.data:
                for value, version_ts, writer_txn_id in partition.data[key]:
                    if version_ts > txn.snapshot_version and writer_txn_id != txn_id:
                        if writer_txn_id in self.transactions:
                            writer = self.transactions[writer_txn_id]
                            if writer.status == TxnStatus.COMMITTED:
                                txn.conflicts.add(writer_txn_id)
                                return True
        
        return False
    
    def run_paxos_election(self, txn_id, timestamp):
        instance_id = f"coordinator_election_{txn_id}"
        
        if instance_id not in self.paxos_instances:
            self.paxos_instances[instance_id] = PaxosInstance(instance_id)
        
        instance = self.paxos_instances[instance_id]
        
        available_nodes = [
            node_id for node_id, node in self.nodes.items()
            if not node.is_failed
        ]
        
        quorum_size = len(self.nodes) // 2 + 1
        if len(available_nodes) < quorum_size:
            return None
        
        instance.chosen_value = available_nodes[0]
        
        if txn_id in self.transactions:
            self.transactions[txn_id].consensus_rounds = 1
        
        return instance.chosen_value
    
    def handle_node_failure(self, node_id, failure_type, timestamp):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)
        
        self.nodes[node_id].is_failed = True
        self.nodes[node_id].failure_type = failure_type
        self.current_time = timestamp
    
    def handle_node_recovery(self, node_id, timestamp):
        if node_id in self.nodes:
            self.nodes[node_id].is_failed = False
        self.current_time = timestamp
    
    def handle_consensus_proposal(self, proposal_id, proposer_id, round_num, value, timestamp):
        instance_id = f"consensus_{proposal_id}"
        
        if instance_id not in self.paxos_instances:
            self.paxos_instances[instance_id] = PaxosInstance(instance_id)
        
        instance = self.paxos_instances[instance_id]
        
        if round_num > instance.promised_round:
            instance.promised_round = round_num
        
        self.current_time = timestamp
    
    def handle_consensus_vote(self, proposal_id, voter_id, vote, promised_round, timestamp):
        instance_id = f"consensus_{proposal_id}"
        
        if instance_id not in self.paxos_instances:
            self.paxos_instances[instance_id] = PaxosInstance(instance_id)
        
        instance = self.paxos_instances[instance_id]
        
        if vote == "accept" and promised_round:
            if promised_round >= instance.accepted_round or instance.accepted_round is None:
                instance.accepted_round = promised_round
        
        self.current_time = timestamp
    
    def abort_transaction(self, txn_id, reason, timestamp):
        if txn_id not in self.transactions:
            return
        
        txn = self.transactions[txn_id]
        txn.status = TxnStatus.ABORTED
        txn.abort_reason = reason
        self.release_locks(txn_id)
        self.current_time = timestamp
    
    def format_locks(self, txn):
        locks_str = []
        for lock in txn.locks_held:
            locks_str.append(f"{lock.partition_id}:{lock.key}:{lock.lock_type.value}")
        return ",".join(sorted(locks_str)) if locks_str else ""
    
    def format_prepare_votes(self, txn):
        votes_str = []
        for partition_id, vote in sorted(txn.prepare_votes.items()):
            votes_str.append(f"{partition_id}:{vote}")
        return ",".join(votes_str) if votes_str else ""
    
    def get_transaction_result(self, txn):
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
            'commit_timestamp': txn.commit_ts if txn.commit_ts else None,
            'abort_reason': txn.abort_reason if txn.abort_reason else None,
            'locks_acquired': self.format_locks(txn),
            'conflicts_detected': ",".join(sorted(txn.conflicts)) if txn.conflicts else "",
            'prepare_votes': self.format_prepare_votes(txn),
            'consensus_rounds': txn.consensus_rounds,
            'recovery_needed': txn.recovery_needed,
            'execution_time_ms': execution_time,
            'snapshot_version': txn.snapshot_version
        }

def load_stream(stream_path: Path) -> List[dict]:
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
    
    results = []
    for txn_id in sorted(system.transactions.keys()):
        txn = system.transactions[txn_id]
        result = system.get_transaction_result(txn)
        results.append(result)
    
    return results

def format_value(value):
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