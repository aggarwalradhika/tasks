#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Reference Solution for Simple In-Memory Database

Implements basic database with primary index and simple query processing.
Indexes are optional - this solution works without them.
"""

import json, csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional, Any
from copy import deepcopy

OPS = Path("/workdir/data/operations.jsonl")
RESULTS = Path("/workdir/results.csv")

class Record:
    """Database record"""
    def __init__(self, id: int, fields: dict):
        self.id = id
        self.fields = fields

class Database:
    """Simple in-memory database"""
    def __init__(self):
        self.records: Dict[int, Record] = {}
        
    def clone(self):
        db = Database()
        db.records = {k: Record(v.id, v.fields.copy()) for k, v in self.records.items()}
        return db
    
    def insert(self, id: int, fields: dict):
        self.records[id] = Record(id, fields)
    
    def update(self, id: int, fields: dict):
        if id in self.records:
            self.records[id].fields.update(fields)
    
    def delete(self, id: int):
        if id in self.records:
            del self.records[id]
    
    def query(self, query_spec: dict) -> dict:
        conditions = query_spec.get('conditions', [])
        join_type = query_spec.get('join_type', 'AND')
        sort_by = query_spec.get('sort_by')
        sort_order = query_spec.get('sort_order', 'asc')
        limit = query_spec.get('limit')
        aggregation = query_spec.get('aggregation')
        
        # Find matching records
        results = []
        for record in self.records.values():
            if self._matches(record, conditions, join_type):
                results.append(record)
        
        # Sort - use record ID as secondary key for stable, deterministic ordering
        if sort_by and results:
            # Check if sort field exists in at least one record
            has_sort_field = any(sort_by in r.fields for r in results)
            if has_sort_field:
                # Sort by (sort_field, id) for deterministic ordering when sort values are equal
                if sort_order == 'desc':
                    results.sort(key=lambda r: (r.fields.get(sort_by, 0), r.id), reverse=True)
                else:
                    results.sort(key=lambda r: (r.fields.get(sort_by, 0), r.id))
        
        # Limit - FIXED: Handle negative and zero limits correctly
        if limit is not None:
            if limit <= 0:
                results = []
            else:
                results = results[:limit]
        
        # Aggregation
        agg_value = ""
        if aggregation:
            agg_value = self._aggregate(results, aggregation)
        
        result_ids = ",".join(str(r.id) for r in results[:100])
        
        return {
            "query_id": query_spec.get('query_id', ''),
            "result_count": str(len(results)),
            "aggregation_value": str(agg_value) if agg_value != "" else "",
            "result_ids": result_ids
        }
    
    def _matches(self, record: Record, conditions: list, join_type: str) -> bool:
        if not conditions:
            return True
        
        matches = []
        for cond in conditions:
            field = cond['field']
            operator = cond['operator']
            value = cond['value']
            
            if field not in record.fields:
                matches.append(False)
                continue
            
            field_value = record.fields[field]
            matches.append(self._compare(field_value, operator, value))
        
        if join_type == 'AND':
            return all(matches)
        else:  # OR
            return any(matches)
    
    def _compare(self, field_value: Any, operator: str, value: Any) -> bool:
        try:
            if operator == '==':
                if isinstance(field_value, str) and isinstance(value, str):
                    return field_value.lower() == value.lower()
                return field_value == value
            elif operator == '!=':
                if isinstance(field_value, str) and isinstance(value, str):
                    return field_value.lower() != value.lower()
                return field_value != value
            elif operator == '<':
                return field_value < value
            elif operator == '>':
                return field_value > value
            elif operator == '<=':
                return field_value <= value
            elif operator == '>=':
                return field_value >= value
            elif operator == 'CONTAINS':
                if isinstance(field_value, list):
                    return any(str(value).lower() == str(v).lower() for v in field_value)
        except:
            return False
        return False
    
    def _aggregate(self, results: List[Record], agg_spec: dict) -> Any:
        agg_type = agg_spec['type']
        field = agg_spec.get('field')
        
        if agg_type == 'count':
            return len(results)
        
        if not field:
            return ""
        
        values = [r.fields.get(field) for r in results if field in r.fields]
        if not values:
            return ""
        
        try:
            if agg_type == 'sum':
                return round(sum(values), 2)
            elif agg_type == 'avg':
                return round(sum(values) / len(values), 2)
            elif agg_type == 'min':
                return min(values)
            elif agg_type == 'max':
                return max(values)
        except:
            return ""
        
        return ""

def parse_compact_insert(s: str) -> Optional[dict]:
    parts = s.split(':')
    if len(parts) < 5 or parts[0] != 'insert':
        return None
    try:
        return {
            "op": "insert",
            "id": int(parts[1]),
            "fields": {
                "category": parts[2],
                "status": parts[3],
                "priority": int(parts[4]),
                "score": 0.5,
                "tags": [],
                "timestamp": 1000
            }
        }
    except:
        return None

def parse_compact_query(s: str) -> Optional[dict]:
    parts = s.split(':')
    if len(parts) < 5 or parts[0] != 'query':
        return None
    try:
        value = parts[4]
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                pass
        
        return {
            "op": "query",
            "query_id": parts[1],
            "conditions": [{"field": parts[2], "operator": parts[3], "value": value}],
            "join_type": "AND"
        }
    except:
        return None

def load_operations(ops_path: Path) -> List[dict]:
    operations = []
    if not ops_path.exists():
        return operations
    
    for raw in ops_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        
        try:
            operations.append(json.loads(s))
            continue
        except:
            pass
        
        if s.startswith("insert:"):
            op = parse_compact_insert(s)
            if op:
                operations.append(op)
        elif s.startswith("query:"):
            op = parse_compact_query(s)
            if op:
                operations.append(op)
    
    return operations

# Main
db = Database()
checkpoints = {}
query_results = []

operations = load_operations(OPS)

for op in operations:
    op_type = op.get('op')
    
    if op_type == 'insert':
        db.insert(op['id'], op.get('fields', {}))
    elif op_type == 'update':
        db.update(op['id'], op.get('fields', {}))
    elif op_type == 'delete':
        db.delete(op['id'])
    elif op_type == 'query':
        result = db.query(op)
        query_results.append(result)
    elif op_type == 'checkpoint':
        cp_id = op.get('checkpoint_id')
        action = op.get('action')
        if action == 'save' and cp_id:
            checkpoints[cp_id] = db.clone()
        elif action == 'restore' and cp_id and cp_id in checkpoints:
            db = checkpoints[cp_id].clone()

# Write results
with RESULTS.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "query_id", "result_count", "aggregation_value", "result_ids"
    ])
    writer.writeheader()
    writer.writerows(query_results)

PYTHON_EOF