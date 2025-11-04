#!/usr/bin/env python3
"""
Grader for Simple In-Memory Database task.
Recomputes expected outputs and validates correctness.
"""
import json, csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any
from copy import deepcopy

# Pydantic shim
try:
    from pydantic import BaseModel
except:
    class BaseModel:
        def __init__(self, **kw): self.__dict__.update(kw)
        def model_dump(self): return self.__dict__

class GradingResult(BaseModel):
    score: float
    feedback: str | None = None
    subscores: dict = {}
    details: dict | None = None
    weights: dict | None = None

WORKDIR = Path("/workdir")
OPS = WORKDIR / "data" / "operations.jsonl"
RESULTS = WORKDIR / "results.csv"

RESULTS_COLS = ["query_id", "result_count", "aggregation_value", "result_ids"]

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

def process_operations(operations: List[dict]) -> List[dict]:
    """Process all operations and generate query results"""
    db = Database()
    checkpoints = {}
    query_results = []
    
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
    
    return query_results

def read_csv_file(path: Path, expected_cols: List[str]) -> Optional[List[dict]]:
    """Read CSV file"""
    if not path.exists():
        return None
    try:
        rows = []
        with path.open() as f:
            reader = csv.DictReader(f)
            if list(reader.fieldnames) != expected_cols:
                return None
            rows = list(reader)
        return rows
    except Exception:
        return None

def grade(transcript: str | None = None) -> GradingResult:
    """
    Grade the submission by comparing against expected outputs.
    Returns score=1.0 only if ALL checks pass, otherwise 0.0.
    """
    subscores = {"all_passes": 0.0}
    weights = {"all_passes": 1.0}
    
    operations = load_operations(OPS)
    if not operations:
        return GradingResult(
            score=0.0,
            feedback="No valid operations in data/operations.jsonl",
            subscores=subscores,
            weights=weights
        )
    
    # Generate expected outputs
    try:
        exp_results = process_operations(operations)
    except Exception as e:
        return GradingResult(
            score=0.0,
            feedback=f"Error processing operations: {str(e)}",
            subscores=subscores,
            weights=weights
        )
    
    # Check results.csv exists and has correct schema
    sol_results = read_csv_file(RESULTS, RESULTS_COLS)
    if sol_results is None:
        return GradingResult(
            score=0.0,
            feedback="Missing or invalid /workdir/results.csv, or incorrect schema/columns",
            subscores=subscores,
            weights=weights
        )
    
    # Compare row count
    if len(sol_results) != len(exp_results):
        return GradingResult(
            score=0.0,
            feedback=f"Results row count mismatch. Expected {len(exp_results)}, got {len(sol_results)}",
            subscores=subscores,
            weights=weights
        )
    
    # Cell-by-cell comparison
    mismatches = []
    for i in range(len(exp_results)):
        for col in RESULTS_COLS:
            exp_val = str(exp_results[i].get(col, ""))
            sol_val = str(sol_results[i].get(col, ""))
            if exp_val != sol_val:
                mismatches.append({
                    "row": i + 1,
                    "column": col,
                    "expected": exp_val,
                    "got": sol_val
                })
                if len(mismatches) >= 50:
                    break
        if len(mismatches) >= 50:
            break
    
    if mismatches:
        return GradingResult(
            score=0.0,
            feedback=f"Found {len(mismatches)}+ cell mismatches. Output does not match expected.",
            subscores=subscores,
            weights=weights,
            details={"mismatches": mismatches[:25]}
        )
    
    # All checks passed
    subscores["all_passes"] = 1.0
    return GradingResult(
        score=1.0,
        feedback="All checks passed! Database operations implemented correctly.",
        subscores=subscores,
        weights=weights
    )

if __name__ == "__main__":
    result = grade(None)
    print(result.model_dump() if hasattr(result, "model_dump") else result.__dict__)