from __future__ import annotations
import csv
from pathlib import Path
from typing import List
from apex_arena._types import GradingResult

EXPECTED = Path("/tests/expected_sessions.csv")
CANDIDATE = Path("/workdir/out/sessions.csv")


def _csv_rows_from_file(path: Path) -> List[List[str]]:
    # Parse CSV into rows of fields; ignore completely blank rows
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader if not all((c.strip() == "" for c in row))]


def grade(transcript: str) -> GradingResult:
    subscores = {"result": 0.0}
    weights = {"result": 1.0}

    if not EXPECTED.exists():
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback="Expected file /tests/expected_sessions.csv not found.",
        )
    if not CANDIDATE.exists():
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback="Output file /workdir/out/sessions.csv not found.",
        )

    try:
        exp_rows = _csv_rows_from_file(EXPECTED)
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback=f"Could not read expected CSV: {e}",
        )
    try:
        got_rows = _csv_rows_from_file(CANDIDATE)
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback=f"Could not read candidate CSV: {e}",
        )

    # Exact comparison: same number of rows and identical field-by-field content
    if len(got_rows) != len(exp_rows):
        return GradingResult(
            score=0.0,
            subscores=subscores,
            weights=weights,
            feedback=f"Row count mismatch: expected {len(exp_rows)}, got {len(got_rows)}.",
        )
    for i, (g_row, e_row) in enumerate(zip(got_rows, exp_rows), start=1):
        if g_row != e_row:
            return GradingResult(
                score=0.0,
                subscores=subscores,
                weights=weights,
                feedback=f"Mismatch at line {i}: expected {e_row} vs got {g_row}",
            )

    # Perfect match
    subscores["result"] = 1.0
    return GradingResult(score=1.0, subscores=subscores, weights=weights, feedback="OK")
