# tests/grader.py
"""
Binary grader:
  1) checks sol.csv exists & parsable as one int per non-empty line
  2) checks row count equals expected
  3) returns 1 only if there is an exact match for all lines
"""

import json
from pathlib import Path
from typing import Optional, List, Dict

try:
    from pydantic import BaseModel, Field
except Exception:
    class BaseModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def model_dump(self):
            return self.__dict__
    def Field(default=None, **kwargs):
        return default

DATA_DIR = Path("/workdir/data")
SOL_PATH  = Path("/workdir/sol.csv")

class GradingResult(BaseModel):
    score: float = Field(..., description="Final score in [0,1]")
    feedback: str = Field(default="")
    # Keep these for platform compatibility; they do NOT affect the binary score.
    subscores: Dict[str, float] = Field(default_factory=dict)
    details: Dict[str, object] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)

def _matches(pch: str, sch: str, wildcards: bool) -> bool:
    return pch == sch or (wildcards and pch == '?')

def _first_index_with_budget(s: str, p: str, k: int, wildcards: bool, banned: set) -> int:
    n, m = len(s), len(p)
    if m > n:
        return -1
    # O(n*m) checker: adequate for *correctness* in grading.
    for i in range(n - m + 1):
        used = 0
        ok = True
        for j in range(m):
            if not _matches(p[j], s[i + j], wildcards):
                if (i + j) in banned:
                    ok = False
                    break
                used += 1
                if used > k:
                    ok = False
                    break
        if ok:
            return i
    return -1

def _expected_outputs() -> Optional[List[int]]:
    try:
        obj = json.loads((DATA_DIR / "cases.json").read_text(encoding="utf-8"))
        cases = obj.get("cases", [])
        exp = []
        for case in cases:
            s = case["s"]
            patterns = case.get("patterns")
            if patterns is None and "pattern" in case:
                # Backward-compat: lift single pattern into the expected "patterns" list
                patterns = [{
                    "pattern": case["pattern"],
                    "k": int(case.get("k", 1)),
                    "wildcards": bool(case.get("wildcards", False)),
                    "banned_indices": case.get("banned_indices", [])
                }]
            if patterns is None:
                raise ValueError("Case missing 'patterns' (or 'pattern')")
            for pat in patterns:
                p      = pat["pattern"]
                k      = int(pat.get("k", 1))
                wild   = bool(pat.get("wildcards", False))
                banned = set(pat.get("banned_indices", []))
                exp.append(_first_index_with_budget(s, p, k, wild, banned))
        return exp
    except Exception:
        return None

def _read_sol_lines(sol_path: Path) -> Optional[List[int]]:
    if not sol_path.exists():
        return None
    try:
        lines = [ln.strip() for ln in sol_path.read_text(encoding="utf-8").splitlines()]
        # ignore trailing empties
        lines = [ln for ln in lines if ln != ""]
        return [int(x) for x in lines]
    except Exception:
        return None

def grade(transcript=None) -> GradingResult:
    subscores = {"exact_match": 0.0}
    weights = {"exact_match": 1.0}
    details: Dict[str, object] = {}
    weights: Dict[str, float] = {}

    expected = _expected_outputs()
    if expected is None:
        return GradingResult(score=0.0, feedback="Invalid or unreadable /workdir/data/cases.json",
                             subscores=subscores, details=details, weights=weights)

    got = _read_sol_lines(SOL_PATH)
    if got is None:
        return GradingResult(score=0.0, feedback="Missing or invalid /workdir/sol.csv",
                             subscores=subscores, details=details, weights=weights)
 

    if len(got) != len(expected):
        details["expected_rows"] = len(expected)
        details["found_rows"] = len(got)
        return GradingResult(score=0.0,
                             feedback=f"Row count mismatch: expected {len(expected)} lines, found {len(got)}",
                             subscores=subscores, details=details, weights=weights)


    if got == expected:
        subscores["exact_match"] = 1.0
        return GradingResult(score=1.0, feedback="Correct.",
                             subscores=subscores, details=details, weights=weights)
    else:
        # brief diff preview
        preview = []
        for i, (g, e) in enumerate(zip(got, expected)):
            if g != e:
                preview.append({"line": i, "expected": e, "got": g})
            if len(preview) >= 10:
                break
        details["mismatch_preview"] = preview
        return GradingResult(score=0.0, feedback="Incorrect outputs.",
                             subscores=subscores, details=details, weights=weights)
