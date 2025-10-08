"""
Regex Subset Construction – Hard Grader (strict + harsh weights)

Schema (unchanged):
  sol.csv header: regex_id,string,accepts
  regex_id: 1-based over VALID regex lines only (malformed skipped)
  string: literal from strings.txt (ε denotes empty string)
  accepts: 0 or 1
Ordering: regex_id asc, then strings.txt order
"""

from pathlib import Path
from dataclasses import dataclass
import csv, io

DATA = Path("/workdir/data")
OUT  = Path("/workdir/sol.csv")
COLS = ["regex_id","string","accepts"]

# ---- Always return dict subscores/weights (pydantic-friendly) ----
try:
    from apex_arena._types import GradingResult  # type: ignore
except Exception:
    @dataclass
    class GradingResult:  # type: ignore
        score: float
        feedback: str
        subscores: dict
        weights: dict
        details: dict | None = None

def _res(score: float, feedback: str, subs: dict, wts: dict, details: dict | None = None):
    try:
        from apex_arena._types import GradingResult  # type: ignore
        return GradingResult(score=float(score), feedback=str(feedback),
                             subscores=dict(subs), weights=dict(wts), details=dict(details or {}))
    except Exception:
        return {"score": float(score), "feedback": str(feedback),
                "subscores": dict(subs), "weights": dict(wts), "details": dict(details or {})}

# ---------------- Regex parsing (with implicit concatenation) ----------------
def tokenize(s: str) -> list[str]:
    # No whitespace anywhere; alphabet limited to a,b,(,),|,*
    if any(ch.isspace() for ch in s):
        raise ValueError("whitespace not allowed")
    ok = set("ab()|*")
    for ch in s:
        if ch not in ok:
            raise ValueError(f"invalid char {ch}")
    return list(s)

def insert_concat(tokens: list[str]) -> list[str]:
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i+1 < len(tokens):
            t1, t2 = tokens[i], tokens[i+1]
            if (t1 in "ab)" or t1 == "*") and (t2 in "ab("):
                out.append(".")
    return out

def to_postfix(tokens: list[str]) -> list[str]:
    prec = {"*":3, ".":2, "|":1}
    out, st = [], []
    for t in tokens:
        if t in ("a","b"):
            out.append(t)
        elif t == "(":
            st.append(t)
        elif t == ")":
            while st and st[-1] != "(":
                out.append(st.pop())
            if not st:
                raise ValueError("mismatched )")
            st.pop()
        elif t == "*":
            out.append(t)  # postfix
        elif t in (".","|"):
            while st and st[-1] != "(" and prec[st[-1]] >= prec[t]:
                out.append(st.pop())
            st.append(t)
        else:
            raise ValueError(f"bad token {t}")
    while st:
        x = st.pop()
        if x == "(":
            raise ValueError("mismatched (")
        out.append(x)
    return out

# ---------------- Thompson NFA & simulate ----------------
class NFA:
    def __init__(self, start, accept, trans, eps):
        self.start = start
        self.accept = accept
        self.trans = trans    # dict[state][symbol] -> set(states)
        self.eps   = eps      # dict[state] -> set(states)

def nfa_concat(a: 'NFA', b: 'NFA') -> 'NFA':
    a.eps.setdefault(a.accept, set()).add(b.start)
    return NFA(a.start, b.accept, {**a.trans, **b.trans}, {**a.eps, **b.eps})

def nfa_union(a: 'NFA', b: 'NFA', ns):
    s = ns(); t = ns()
    eps, trans = {}, {}
    for d in (a.eps,b.eps):
        for k,v in d.items(): eps.setdefault(k,set()).update(v)
    for d in (a.trans,b.trans):
        for k,mp in d.items():
            for c,vs in mp.items():
                trans.setdefault(k,{}).setdefault(c,set()).update(vs)
    eps.setdefault(s,set()).update([a.start, b.start])
    eps.setdefault(a.accept,set()).add(t)
    eps.setdefault(b.accept,set()).add(t)
    return NFA(s,t,trans,eps)

def nfa_star(a: 'NFA', ns):
    s = ns(); t = ns()
    eps, trans = {}, {}
    for k,v in a.eps.items(): eps.setdefault(k,set()).update(v)
    for k,mp in a.trans.items():
        for c,vs in mp.items():
            trans.setdefault(k,{}).setdefault(c,set()).update(vs)
    eps.setdefault(s,set()).update([a.start, t])
    eps.setdefault(a.accept,set()).update([a.start, t])
    return NFA(s,t,trans,eps)

def nfa_symbol(c: str, ns) -> 'NFA':
    s = ns(); t = ns()
    return NFA(s,t,{s:{c:{t}}},{})

def compile_nfa(postfix: list[str]) -> 'NFA':
    next_state = iter(range(1_000_000)).__next__
    st = []
    for t in postfix:
        if t in ("a","b"):
            st.append(nfa_symbol(t, next_state))
        elif t == ".":
            b = st.pop(); a = st.pop()
            st.append(nfa_concat(a,b))
        elif t == "|":
            b = st.pop(); a = st.pop()
            st.append(nfa_union(a,b,next_state))
        elif t == "*":
            a = st.pop()
            st.append(nfa_star(a,next_state))
        else:
            raise ValueError(f"bad token in postfix: {t}")
    if len(st) != 1:
        raise ValueError("invalid postfix stack")
    return st[0]

def eps_closure(nfa: NFA, states: set[int]) -> set[int]:
    stack = list(states); seen = set(states)
    while stack:
        s = stack.pop()
        for t in nfa.eps.get(s, ()):
            if t not in seen:
                seen.add(t); stack.append(t)
    return seen

def move(nfa: NFA, states: set[int], ch: str) -> set[int]:
    nxt = set()
    for s in states:
        for t in nfa.trans.get(s, {}).get(ch, ()):
            nxt.add(t)
    return nxt

def accepts(nfa: NFA, s: str) -> bool:
    cur = eps_closure(nfa, {nfa.start})
    for ch in s:
        cur = eps_closure(nfa, move(nfa, cur, ch))
        if not cur:
            break
    return nfa.accept in cur

# ---------------- IO & ground truth ----------------
def read_regexes(path: Path) -> list[tuple[int,str]]:
    raw = path.read_text(encoding="utf-8").splitlines()
    valid = []
    for _orig_idx, line in enumerate(raw, start=1):
        line = line.strip()
        if not line:
            continue  # empty regex not allowed
        try:
            pf = to_postfix(insert_concat(tokenize(line)))
            _ = compile_nfa(pf)  # validate build (malformed raises)
            valid.append((len(valid)+1, line))  # reindex valid regex_id
        except Exception:
            continue  # skip malformed
    return valid

def read_strings(path: Path) -> list[str]:
    # Preserve duplicates and order; treat literal ε as empty string
    return [("" if s.strip() == "ε" else s.strip())
            for s in path.read_text(encoding="utf-8").splitlines()]

def ground_truth(regexes: list[tuple[int,str]], strings: list[str]) -> list[dict]:
    rows = []
    for rid, rx in regexes:
        pf = to_postfix(insert_concat(tokenize(rx)))
        nfa = compile_nfa(pf)
        for s in strings:
            ok = accepts(nfa, s)
            rows.append({"regex_id": str(rid),
                         "string": ("ε" if s == "" else s),
                         "accepts": "1" if ok else "0"})
    return rows

def read_submission() -> list[dict]:
    if not OUT.exists():
        raise FileNotFoundError("Missing /workdir/sol.csv")
    rdr = csv.reader(io.StringIO(OUT.read_text(encoding="utf-8-sig"), newline=""))
    try:
        header = next(rdr)
    except StopIteration:
        raise ValueError("Empty CSV")
    if header != COLS:
        raise ValueError(f"Header mismatch. Expected {COLS}, got {header}")
    rows = []
    for line_no, row in enumerate(rdr, start=2):
        if len(row) != 3:
            raise ValueError(f"Row {line_no}: wrong column count")
        rid, s, acc = row
        # No leading/trailing spaces allowed
        if rid != rid.strip() or s != s.strip() or acc != acc.strip():
            raise ValueError(f"Row {line_no}: fields must not contain leading/trailing spaces")
        if acc not in ("0","1"):
            raise ValueError(f"Row {line_no}: accepts must be 0/1")
        rows.append({"regex_id": rid, "string": s, "accepts": acc})
    if not rows:
        raise ValueError("No data rows")
    return rows

# ---------------- Grade ----------------
def grade(_: str | None = None):
    # HARSH weights: exact match dominates the score
    subs = {"has_file": 0.0, "header_ok": 0.0, "row_count": 0.0, "exact_values": 0.0}
    wts  = {"has_file": 0.02, "header_ok": 0.03, "row_count": 0.0, "exact_values": 0.95}
    def total() -> float: return float(sum(subs[k]*wts[k] for k in subs))

    try:
        regexes = read_regexes(DATA/"regexes.txt")
        strings = read_strings(DATA/"strings.txt")
        gt = ground_truth(regexes, strings)

        try:
            sub = read_submission()
            subs["has_file"] = 1.0
            subs["header_ok"] = 1.0
        except Exception as e:
            return _res(0.0, f"{e}", subs, wts, {})

        # Enforce exact expected row count
        if len(sub) != len(gt):
            return _res(total(), f"Row count mismatch: expected {len(gt)}, got {len(sub)}", subs, wts, {})

        # Enforce exact row order and one-to-one mapping
        for i, (a,b) in enumerate(zip(sub, gt), 1):
            if a != b:
                return _res(total(), "Values do not match expected.",
                            subs, wts, {"diff":[{"row":i,"got":a,"expected":b}]})

        subs["exact_values"] = 1.0
        return _res(total(), "Correct!", subs, wts, {})

    except Exception as e:
        return _res(0.0, f"Failed: {e}", subs, wts, {})
