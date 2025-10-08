#!/usr/bin/env bash
set -euo pipefail

python3 - << 'PY'
from pathlib import Path
import csv

DATA = Path("/workdir/data")
OUT  = Path("/workdir/sol.csv")

# ---------- Same tokenizer / parser as grader ----------
def tokenize(s: str) -> list[str]:
    if any(ch.isspace() for ch in s): raise ValueError("ws")
    ok = set("ab()|*")
    for ch in s:
        if ch not in ok: raise ValueError("char")
    return list(s)

def insert_concat(ts):
    out=[]
    for i,t in enumerate(ts):
        out.append(t)
        if i+1<len(ts):
            t1,t2=ts[i],ts[i+1]
            if (t1 in "ab)" or t1=="*") and (t2 in "ab("):
                out.append(".")
    return out

def to_postfix(ts):
    prec={"*":3,".":2,"|":1}
    out=[]; st=[]
    for t in ts:
        if t in ("a","b"): out.append(t)
        elif t=="(": st.append(t)
        elif t==")":
            while st and st[-1]!="(": out.append(st.pop())
            if not st: raise ValueError(")")
            st.pop()
        elif t=="*": out.append(t)
        elif t in (".","|"):
            while st and st[-1]!="(" and prec[st[-1]]>=prec[t]:
                out.append(st.pop())
            st.append(t)
        else: raise ValueError("tok")
    while st:
        x=st.pop()
        if x=="(": raise ValueError("(")
        out.append(x)
    return out

# ---------- Thompson NFA ----------
class NFA:
    def __init__(self,s,a,trans,eps):
        self.start=s; self.accept=a; self.trans=trans; self.eps=eps

def nfa_symbol(c,ns):
    s=ns(); t=ns()
    return NFA(s,t,{s:{c:{t}}},{})

def nfa_concat(a,b):
    a.eps.setdefault(a.accept,set()).add(b.start)
    trans={**a.trans,**b.trans}; eps={**a.eps,**b.eps}
    return NFA(a.start,b.accept,trans,eps)

def nfa_union(a,b,ns):
    s=ns(); t=ns()
    trans={}; eps={}
    for d in (a.trans,b.trans):
        for k,m in d.items():
            for c,vs in m.items():
                trans.setdefault(k,{}).setdefault(c,set()).update(vs)
    for d in (a.eps,b.eps):
        for k,vs in d.items():
            eps.setdefault(k,set()).update(vs)
    eps.setdefault(s,set()).update([a.start,b.start])
    eps.setdefault(a.accept,set()).add(t)
    eps.setdefault(b.accept,set()).add(t)
    return NFA(s,t,trans,eps)

def nfa_star(a,ns):
    s=ns(); t=ns()
    trans={}; eps={}
    for k,m in a.trans.items():
        for c,vs in m.items():
            trans.setdefault(k,{}).setdefault(c,set()).update(vs)
    for k,vs in a.eps.items(): eps.setdefault(k,set()).update(vs)
    eps.setdefault(s,set()).update([a.start,t])
    eps.setdefault(a.accept,set()).update([a.start,t])
    return NFA(s,t,trans,eps)

def compile_nfa(pf):
    ns=iter(range(1_000_000)).__next__
    st=[]
    for t in pf:
        if t in ("a","b"): st.append(nfa_symbol(t,ns))
        elif t==".": b=st.pop(); a=st.pop(); st.append(nfa_concat(a,b))
        elif t=="|": b=st.pop(); a=st.pop(); st.append(nfa_union(a,b,ns))
        elif t=="*": a=st.pop(); st.append(nfa_star(a,ns))
        else: raise ValueError("pf")
    if len(st)!=1: raise ValueError("stack")
    return st[0]

# epsilon-closure, move
def eps_closure(nfa, S):
    stack=list(S); seen=set(S)
    while stack:
        s=stack.pop()
        for t in nfa.eps.get(s, ()):
            if t not in seen:
                seen.add(t); stack.append(t)
    return seen

def move(nfa, S, ch):
    nxt=set()
    for s in S:
        for t in nfa.trans.get(s, {}).get(ch, ()):
            nxt.add(t)
    return nxt

# ---------- Subset construction ----------
def nfa_to_dfa(nfa):
    alpha=["a","b"]
    start=frozenset(eps_closure(nfa,{nfa.start}))
    dstates=[start]; dtrans={}; accept=set()
    seen={start}
    i=0
    while i<len(dstates):
        S=dstates[i]
        if nfa.accept in S: accept.add(S)
        for c in alpha:
            U = eps_closure(nfa, move(nfa, S, c))
            FU = frozenset(U)
            dtrans.setdefault(S,{})[c]=FU
            if FU not in seen:
                seen.add(FU); dstates.append(FU)
        i+=1
    return start, accept, dtrans

def dfa_accepts(start, accept, dtrans, s):
    cur=start
    for ch in s:
        cur=dtrans.get(cur,{}).get(ch,frozenset())
    return cur in accept

# ---------- IO ----------
def read_regexes(p):
    valid=[]
    for line in p.read_text(encoding="utf-8").splitlines():
        rx=line.strip()
        if not rx: continue
        try:
            pf=to_postfix(insert_concat(tokenize(rx)))
            _=compile_nfa(pf)  # validate
            valid.append(rx)
        except Exception:
            # skip malformed
            pass
    return valid

def read_strings(p):
    arr=[]
    for line in p.read_text(encoding="utf-8").splitlines():
        t=line.strip()
        arr.append("" if t=="ε" else t)
    return arr

# ---------- main ----------
regexes = read_regexes(DATA/"regexes.txt")
strings = read_strings(DATA/"strings.txt")

rows=[]
for rid, rx in enumerate(regexes, start=1):
    pf=to_postfix(insert_concat(tokenize(rx)))
    nfa=compile_nfa(pf)
    start,acc,tran = nfa_to_dfa(nfa)
    for s in strings:
        ok = dfa_accepts(start, acc, tran, s)
        rows.append((str(rid), ("ε" if s=="" else s), "1" if ok else "0"))

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["regex_id","string","accepts"])
    w.writerows(rows)
PY