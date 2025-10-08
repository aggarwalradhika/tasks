#!/bin/bash
set -euo pipefail

python3 - << 'PY'
import json

def matches(pch, sch, wild):
    return pch == sch or (wild and pch == '?')

def first_index_with_budget(s, p, k, wild, banned):
    n, m = len(s), len(p)
    if m > n: return -1
    for i in range(n - m + 1):
        used = 0
        ok = True
        for j in range(m):
            if not matches(p[j], s[i+j], wild):
                if (i+j) in banned:
                    ok = False
                    break
                used += 1
                if used > k:
                    ok = False
                    break
        if ok:
            return i
    return -1

with open("/workdir/data/cases.json","r",encoding="utf-8") as f:
    cases = json.load(f)["cases"]

outs = []
for case in cases:
    s = case["s"]
    for pat in case["patterns"]:
        p      = pat["pattern"]
        k      = int(pat.get("k", 1))
        wild   = bool(pat.get("wildcards", False))
        banned = set(pat.get("banned_indices", []))
        outs.append(str(first_index_with_budget(s, p, k, wild, banned)))

with open("/workdir/sol.csv","w",encoding="utf-8") as f:
    f.write("\n".join(outs))
PY
