#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3

$PYTHON - << 'PY'
import json, math, csv
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

DATA = Path("/workdir/data")
SOL  = Path("/workdir/sol.csv")

def load(name):
    with open(DATA/name, "r", encoding="utf-8") as f:
        return json.load(f)

campaigns = load("campaigns.json")          # list of dicts
impressions = load("impressions.json")      # list of records: {campaign_id, channel, impressions}
clicks = load("clicks.json")                # list of records: {campaign_id, channel, ip, ts}
conversions = load("conversions.json")      # list of records: {campaign_id, ts}
ip_meta = load("ip_metadata.json")          # optional not used by formula

# ---- Build indices ----
camp_by_id = {(c["campaign_id"]): c for c in campaigns}

# Aggregate per-channel impressions and clicks
impr_by_camp_chan = defaultdict(int)
for r in impressions:
    impr_by_camp_chan[(r["campaign_id"], r["channel"])] += int(r["impressions"])

clicks_by_camp_chan = defaultdict(int)
clicks_by_camp_min  = defaultdict(lambda: defaultdict(int))  # camp -> minute -> count
clicks_by_camp_ip   = defaultdict(Counter)                   # camp -> Counter(ip)

def to_min(ts):
    # ts like "2025-07-01T12:34:10Z"
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%dT%H:%M")

for r in clicks:
    cid = r["campaign_id"]; chan = r["channel"]; ip = r["ip"]; ts = r["ts"]
    clicks_by_camp_chan[(cid, chan)] += 1
    clicks_by_camp_min[cid][to_min(ts)] += 1
    clicks_by_camp_ip[cid][ip] += 1

conversions_by_camp = defaultdict(int)
for r in conversions:
    conversions_by_camp[r["campaign_id"]] += 1

# ---- Eligibility ----
eligible = []
for c in campaigns:
    if not c.get("active", False): 
        continue
    if float(c.get("spend_last_30d", 0)) < 500:
        continue
    total_impr = sum(impr_by_camp_chan[(c["campaign_id"], ch)] for ch in c.get("channels", []))
    if total_impr < 2000:
        continue
    eligible.append(c["campaign_id"])

# ---- Fraud features ----
rows = []
for cid in eligible:
    c = camp_by_id[cid]
    channels = c.get("channels", [])
    # 1) abnormal_click_rate
    ctrs = []
    for ch in channels:
        impr = impr_by_camp_chan[(cid, ch)]
        clk  = clicks_by_camp_chan[(cid, ch)]
        ctrs.append( (clk / impr) if impr > 0 else 0.0 )
    channel_ctr_max = max(ctrs) if ctrs else 0.0
    expected_ctr = 0.02
    avg_ctr_by_ch = c.get("average_ctr_by_channel")  # dict channel->expected
    if isinstance(avg_ctr_by_ch, dict) and len(avg_ctr_by_ch)>0:
        # if multi-channel, we are told "expected_ctr = campaigns.average_ctr_by_channel"
        # The spec is slightly ambiguous; we interpret as *mean of provided channel expectations*.
        expected_ctr = sum(avg_ctr_by_ch.values())/len(avg_ctr_by_ch)
    abnormal_click_rate = max(0.0, channel_ctr_max - expected_ctr)

    # 2) ip_aggregation_score
    ip_counter = clicks_by_camp_ip[cid]
    total_clicks = sum(ip_counter.values())
    if total_clicks == 0:
        ip_agg = 0.0
    else:
        top3 = sum(v for _,v in ip_counter.most_common(3))
        ip_agg = min(1.0, top3/total_clicks)

    # 3) rapid_fire_clicks
    mins = clicks_by_camp_min[cid]
    if mins:
        burst = sum(1 for m,cnt in mins.items() if cnt > 5)
        rapid_fire = max(0.0, min(1.0, burst/len(mins)))
    else:
        rapid_fire = 0.0

    # 4) conversion_quality_penalty
    convs = conversions_by_camp[cid]
    cr = (convs/total_clicks) if total_clicks>0 else 0.0
    ecr = float(c.get("expected_conversion_rate", 0.05))
    if ecr <= 0:
        ecr = 0.05
    penalty = ((ecr - cr)/ecr) if cr < ecr else 0.0
    if penalty < 0: penalty = 0.0
    if penalty > 1: penalty = 1.0

    fraud = 4.0*abnormal_click_rate + 3.0*ip_agg + 2.0*rapid_fire + 3.5*penalty

    rows.append({
        "account_id": c["account_id"],
        "campaign_id": c["campaign_id"],
        "campaign_name": c["campaign_name"],
        "spend_last_30d": float(c["spend_last_30d"]),
        "abnormal_click_rate": abnormal_click_rate,
        "ip_aggregation_score": ip_agg,
        "rapid_fire_clicks": rapid_fire,
        "conversion_quality_penalty": penalty,
        "fraud_score": fraud
    })

# sorting and top-5
rows.sort(key=lambda r: (-r["fraud_score"], r["account_id"], r["campaign_id"]))
rows = rows[:5]
for i,r in enumerate(rows, start=1):
    r["rank"] = i

# exact formatting
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"

with open(SOL, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["account_id","campaign_id","campaign_name","spend_last_30d",
                "abnormal_click_rate","ip_aggregation_score","rapid_fire_clicks",
                "conversion_quality_penalty","fraud_score","rank"])
    for r in rows:
        w.writerow([
            r["account_id"],
            r["campaign_id"],
            r["campaign_name"],
            f2(r["spend_last_30d"]),
            f3(r["abnormal_click_rate"]),
            f3(r["ip_aggregation_score"]),
            f3(r["rapid_fire_clicks"]),
            f3(r["conversion_quality_penalty"]),
            f3(r["fraud_score"]),
            r["rank"]
        ])
PY
echo "OK"
