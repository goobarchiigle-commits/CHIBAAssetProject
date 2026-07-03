#!/usr/bin/env python3
"""Daily execution constraint analysis — read-only log parser."""
import re, sys
from collections import defaultdict
from pathlib import Path

# Auto-detect log file: prefer live_latest.log, then dryrun_latest.log, then morning_signal.log
CANDIDATES = [
    r"logs\live_latest.log",
    r"logs\dryrun_latest.log",
    r"logs\morning_signal.log",
]
log_path = None
for c in CANDIDATES:
    if Path(c).exists():
        log_path = c
        break

if len(sys.argv) > 1:
    log_path = sys.argv[1]

if not log_path or not Path(log_path).exists():
    print(f"No log file found. Tried: {CANDIDATES}")
    print("Usage: python scripts/analyze_log.py [path/to/log]")
    sys.exit(1)

print(f"Parsing: {log_path}\n")

pat_skip  = re.compile(r'(\d{4}-\d{2}-\d{2})[\sT].*?\[skip\]\s+(\S+)\s+reason=alloc_cap\s+required=([\d,]+)\s+cap=([\d,]+)\s+rank=(\d+)\s+score=([\d.]+)')
pat_entry = re.compile(r'(\d{4}-\d{2}-\d{2})[\sT].*?\[entry\]\s+(\S+)\s+reason=trend_follow\s+required=([\d,None]+)\s+rank=(\d+)\s+score=([\d.]+)')
pat_tf    = re.compile(r'(\d{4}-\d{2}-\d{2})[\sT].*?\[trend_follow\].*?candidates=(\d+).*?fallback_count=(\d+)')

dates = defaultdict(lambda: {'skips':[], 'entries':[], 'tf_cands':0, 'fallbacks':0})

with open(log_path, encoding='utf-8', errors='replace') as f:
    for line in f:
        m = pat_skip.search(line)
        if m:
            d,sym,req,cap,rank,score = m.groups()
            dates[d]['skips'].append({'sym':sym,'req':int(req.replace(',','')),'cap':int(cap.replace(',','')),'rank':int(rank),'score':float(score)})
            continue
        m = pat_entry.search(line)
        if m:
            d,sym,req,rank,score = m.groups()
            dates[d]['entries'].append({'sym':sym,'req':None if req=='None' else int(req.replace(',','')),'rank':int(rank),'score':float(score)})
            continue
        m = pat_tf.search(line)
        if m:
            d = m.group(1)
            dates[d]['tf_cands'] = max(dates[d]['tf_cands'], int(m.group(2)))
            dates[d]['fallbacks'] += int(m.group(3))

if not any(v['skips'] or v['entries'] for v in dates.values()):
    print("No [skip]/[entry] lines matched. Log may not contain the new log format yet.")
    print("The [skip]/[entry] logs are only produced after today's Task Scheduler runs (08:41/08:42).")
    sys.exit(0)

print(f"{'date':<12} {'entries':>7} {'skips':>5} {'r1_entry':>8} {'r1_skip':>7} {'avg_req':>12} {'max_req':>12} {'req/cap':>7} {'fb':>5}")
print('-'*82)
for d in sorted(dates):
    v = dates[d]
    sk, en = v['skips'], v['entries']
    reqs = [s['req'] for s in sk] + [e['req'] for e in en if e['req']]
    ratios = [s['req']/s['cap'] for s in sk if s['cap']]
    print(f"{d:<12} {len(en):>7} {len(sk):>5} "
          f"{sum(1 for e in en if e['rank']==1):>8} "
          f"{sum(1 for s in sk if s['rank']==1):>7} "
          f"{int(sum(reqs)/len(reqs)) if reqs else 0:>12,} "
          f"{max(reqs) if reqs else 0:>12,} "
          f"{sum(ratios)/len(ratios) if ratios else 0:>7.2f} "
          f"{v['fallbacks']:>5}")

print("\n--- rank別約定率 ---")
all_ranks = sorted({s['rank'] for v in dates.values() for s in v['skips']} |
                   {e['rank'] for v in dates.values() for e in v['entries']})
for r in all_ranks:
    ec = sum(1 for v in dates.values() for e in v['entries'] if e['rank']==r)
    sc = sum(1 for v in dates.values() for s in v['skips'] if s['rank']==r)
    t  = ec + sc
    print(f"  rank={r}: entry={ec} skip={sc} fill_rate={f'{ec/t:.0%}' if t else 'N/A'}")
