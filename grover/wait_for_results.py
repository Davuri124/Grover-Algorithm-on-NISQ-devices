"""
wait_for_results.py
-------------------
Polls for results every 10 minutes until all extended jobs are complete.
Usage: python wait_for_results.py --token YOUR_TOKEN
"""
import argparse
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
parser.add_argument('--interval', type=int, default=600, help='Poll interval in seconds (default: 600 = 10 min)')
args = parser.parse_args()

LOG_DIR = Path("results")

def count_pending():
    """Count total pending jobs across all extended log files."""
    total_pending = 0
    total_done = 0
    
    for log_file in LOG_DIR.glob("extended_*_jobs.jsonl"):
        results_file = log_file.parent / log_file.name.replace('_jobs.jsonl', '_results.jsonl')
        
        collected_ids = set()
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        collected_ids.add(r['job_id'])
                        total_done += 1
        
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get('status') == 'submitted' and r.get('job_id') not in collected_ids:
                        total_pending += 1
    
    return total_done, total_pending

print(f"\n{'='*50}")
print(f"Auto-polling for results every {args.interval//60} minutes")
print(f"Press Ctrl+C to stop")
print(f"{'='*50}\n")

poll = 0
while True:
    poll += 1
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[Poll #{poll} at {now}] Running collect...")
    print("-" * 40)
    
    subprocess.run(
        ["python", "submit_extended.py", "--mode", "collect", "--token", args.token],
        check=False
    )
    
    done, pending = count_pending()
    print(f"\n  → Total collected: {done} | Still pending: {pending}")
    
    if pending == 0:
        print(f"\n{'='*50}")
        print("ALL JOBS COMPLETE!")
        print("Running final extended analysis...")
        print(f"{'='*50}\n")
        subprocess.run(["python", "analysis.py", "--extended"], check=False)
        break
    
    print(f"\n  Waiting {args.interval//60} minutes before next poll...")
    print(f"  (Next check at {datetime.fromtimestamp(time.time() + args.interval).strftime('%H:%M:%S')})")
    time.sleep(args.interval)
