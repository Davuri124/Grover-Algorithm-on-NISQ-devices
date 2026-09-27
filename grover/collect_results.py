"""
collect_results.py
------------------
Polls IBM Quantum for completed jobs and saves raw measurement counts.
Run this periodically after submit_jobs.py — jobs may take hours to complete in queue.

Usage:
    python collect_results.py --account 1 --token YOUR_IBM_TOKEN
    
Run repeatedly until all jobs show status DONE.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

LOG_DIR = Path("results")


def load_submitted_jobs(account_id: int) -> list:
    """Load all submitted job records for this account."""
    log_file = LOG_DIR / f"account_{account_id}_jobs.jsonl"
    if not log_file.exists():
        print(f"No job log found for account {account_id}. Run submit_jobs.py first.")
        return []
    
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_success_count(counts: dict, marked_state: str, n: int) -> int:
    """
    Extract the count for the marked state from measurement results.
    IBM returns counts as {'000': 512, '111': 3584, ...}
    Bit ordering may be reversed — we check both orderings.
    """
    # Direct lookup
    if marked_state in counts:
        return counts[marked_state]
    
    # Try reversed bit order (IBM sometimes reverses qubit ordering)
    reversed_state = marked_state[::-1]
    if reversed_state in counts:
        return counts[reversed_state]
    
    # Not found — return 0 but warn
    print(f"  WARNING: marked state '{marked_state}' not found in counts: {list(counts.keys())[:5]}")
    return 0


def collect_results(account_id: int, token: str):
    """
    Poll IBM Quantum for all jobs from this account.
    Save completed results to results/account_{id}_raw_results.jsonl
    """
    print(f"\n{'='*60}")
    print(f"ACCOUNT {account_id} — Collecting results")
    print(f"{'='*60}")

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/413d427bf5164aaf9d0dea685f2788da:861c7c74-aec9-4205-a155-e4ea666bfc33::"
    )
    
    records = load_submitted_jobs(account_id)
    submitted_records = [r for r in records if r.get('status') == 'submitted' and r.get('job_id')]
    
    print(f"Found {len(submitted_records)} submitted jobs to check.\n")

    results_file = LOG_DIR / f"account_{account_id}_raw_results.jsonl"
    
    # Load already-collected job IDs to avoid re-fetching
    collected_ids = set()
    if results_file.exists():
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    collected_ids.add(r['job_id'])

    done_count = len(collected_ids)
    pending_count = 0
    failed_count = 0

    for record in submitted_records:
        job_id = record['job_id']
        
        if job_id in collected_ids:
            continue  # Already collected
        
        try:
            job = service.job(job_id)
            status = job.status()
            # New qiskit-ibm-runtime returns status as a string directly
            status_str = status if isinstance(status, str) else status.name
            
            if status_str == 'DONE':
                # Extract results
                result = job.result()
                pub_result = result[0]
                counts = pub_result.data.c.get_counts()
                
                marked_state = record['marked_state']
                n = record['n_qubits']
                shots = record['shots']
                success_count = extract_success_count(counts, marked_state, n)
                p_empirical = success_count / shots

                result_record = {
                    **record,
                    'job_id': job_id,
                    'collection_timestamp': datetime.utcnow().isoformat(),
                    'status': 'DONE',
                    'raw_counts': counts,
                    'success_count': success_count,
                    'p_empirical': p_empirical,
                }

                with open(results_file, 'a') as f:
                    f.write(json.dumps(result_record) + '\n')

                collected_ids.add(job_id)
                done_count += 1
                print(f"  ✓ DONE  n={n}, k={record['k_iterations']}, seed={record['seed']} "
                      f"→ P={p_empirical:.4f}  (job: {job_id[:12]}...)")

            elif status_str in ['CANCELLED', 'ERROR', 'FAILED']:
                failed_count += 1
                print(f"  ✗ {status_str}  n={record['n_qubits']}, k={record['k_iterations']}, "
                      f"seed={record['seed']}  (job: {job_id[:12]}...)")
            else:
                pending_count += 1
                print(f"  ○ {status_str:10s}  n={record['n_qubits']}, k={record['k_iterations']}, "
                      f"seed={record['seed']}  (job: {job_id[:12]}...)")

        except Exception as e:
            print(f"  ✗ ERROR fetching job {job_id[:12]}...: {str(e)}")
            failed_count += 1

        time.sleep(0.3)  # Rate limit

    print(f"\n{'='*60}")
    print(f"Account {account_id} collection summary:")
    print(f"  Collected (DONE):  {done_count}")
    print(f"  Still pending:     {pending_count}")
    print(f"  Failed/Cancelled:  {failed_count}")
    print(f"  Results saved to:  {results_file}")
    if pending_count > 0:
        print(f"\n  Run this script again later to collect remaining {pending_count} jobs.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Grover experiment results from IBM Quantum")
    parser.add_argument('--account', type=int, required=True, choices=[1, 2, 3, 4],
                        help="Account ID (1-4)")
    parser.add_argument('--token', type=str, required=True,
                        help="IBM Quantum API token for this account")
    args = parser.parse_args()

    collect_results(account_id=args.account, token=args.token)
