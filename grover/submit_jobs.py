"""
submit_jobs.py
--------------
Submits all Grover experiment jobs to IBM Quantum.
Logs every job with full metadata before submission.

Run this once per account. Change ACCOUNT_ID and IBM_TOKEN for each account.

Usage:
    python submit_jobs.py --account 1 --token YOUR_IBM_TOKEN
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from grover_circuit import build_grover_circuit, get_transpiled_gate_count

# ─── Experiment parameters (fixed across all accounts) ───────────────────────

EXPERIMENT_CONFIG = {
    3: {
        'marked_state': '111',
        'N': 8,
        'k_range': list(range(1, 7)),   # k = 1..6 (K_max = 2 * k_ideal = 4, extended to 6)
        'seeds': [0, 1, 2, 3, 4],
        'shots': 4096,
    },
    4: {
        'marked_state': '1111',
        'N': 16,
        'k_range': list(range(1, 9)),   # k = 1..8 (K_max = 2 * k_ideal = 6, extended to 8)
        'seeds': [0, 1, 2, 3, 4],
        'shots': 4096,
    }
}

# Account-to-job assignment (distribute across 4 accounts)
ACCOUNT_JOBS = {
    1: {'n_values': [3], 'seeds': [0, 1, 2]},        # 18 jobs
    2: {'n_values': [3], 'seeds': [3, 4]},             # 12 jobs  (+ account 2 also runs n=4 seeds 0,1)
    3: {'n_values': [4], 'seeds': [0, 1, 2]},          # 24 jobs
    4: {'n_values': [4], 'seeds': [3, 4]},             # 16 jobs
}

# ─── Account 2 special: also runs n=4, seeds 0,1 (override above) ────────────
ACCOUNT_JOBS_FULL = {
    1: [{'n': 3, 'seeds': [0, 1, 2]}],
    2: [{'n': 3, 'seeds': [3, 4]}, {'n': 4, 'seeds': [0, 1]}],
    3: [{'n': 4, 'seeds': [2, 3, 4]}],
    4: None,  # Reserve account — only use if another crashes
}

LOG_DIR = Path("results")
LOG_DIR.mkdir(exist_ok=True)


def get_service(token: str) -> QiskitRuntimeService:
    """Connect to IBM Quantum with provided token and instance."""
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/413d427bf5164aaf9d0dea685f2788da:861c7c74-aec9-4205-a155-e4ea666bfc33::"
    )
    return service


def get_backend(service: QiskitRuntimeService):
    """Get least-busy available real backend with >= 4 qubits."""
    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=4
    )
    print(f"  → Assigned backend: {backend.name} ({backend.num_qubits} qubits)")
    return backend


def submit_single_job(sampler, transpiled_qc, shots: int) -> str:
    """Submit one job and return job ID."""
    job = sampler.run([transpiled_qc], shots=shots)
    return job.job_id()


def run_account(account_id: int, token: str):
    """
    Submit all jobs assigned to this account.
    Logs every job to results/account_{id}_jobs.jsonl before submission.
    """
    print(f"\n{'='*60}")
    print(f"ACCOUNT {account_id} — Starting job submission")
    print(f"{'='*60}")

    service = get_service(token)
    backend = get_backend(service)
    sampler = Sampler(mode=backend)

    log_file = LOG_DIR / f"account_{account_id}_jobs.jsonl"
    results_file = LOG_DIR / f"account_{account_id}_raw_results.jsonl"

    job_assignments = ACCOUNT_JOBS_FULL.get(account_id)
    if job_assignments is None:
        print(f"Account {account_id} is the RESERVE account. Only use if another crashes.")
        return

    submitted = []

    for assignment in job_assignments:
        n = assignment['n']
        seeds = assignment['seeds']
        config = EXPERIMENT_CONFIG[n]

        for k in config['k_range']:
            for seed in seeds:
                print(f"\n  Preparing: n={n}, k={k}, seed={seed}")

                # Build and transpile circuit
                qc = build_grover_circuit(
                    n=n,
                    k=k,
                    marked_state=config['marked_state']
                )

                gate_info = get_transpiled_gate_count(qc, backend)
                transpiled_qc = gate_info['transpiled_qc']

                # Build complete metadata record BEFORE submission
                record = {
                    'account_id': account_id,
                    'backend_name': backend.name,
                    'backend_num_qubits': backend.num_qubits,
                    'submission_timestamp': datetime.utcnow().isoformat(),
                    'n_qubits': n,
                    'N': config['N'],
                    'k_iterations': k,
                    'seed': seed,
                    'marked_state': config['marked_state'],
                    'shots': config['shots'],
                    'transpiled_total_gates': gate_info['total_gates'],
                    'transpiled_cx_count': gate_info['cx_count'],
                    'transpiled_depth': gate_info['depth'],
                    'gate_counts_dict': gate_info['gate_counts_dict'],
                    'job_id': None,  # filled after submission
                    'status': 'pending'
                }

                # Submit
                try:
                    job_id = submit_single_job(sampler, transpiled_qc, config['shots'])
                    record['job_id'] = job_id
                    record['status'] = 'submitted'
                    print(f"    ✓ Submitted — job_id: {job_id}")
                except Exception as e:
                    record['status'] = f'failed: {str(e)}'
                    print(f"    ✗ Failed — {str(e)}")

                # Log immediately after every submission attempt
                with open(log_file, 'a') as f:
                    f.write(json.dumps(record) + '\n')

                submitted.append(record)

                # Small delay to avoid flooding the API
                time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Account {account_id} complete.")
    print(f"  Jobs submitted: {sum(1 for r in submitted if r['status'] == 'submitted')}")
    print(f"  Jobs failed:    {sum(1 for r in submitted if 'failed' in r['status'])}")
    print(f"  Log saved to:   {log_file}")
    print(f"{'='*60}")

    return submitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Grover experiment jobs to IBM Quantum")
    parser.add_argument('--account', type=int, required=True, choices=[1, 2, 3, 4],
                        help="Account ID (1-4)")
    parser.add_argument('--token', type=str, required=True,
                        help="IBM Quantum API token for this account")
    args = parser.parse_args()

    run_account(account_id=args.account, token=args.token)
