"""
submit_extended.py
------------------
Extended submission for four additional experiments:
  1. 10+ seeds per (n, k) for tighter CI on ε_device
  2. Multi-backend: targets ibm_marrakesh and ibm_kingston specifically
  3. Temporal fingerprinting: same circuits run after IBM recalibration
  4. IBM calibration data fetch for direct comparison with Algorithm 1

Usage:
    # Run 10 seeds on whatever backend is least busy (for tighter CI)
    python submit_extended.py --mode seeds --token YOUR_TOKEN

    # Target a specific backend for multi-backend fingerprinting
    python submit_extended.py --mode backend --target ibm_marrakesh --token YOUR_TOKEN
    python submit_extended.py --mode backend --target ibm_kingston --token YOUR_TOKEN

    # Temporal re-run (same as seeds mode, but saves with timestamp label)
    python submit_extended.py --mode temporal --token YOUR_TOKEN

    # Fetch IBM calibration data (no circuits submitted)
    python submit_extended.py --mode calibration --token YOUR_TOKEN
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from grover_circuit import build_grover_circuit, get_transpiled_gate_count

CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/1275eb7ff58d4e5286f1034a061214a4:5580881d-74f8-4fe5-bc54-9539f010aa4c::"

LOG_DIR = Path("results")
LOG_DIR.mkdir(exist_ok=True)

# ─── Extended experiment config ───────────────────────────────────────────────

EXPERIMENT_CONFIG = {
    3: {
        'marked_state': '111',
        'N': 8,
        'k_range': list(range(1, 7)),
        'shots': 4096,
    },
    4: {
        'marked_state': '1111',
        'N': 16,
        'k_range': list(range(1, 9)),
        'shots': 4096,
    }
}

# 10 seeds, indices 5-14 (0-4 already done in original run)
EXTENDED_SEEDS = list(range(5, 15))  # seeds 5..14


def get_service(token: str) -> QiskitRuntimeService:
    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=CRN
    )


def get_backend_by_name(service, name: str):
    """Get a specific named backend."""
    backend = service.backend(name)
    print(f"  → Targeting backend: {backend.name} ({backend.num_qubits} qubits)")
    # Check it's operational
    if not backend.status().operational:
        print(f"  WARNING: {name} is not operational! Falling back to least_busy.")
        return get_least_busy(service)
    return backend


def get_least_busy(service):
    """Get least-busy real backend."""
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=4)
    print(f"  → Least-busy backend: {backend.name} ({backend.num_qubits} qubits)")
    return backend


def submit_batch(service, backend, n_values, seeds, mode_label, log_suffix):
    """Submit a batch of jobs and log results."""
    sampler = Sampler(mode=backend)
    log_file = LOG_DIR / f"extended_{mode_label}_{log_suffix}_jobs.jsonl"
    submitted = []

    for n in n_values:
        config = EXPERIMENT_CONFIG[n]
        for k in config['k_range']:
            for seed in seeds:
                print(f"  Preparing: n={n}, k={k}, seed={seed} [{mode_label}]")

                qc = build_grover_circuit(n=n, k=k, marked_state=config['marked_state'])
                gate_info = get_transpiled_gate_count(qc, backend)
                transpiled_qc = gate_info['transpiled_qc']

                record = {
                    'mode': mode_label,
                    'backend_name': backend.name,
                    'backend_num_qubits': backend.num_qubits,
                    'submission_timestamp': datetime.now(timezone.utc).isoformat(),
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
                    'job_id': None,
                    'status': 'pending'
                }

                try:
                    job = sampler.run([transpiled_qc], shots=config['shots'])
                    record['job_id'] = job.job_id()
                    record['status'] = 'submitted'
                    print(f"    ✓ Submitted — job_id: {job.job_id()}")
                except Exception as e:
                    record['status'] = f'failed: {str(e)}'
                    print(f"    ✗ Failed — {str(e)}")

                with open(log_file, 'a') as f:
                    f.write(json.dumps(record) + '\n')

                submitted.append(record)
                time.sleep(0.5)

    done = sum(1 for r in submitted if r['status'] == 'submitted')
    failed = sum(1 for r in submitted if 'failed' in r['status'])
    print(f"\n  Jobs submitted: {done} | Failed: {failed} | Log: {log_file}")
    return submitted


def fetch_calibration_data(service, token: str):
    """
    Fetch IBM published calibration data for all available backends.
    Saves CNOT error rates and T1/T2 times for comparison with Algorithm 1.
    """
    print("\n[Calibration] Fetching IBM backend calibration data...")
    cal_file = LOG_DIR / "ibm_calibration_data.json"
    calibration = {}

    backend_names = ['ibm_fez', 'ibm_marrakesh', 'ibm_kingston']

    for name in backend_names:
        try:
            backend = service.backend(name)
            props = backend.properties()

            if props is None:
                print(f"  {name}: No properties available")
                calibration[name] = {'error': 'no properties'}
                continue

            # Extract gate errors
            cx_errors = []
            single_qubit_errors = []
            t1_times = []
            t2_times = []
            readout_errors = []

            for gate in props.gates:
                if gate.gate in ['cx', 'ecr', 'cz']:
                    for param in gate.parameters:
                        if param.name == 'gate_error':
                            cx_errors.append({
                                'qubits': gate.qubits,
                                'gate': gate.gate,
                                'error': param.value,
                                'date': str(param.date) if param.date else None
                            })
                elif gate.gate in ['x', 'sx', 'rz']:
                    for param in gate.parameters:
                        if param.name == 'gate_error':
                            single_qubit_errors.append(param.value)

            for qubit_idx in range(min(10, backend.num_qubits)):
                try:
                    t1 = props.t1(qubit_idx)
                    t2 = props.t2(qubit_idx)
                    ro = props.readout_error(qubit_idx)
                    if t1: t1_times.append(t1)
                    if t2: t2_times.append(t2)
                    if ro: readout_errors.append(ro)
                except:
                    pass

            import numpy as np
            cx_error_values = [e['error'] for e in cx_errors]

            summary = {
                'backend': name,
                'num_qubits': backend.num_qubits,
                'fetch_timestamp': datetime.now(timezone.utc).isoformat(),
                'cx_gate_errors': cx_errors[:20],  # First 20 CX pairs
                'cx_error_mean': float(np.mean(cx_error_values)) if cx_error_values else None,
                'cx_error_min': float(np.min(cx_error_values)) if cx_error_values else None,
                'cx_error_max': float(np.max(cx_error_values)) if cx_error_values else None,
                'single_qubit_error_mean': float(np.mean(single_qubit_errors)) if single_qubit_errors else None,
                't1_mean_us': float(np.mean(t1_times)) * 1e6 if t1_times else None,
                't2_mean_us': float(np.mean(t2_times)) * 1e6 if t2_times else None,
                'readout_error_mean': float(np.mean(readout_errors)) if readout_errors else None,
            }

            calibration[name] = summary
            print(f"  {name}: CX error mean={summary['cx_error_mean']:.6f}, "
                  f"range=[{summary['cx_error_min']:.6f}, {summary['cx_error_max']:.6f}]")

        except Exception as e:
            print(f"  {name}: Error fetching — {str(e)}")
            calibration[name] = {'error': str(e)}

    with open(cal_file, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"\n  Calibration data saved to: {cal_file}")
    return calibration


def collect_extended_results(token: str):
    """
    Collect results from all extended job logs.
    Run this after jobs complete.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=CRN)

    log_files = list(LOG_DIR.glob("extended_*_jobs.jsonl"))
    print(f"\nFound {len(log_files)} extended job log files.")

    for log_file in log_files:
        results_file = log_file.parent / log_file.name.replace('_jobs.jsonl', '_results.jsonl')
        collected_ids = set()

        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        collected_ids.add(json.loads(line)['job_id'])

        with open(log_file) as f:
            records = [json.loads(l) for l in f if l.strip()]

        pending = [r for r in records if r.get('status') == 'submitted'
                   and r.get('job_id') and r['job_id'] not in collected_ids]

        print(f"\n{log_file.name}: {len(pending)} jobs to collect")
        done = pending_count = failed = 0

        for record in pending:
            job_id = record['job_id']
            try:
                job = service.job(job_id)
                status_raw = job.status()
                status = status_raw if isinstance(status_raw, str) else status_raw.name

                if status == 'DONE':
                    result = job.result()
                    counts = result[0].data.c.get_counts()
                    marked = record['marked_state']
                    success_count = counts.get(marked, counts.get(marked[::-1], 0))
                    p_emp = success_count / record['shots']

                    result_record = {
                        **record,
                        'collection_timestamp': datetime.now(timezone.utc).isoformat(),
                        'status': 'DONE',
                        'raw_counts': counts,
                        'success_count': success_count,
                        'p_empirical': p_emp,
                    }
                    with open(results_file, 'a') as f:
                        f.write(json.dumps(result_record) + '\n')
                    done += 1
                    print(f"  ✓ n={record['n_qubits']}, k={record['k_iterations']}, "
                          f"seed={record['seed']} → P={p_emp:.4f}")
                elif status in ['CANCELLED', 'ERROR', 'FAILED']:
                    failed += 1
                    print(f"  ✗ {status}: n={record['n_qubits']}, k={record['k_iterations']}")
                else:
                    pending_count += 1
                    print(f"  ○ {status}: n={record['n_qubits']}, k={record['k_iterations']}")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                failed += 1
            time.sleep(0.3)

        print(f"  Done: {done} | Pending: {pending_count} | Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['seeds', 'backend', 'temporal', 'calibration', 'collect'],
                        help='seeds: 10-seed rerun | backend: target specific backend | '
                             'temporal: rerun for drift | calibration: fetch IBM data | '
                             'collect: gather results')
    parser.add_argument('--token', required=True, help='IBM Quantum API token')
    parser.add_argument('--target', default=None,
                        help='Backend name for --mode backend (e.g. ibm_marrakesh)')
    args = parser.parse_args()

    service = get_service(args.token)

    if args.mode == 'calibration':
        fetch_calibration_data(service, args.token)

    elif args.mode == 'collect':
        collect_extended_results(args.token)

    elif args.mode == 'seeds':
        print("\n[Mode: seeds] Running 10 additional seeds on least-busy backend")
        backend = get_least_busy(service)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        submit_batch(service, backend, n_values=[3, 4],
                     seeds=EXTENDED_SEEDS, mode_label='seeds', log_suffix=timestamp)

    elif args.mode == 'backend':
        if not args.target:
            print("ERROR: --target required for backend mode (e.g. --target ibm_marrakesh)")
            exit(1)
        print(f"\n[Mode: backend] Targeting {args.target}")
        backend = get_backend_by_name(service, args.target)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        submit_batch(service, backend, n_values=[3, 4],
                     seeds=list(range(5)),  # 5 seeds per backend
                     mode_label=f'backend_{args.target}',
                     log_suffix=timestamp)

    elif args.mode == 'temporal':
        print("\n[Mode: temporal] Re-running for temporal fingerprinting (post-recalibration)")
        backend = get_least_busy(service)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        submit_batch(service, backend, n_values=[3, 4],
                     seeds=list(range(5)),  # same 5 seeds as original
                     mode_label='temporal',
                     log_suffix=timestamp)

    print("\nDone.")
