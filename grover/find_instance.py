"""
find_instance.py
Run this first to find your IBM Quantum instance name.
Usage: python find_instance.py --token YOUR_TOKEN
"""
import argparse
from qiskit_ibm_runtime import QiskitRuntimeService

parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
args = parser.parse_args()

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=args.token,
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/413d427bf5164aaf9d0dea685f2788da:861c7c74-aec9-4205-a155-e4ea666bfc33::"
)

instances = service.instances()
print("\nYour available instances:")
for i, inst in enumerate(instances):
    print(f"  [{i}] {inst}")

print("\nAvailable backends:")
backends = service.backends(operational=True, simulator=False)
for b in backends:
    print(f"  {b.name} — {b.num_qubits} qubits")
