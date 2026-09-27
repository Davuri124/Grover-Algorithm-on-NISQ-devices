"""
grover_circuit.py
-----------------
Builds n-qubit Grover circuits for a fixed marked state and k iterations.
Records transpiled gate count g (actual, post-transpilation) for ε_eff computation.

Usage:
    from grover_circuit import build_grover_circuit, get_transpiled_gate_count
"""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np


def build_oracle(n: int, marked_state: str) -> QuantumCircuit:
    """
    Build a phase oracle for a single marked state.
    marked_state: binary string of length n, e.g. '111' for n=3.
    Returns a QuantumCircuit implementing the phase kickback oracle.
    """
    assert len(marked_state) == n, "marked_state length must equal n"
    
    qc = QuantumCircuit(n)
    
    # Flip qubits where marked_state has '0' (so all become |1>)
    for i, bit in enumerate(reversed(marked_state)):
        if bit == '0':
            qc.x(i)
    
    # Multi-controlled Z gate: flips phase of |11...1>
    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    else:
        # Use multi-controlled X with ancilla trick via phase kickback
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    
    # Unflip the qubits we flipped
    for i, bit in enumerate(reversed(marked_state)):
        if bit == '0':
            qc.x(i)
    
    return qc


def build_grover_circuit(n: int, k: int, marked_state: str) -> QuantumCircuit:
    """
    Build a Grover circuit with k iterations for an n-qubit search space.
    
    Parameters
    ----------
    n            : number of qubits
    k            : number of Grover iterations (1 <= k <= K_max)
    marked_state : binary string of length n, e.g. '111'
    
    Returns
    -------
    QuantumCircuit with measurement on all qubits.
    """
    qc = QuantumCircuit(n, n)
    
    # Step 1: Initialize uniform superposition
    qc.h(range(n))
    qc.barrier()
    
    # Step 2: Build oracle and diffusion operator
    oracle = build_oracle(n, marked_state)
    
    # Diffusion operator: 2|s><s| - I
    diffusion = QuantumCircuit(n)
    diffusion.h(range(n))
    diffusion.x(range(n))
    diffusion.h(n - 1)
    diffusion.mcx(list(range(n - 1)), n - 1)
    diffusion.h(n - 1)
    diffusion.x(range(n))
    diffusion.h(range(n))
    
    # Step 3: Apply k Grover iterations
    for iteration in range(k):
        qc.compose(oracle, inplace=True)
        qc.barrier()
        qc.compose(diffusion, inplace=True)
        qc.barrier()
    
    # Step 4: Measure
    qc.measure(range(n), range(n))
    
    return qc


def get_transpiled_gate_count(qc: QuantumCircuit, backend) -> dict:
    """
    Transpile circuit to backend's native gate set and return gate counts.
    This gives the ACTUAL g used in ε_eff = 1 - (1-ε)^g.
    
    Returns dict with:
        - total_gates     : total gate count post-transpilation
        - cx_count        : number of CNOT/CX gates (dominant noise source)
        - depth           : circuit depth
        - transpiled_qc   : the transpiled circuit object
    """
    transpiled = transpile(qc, backend=backend, optimization_level=1)
    
    gate_counts = transpiled.count_ops()
    cx_count = gate_counts.get('cx', 0) + gate_counts.get('ecr', 0) + gate_counts.get('cz', 0)
    total_gates = sum(gate_counts.values()) - gate_counts.get('measure', 0) - gate_counts.get('barrier', 0)
    
    return {
        'total_gates': total_gates,
        'cx_count': cx_count,
        'depth': transpiled.depth(),
        'gate_counts_dict': dict(gate_counts),
        'transpiled_qc': transpiled
    }


def ideal_success_probability(k: int, N: int) -> float:
    """
    Noiseless Grover success probability after k iterations.
    P_ideal(k, N) = sin^2((2k+1) * theta) where theta = arcsin(1/sqrt(N)).
    Used as denominator of GFS.
    """
    theta = np.arcsin(1.0 / np.sqrt(N))
    return np.sin((2 * k + 1) * theta) ** 2


def k_ideal(N: int) -> int:
    """
    Standard (noiseless) optimal Grover iteration count.
    k_ideal = floor(pi / (4 * theta))
    """
    theta = np.arcsin(1.0 / np.sqrt(N))
    return int(np.floor(np.pi / (4 * theta)))


if __name__ == "__main__":
    # Sanity check: build and print a 3-qubit Grover circuit with k=2
    qc = build_grover_circuit(n=3, k=2, marked_state='111')
    print("3-qubit Grover circuit, k=2:")
    print(qc)
    print(f"\nk_ideal for N=8:  {k_ideal(8)}")
    print(f"k_ideal for N=16: {k_ideal(16)}")
    print(f"\nIdeal P at k=2, N=8:  {ideal_success_probability(2, 8):.4f}")
    print(f"Ideal P at k=3, N=16: {ideal_success_probability(3, 16):.4f}")
