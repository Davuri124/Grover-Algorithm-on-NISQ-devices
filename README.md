<div align="center">

# ⚛️ k\*(N, ε)
### The Noise-Optimal Iteration Count for Grover's Algorithm on NISQ Devices

**A closed-form theory of when to *stop* Grover, a zero-overhead noise probe, and an algorithm-native hardware score, all validated on real IBM Quantum hardware**

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Qiskit](https://img.shields.io/badge/Qiskit-Runtime-6929c4.svg)
![Hardware](https://img.shields.io/badge/hardware-ibm__fez%20%7C%20ibm__kingston-052fad.svg)
![Status](https://img.shields.io/badge/status-under%20review-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

---

## 📖 Overview

Grover's algorithm gives a provable quadratic speedup for unstructured search, and the textbook iteration count is

```
k_ideal = ⌊π√N / 4⌋
```

That count assumes **zero noise**. On NISQ hardware every iteration adds amplitude *and* noise, and past a crossover point extra iterations actively **reduce** success probability.

This repo accompanies a paper that answers a previously open question:

> **Given a search space of size N and a device with per-gate noise rate ε, how many Grover iterations should you actually run?**

### 🎯 Headline result

On `ibm_fez` with **n = 4 qubits**, running the noise-optimal **k\* = 2** instead of the textbook **k_ideal = 3** raises success probability from **0.359 → 0.462**, a **+28.4% improvement** just from choosing the right iteration count.

| | Standard (k = 3) | Noise-optimal (k\* = 2) | Δ |
|---|---:|---:|---:|
| Grover iterations | 3 | 2 | 1 fewer |
| Transpiled gates | 1265 | 843 | −422 (−33%) |
| ε_eff | 31.3% | 22.2% | −9.1% |
| **P(success)** | 0.359 | **0.462** | **+28.4%** |
| GFS | 0.373 | 0.480 | +0.107 |

---

## ✨ Contributions

### 1️⃣ Theorem 4.4: the k\* Theorem

Under a depolarizing noise model, the success probability after `k` iterations is

```
P(k, N, ε) = λᵏ · sin²((2k+1)θ) + (1 − λᵏ) / N
```

where `θ = arcsin(1/√N)`, `λ = (1−ε)^g`, and `g` is the transpiled gate count per iteration.

- **k\* satisfies a transcendental optimality equation**, derived as a formal theorem.
- **Corollary 4.5:** `k* < k_ideal` for every ε > 0.
- **Corollary 4.6:** `k* → k_ideal` as ε → 0.
- **Corollary 4.7:** a closed-form asymptotic approximation, `k* ≈ (1/2θ)·arctan(4θ / |ln λ|) − 1/2`, which lands **within 2.4%** of the exact numerical solution on real hardware (1.90 vs. 1.95).

### 2️⃣ Algorithm 1: Grover Noise Fingerprinting (GNF)

Extracts a device's per-gate error rate **ε_device using Grover itself as the probe**.

- ✅ No process tomography.
- ✅ Zero additional circuits (the probe *is* the algorithm circuit).
- ✅ A single-parameter nonlinear least-squares fit of the P(k) curve, with a 95% bootstrap CI (B = 1000).

### 3️⃣ Definition 6.1: Grover Fidelity Score (GFS)

```
GFS(B, n) = P(k*, N, ε_device) / P(k_ideal, N, 0)   ∈ (0, 1]
```

An algorithm-native hardware readiness metric. A GFS of 0.815 means the device delivers 81.5% of the ideal Grover advantage.

| Metric | Alg.-native | Interpretable | Theory-grounded | Noise-optimal |
|---|:---:|:---:|:---:|:---:|
| Quantum Volume | No | Partial | Partial | No |
| CLOPS | No | No | No | No |
| RB Fidelity | No | No | Yes | No |
| GRADE | Yes | Partial | No | No |
| **GFS (this work)** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## 🧪 Experimental Setup

| Parameter | n = 3 (N = 8) | n = 4 (N = 16) |
|---|---|---|
| Marked state | `111` | `1111` |
| k_ideal | 2 | 3 |
| k sweep | 1…6 | 1…8 |
| Shots per circuit | 4096 | 4096 |
| Seeds | 5 (`ibm_fez`), 10 (`ibm_kingston`) | same |
| Transpiled gates g | 409 | 1265 |

- **Hardware:** `ibm_fez` and `ibm_kingston` (both 156-qubit Heron r2), via `QiskitRuntimeService` with `SamplerV2` at `optimization_level=1`.
- **Scale:** 280+ circuit executions, with 95% bootstrap confidence intervals.
- **Circuits:** the oracle uses phase kickback, `Hⁿ⁻¹ · MCX · Hⁿ⁻¹`, and the diffusion operator is `H⊗ⁿ · X⊗ⁿ · MCX · X⊗ⁿ · H⊗ⁿ`.
- **Routing overhead:** heavy-hex SWAP routing inflates gate counts well beyond the theoretical ≈14 (n = 3) and ≈20 (n = 4).

---

## 📊 Results

### Theory vs. hardware: P(k) on `ibm_fez`

The Theorem 4.2 fit tracks the data closely (MSE < 0.003), including the collapse near k = 4 and the partial recovery at k = 5, 6.

| k | n = 3 hardware | n = 3 fit | n = 4 hardware | n = 4 fit |
|:-:|:--:|:--:|:--:|:--:|
| 1 | 0.653 ± 0.007 | 0.707 | 0.351 ± 0.026 | 0.344 |
| 2 | 0.725 ± 0.007 | 0.770 | **0.462 ± 0.077** | 0.461 |
| 3 | 0.253 ± 0.012 | 0.268 | 0.359 ± 0.057 | 0.353 |
| 4 | 0.056 ± 0.007 | 0.055 | 0.178 ± 0.034 | 0.178 |
| 5 | 0.382 ± 0.016 | 0.357 | 0.069 ± 0.012 | 0.072 |
| 6 | 0.577 ± 0.013 | 0.551 | 0.066 ± 0.006 | 0.058 |
| 7 | — | — | 0.079 ± 0.024 | 0.084 |
| 8 | — | — | 0.080 ± 0.016 | 0.101 |

### Algorithm 1 and k\* validation

| Backend | n | g | ε_eff | ε_device | k\*_num | k\*_asymp | GFS |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `ibm_fez` | 3 | 409 | 0.113 | 2.94×10⁻⁴ | 1.57 | 1.56 | 0.815 |
| `ibm_fez` | 4 | 1265 | 0.313 | 2.97×10⁻⁴ | 1.95 | 1.90 | 0.480 |
| `ibm_kingston` | 3 | 409 | — | 2.67×10⁻⁴ | 1.58 | — | 0.830 |
| `ibm_kingston` | 4 | 1265 | — | 1.78×10⁻⁴ | 2.20 | — | 0.626 |

**Key takeaways**

- 🔁 **Self-consistency:** on `ibm_fez`, ε_device from circuits of very different depth (g = 409 vs. 1265) agrees within the 95% CI (2.94×10⁻⁴ vs. 2.97×10⁻⁴).
- 🏆 **Device ranking:** `ibm_kingston` scores higher GFS at both scales, consistent with its lower CX error rate. Algorithm 1 recovers this ranking from circuit execution alone, **without calibration data**.
- 📈 **Scale dependence:** the GFS gap between backends widens from 1.5 points (n = 3) to 14.6 points (n = 4).
- ℹ️ At **n = 3**, k\* = 1.57 rounds to 2, which equals k_ideal, so the practical gain appears at n = 4.

---

## ⚠️ Assumptions, Caveats & Honest Notes

- **Noise model:** the theory assumes independent per-gate depolarizing noise, with Pauli-twirling giving a global n-qubit depolarizing channel.
- **Low-noise regime:** the asymptotic form needs `gε ≪ 1`. This holds for n = 3 (gε ≈ 0.12) but is **marginal for n = 4** (gε ≈ 0.38), so the n = 4 results are treated as practical validation.
- **Calibration discrepancy:** IBM's reported mean CX error for `ibm_fez` is 0.0332, versus ε_device ≈ 2.94×10⁻⁴ (about 113× lower). These measure different things: ε_device is a *per-gate average* over all 409 gates (≈180 CX plus ≈229 single-qubit gates), while IBM's figure is for a single isolated CX. Both agree on device **ranking**.
- **Scope:** results cover n = 3 and n = 4 qubits with a single marked state.
- **Future work:** n ≥ 5 qubits and multiple marked states; dropping the Pauli-twirl assumption; temporal fingerprinting of ε_device drift across calibration cycles.

---

## 🚀 Getting Started

> ⚙️ *Adjust the commands and paths below to match your actual code layout.*

```bash
git clone https://github.com/<your-username>/grover-kstar.git
cd grover-kstar
pip install -r requirements.txt   # qiskit, qiskit-ibm-runtime, numpy, scipy, matplotlib
```

```bash
# Compute k* numerically and via the closed-form approximation
python kstar.py --n 4 --gates 1265 --eps-device 2.97e-4

# Run the P(k) sweep on hardware (requires an IBM Quantum account)
python run_sweep.py --backend ibm_fez --n 4 --shots 4096 --seeds 5

# Fingerprint the device (Algorithm 1) and compute GFS
python gnf.py --results results/ibm_fez_n4.json
```

### 📁 Suggested repository structure

```
grover-kstar/
├── theory/        # P(k,N,ε), k* solver, asymptotic closed form
├── circuits/      # Oracle, diffusion, Grover circuit builders
├── hardware/      # Qiskit Runtime sweep scripts
├── gnf/           # Algorithm 1 (NLS fit + bootstrap CI)
├── gfs/           # Grover Fidelity Score
├── results/       # Raw counts, fitted parameters
├── figures/       # P(k) curves, k*(N,ε), GFS plots
└── README.md
```

## 📚 Key References

Grover (STOC 1996) · Boyer et al., *Tight Bounds on Quantum Searching* · Salas, *Noise effect on Grover algorithm* · Vrana et al., *Fault-Ignorant Quantum Search* · Leng et al., *Noise-Tolerant Grover's Algorithm* · Ganguly & Ahn, *Noise-Aware Transpiler for Amplitude Amplification* · Cross et al. (Quantum Volume) · Magesan et al. (Randomized Benchmarking) · Preskill, *Quantum Computing in the NISQ Era and Beyond*

## 📄 License

Released under the MIT License. See `LICENSE` for details.
