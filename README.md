# Noise-Optimal Grover

## Overview

**Noise-Optimal Grover** studies how noise affects the optimal number of iterations in Grover's search algorithm on Noisy Intermediate-Scale Quantum (NISQ) devices.

In the ideal noiseless setting, Grover's algorithm uses the textbook iteration count:

k_ideal = floor(π√N / 4)

However, on real quantum hardware, every gate operation introduces noise. As the number of Grover iterations increases, the accumulated noise can eventually outweigh the amplitude-amplification benefit.

This project investigates the fundamental question:

> **What is the optimal number of Grover iterations when the quantum hardware is noisy?**

The work derives a noise-dependent optimal iteration count, introduces a hardware-probing method based on Grover's algorithm itself, and proposes an algorithm-native hardware readiness metric.

---

## Key Contributions

### 1. Noise-Optimal Grover Iteration Count

The project derives:

**k*(N, ε)**

the noise-optimal number of Grover iterations as a function of the search-space size `N` and the per-gate noise rate `ε`.

A formal theorem establishes that:

```text
k* < k_ideal
