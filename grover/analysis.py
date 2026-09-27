"""
analysis.py
-----------
Core analysis pipeline. Run after all results are collected.

Performs:
1. Ideal baseline (simulator) — P_ideal(k) for n=3 and n=4
2. Empirical P(k) curves from hardware results (mean ± 95% CI)
3. ε_device extraction via Algorithm 1 (nonlinear least squares fit)
4. k* computation (numerical from transcendental equation + asymptotic formula)
5. GFS computation
6. Publication-quality plots

Usage:
    python analysis.py
    
Outputs in results/analysis/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import bootstrap
from collections import defaultdict

# Use non-interactive backend for saving figures
matplotlib.use('Agg')

LOG_DIR = Path("results")
OUT_DIR = Path("results/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Theorem 1: P(k, N, ε_eff) ───────────────────────────────────────────────

def theta(N: int) -> float:
    return np.arcsin(1.0 / np.sqrt(N))


def p_ideal(k: float, N: int) -> float:
    """Noiseless Grover: P_ideal(k,N) = sin^2((2k+1)*theta)"""
    th = theta(N)
    return np.sin((2 * k + 1) * th) ** 2


def p_noisy(k: float, N: int, eps_eff: float) -> float:
    """
    Theorem 1: P(k, N, ε_eff) = λ^k sin^2((2k+1)θ) + (1-λ^k)/N
    λ = 1 - ε_eff
    """
    lam = 1.0 - eps_eff
    th = theta(N)
    ideal_term = (lam ** k) * np.sin((2 * k + 1) * th) ** 2
    noise_floor = (1 - lam ** k) / N
    return ideal_term + noise_floor


def dp_dk(k: float, N: int, eps_eff: float) -> float:
    """
    Derivative dP/dk from Theorem 2 transcendental equation LHS.
    Used to find k* numerically.
    """
    lam = 1.0 - eps_eff
    th = theta(N)
    phi = (2 * k + 1) * th
    ln_lam = np.log(lam)  # negative
    
    term1 = ln_lam * (np.sin(phi) ** 2 - 1.0 / N)
    term2 = 2 * th * np.sin(2 * phi)
    return term1 + term2


def k_ideal_exact(N: int) -> int:
    """Standard noiseless optimal iteration count."""
    th = theta(N)
    return int(np.floor(np.pi / (4 * th)))


def k_star_numerical(N: int, eps_eff: float) -> float:
    """
    Solve Theorem 2 transcendental equation numerically.
    Returns k* as a float (round to nearest int for practical use).
    """
    k_id = k_ideal_exact(N)
    
    # k* is in (0, k_ideal) — search this interval
    # dP/dk should be positive near k=0 and negative near k=k_ideal
    try:
        k_star = brentq(dp_dk, 0.1, k_id - 0.01, args=(N, eps_eff))
    except ValueError:
        # If brentq fails (monotone in interval), k* is at boundary
        # This happens in very high noise — P is always decreasing, k*=1
        k_star = 1.0
    
    return k_star


def k_star_asymptotic(N: int, eps_eff: float) -> float:
    """
    Theorem 2(iii) asymptotic formula (noisy-dominated regime).
    Valid when g*eps >> 2/sqrt(N), i.e. |ln(lambda)| >> 2*theta.
    k*(N,ε) ≈ (1/2θ) arctan(4θ/|ln λ|) - 1/2
    """
    th = theta(N)
    lam = 1.0 - eps_eff
    ln_lam_abs = abs(np.log(lam))
    
    phi_star = np.arctan(4 * th / ln_lam_abs)
    return phi_star / (2 * th) - 0.5


# ─── Algorithm 1: Grover Noise Fingerprinting ─────────────────────────────────

def fit_eps_eff(k_values: list, p_observed: list, N: int) -> dict:
    """
    Algorithm 1, Step 3: Fit ε_eff to observed P(k) curve.
    Single free parameter: ε_eff ∈ (0, 1).
    
    Returns dict with eps_eff, residuals, fitted_curve.
    """
    k_arr = np.array(k_values)
    p_arr = np.array(p_observed)
    
    def loss(eps_eff):
        if eps_eff <= 0 or eps_eff >= 1:
            return 1e10
        p_fit = np.array([p_noisy(k, N, eps_eff) for k in k_arr])
        return np.sum((p_fit - p_arr) ** 2)
    
    result = minimize_scalar(loss, bounds=(1e-6, 0.999), method='bounded')
    eps_eff_opt = result.x
    
    p_fitted = np.array([p_noisy(k, N, eps_eff_opt) for k in k_arr])
    residuals = p_arr - p_fitted
    
    return {
        'eps_eff': eps_eff_opt,
        'loss': result.fun,
        'p_fitted': p_fitted,
        'residuals': residuals,
        'converged': result.success
    }


def extract_eps_device(eps_eff: float, g: int) -> float:
    """
    Algorithm 1, Step 4: Recover per-gate error rate from ε_eff.
    ε_device = 1 - (1 - ε_eff)^(1/g)
    """
    return 1.0 - (1.0 - eps_eff) ** (1.0 / g)


def bootstrap_ci(k_values, p_matrix, N, n_boot=1000, ci=0.95):
    """
    Algorithm 1, Step 5: Bootstrap 95% CI on ε_eff.
    p_matrix: (n_seeds, len(k_values)) array of empirical probabilities.
    """
    eps_boots = []
    n_seeds = p_matrix.shape[0]
    
    for _ in range(n_boot):
        # Resample seeds with replacement
        idx = np.random.choice(n_seeds, size=n_seeds, replace=True)
        p_boot = p_matrix[idx].mean(axis=0)
        fit = fit_eps_eff(k_values, p_boot.tolist(), N)
        eps_boots.append(fit['eps_eff'])
    
    eps_boots = np.array(eps_boots)
    alpha = (1 - ci) / 2
    lower = np.percentile(eps_boots, 100 * alpha)
    upper = np.percentile(eps_boots, 100 * (1 - alpha))
    
    return {
        'median': np.median(eps_boots),
        'lower': lower,
        'upper': upper,
        'std': np.std(eps_boots)
    }


# ─── Definition 1: GFS ────────────────────────────────────────────────────────

def compute_gfs(N: int, eps_eff: float) -> dict:
    """
    Definition 1: GFS(B, n) = P(k*, N, ε_device) / P(k_ideal, N, 0)
    
    Returns dict with gfs, k_star, k_ideal, p_at_k_star, p_ideal_peak.
    """
    k_id = k_ideal_exact(N)
    p_ideal_peak = p_ideal(k_id, N)
    
    k_star = k_star_numerical(N, eps_eff)
    k_star_int = max(1, round(k_star))
    p_at_k_star = p_noisy(k_star_int, N, eps_eff)
    
    gfs = p_at_k_star / p_ideal_peak
    
    return {
        'gfs': gfs,
        'k_star_continuous': k_star,
        'k_star_int': k_star_int,
        'k_ideal': k_id,
        'p_at_k_star': p_at_k_star,
        'p_ideal_peak': p_ideal_peak,
        'k_star_asymptotic': k_star_asymptotic(N, eps_eff)
    }


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_all_results() -> list:
    """Load all collected results across all accounts."""
    all_results = []
    for account_id in [1, 2, 3, 4]:
        results_file = LOG_DIR / f"account_{account_id}_raw_results.jsonl"
        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_results.append(json.loads(line))
    print(f"Loaded {len(all_results)} total results across all accounts.")
    return all_results


def group_by_backend_n(results: list) -> dict:
    """
    Group results by (backend_name, n_qubits).
    Returns dict: {(backend, n): [(k, seed, p_empirical, g), ...]}
    """
    groups = defaultdict(list)
    for r in results:
        key = (r['backend_name'], r['n_qubits'])
        groups[key].append({
            'k': r['k_iterations'],
            'seed': r['seed'],
            'p': r['p_empirical'],
            'g': r['transpiled_total_gates'],
            'cx_count': r['transpiled_cx_count'],
            'N': r['N']
        })
    return dict(groups)


# ─── Main Analysis Pipeline ───────────────────────────────────────────────────

def run_analysis():
    print("\n" + "="*60)
    print("ANALYSIS PIPELINE — Grover NISQ k* Paper")
    print("="*60)

    # ── Step 0: Ideal baseline (no IBM needed) ──
    print("\n[Step 0] Computing ideal (noiseless) baselines...")
    ideal_baselines = {}
    for n, N in [(3, 8), (4, 16)]:
        k_vals = list(range(1, 9))
        ideal_baselines[n] = {
            'k_values': k_vals,
            'p_ideal': [p_ideal(k, N) for k in k_vals],
            'k_ideal': k_ideal_exact(N),
            'p_ideal_peak': p_ideal(k_ideal_exact(N), N)
        }
        print(f"  n={n}: k_ideal={ideal_baselines[n]['k_ideal']}, "
              f"P_ideal_peak={ideal_baselines[n]['p_ideal_peak']:.4f}")

    # ── Step 1: Load hardware results ──
    print("\n[Step 1] Loading hardware results...")
    results = load_all_results()
    
    if not results:
        print("\n  No results found yet. Run collect_results.py first.")
        print("  Generating ideal-only plots for verification...")
        plot_ideal_curves(ideal_baselines)
        return

    groups = group_by_backend_n(results)
    print(f"  Found {len(groups)} (backend, n) combinations:")
    for key in groups:
        print(f"    {key[0]}, n={key[1]}: {len(groups[key])} data points")

    # ── Step 2: For each group, compute P(k) mean ± CI, fit ε_eff ──
    summary_records = []
    
    for (backend, n), data in groups.items():
        N = data[0]['N']
        g = int(np.median([d['g'] for d in data]))  # Use median gate count
        
        print(f"\n[Step 2] Backend={backend}, n={n}, N={N}, g={g}")

        k_values = sorted(set(d['k'] for d in data))
        
        # Build p_matrix: (n_seeds, len(k_values))
        seeds = sorted(set(d['seed'] for d in data))
        p_matrix = np.zeros((len(seeds), len(k_values)))
        
        for i, seed in enumerate(seeds):
            for j, k in enumerate(k_values):
                matching = [d['p'] for d in data if d['k'] == k and d['seed'] == seed]
                if matching:
                    p_matrix[i, j] = matching[0]
                else:
                    p_matrix[i, j] = np.nan

        # Mean and 95% CI across seeds
        p_mean = np.nanmean(p_matrix, axis=0)
        p_std = np.nanstd(p_matrix, axis=0)
        p_ci = 1.96 * p_std / np.sqrt(np.sum(~np.isnan(p_matrix), axis=0))

        # Fit ε_eff
        fit = fit_eps_eff(k_values, p_mean.tolist(), N)
        eps_eff = fit['eps_eff']
        eps_device = extract_eps_device(eps_eff, g)
        
        # Bootstrap CI on ε_eff
        ci_result = bootstrap_ci(k_values, p_matrix, N)
        eps_device_lower = extract_eps_device(ci_result['lower'], g)
        eps_device_upper = extract_eps_device(ci_result['upper'], g)

        # Compute k* and GFS
        gfs_result = compute_gfs(N, eps_eff)
        
        print(f"  ε_eff      = {eps_eff:.5f}")
        print(f"  ε_device   = {eps_device:.5f} [{eps_device_lower:.5f}, {eps_device_upper:.5f}]")
        print(f"  k_ideal    = {gfs_result['k_ideal']}")
        print(f"  k*         = {gfs_result['k_star_continuous']:.2f} → {gfs_result['k_star_int']}")
        print(f"  k* (asymp) = {gfs_result['k_star_asymptotic']:.2f}")
        print(f"  GFS        = {gfs_result['gfs']:.4f}")

        record = {
            'backend': backend,
            'n': n,
            'N': N,
            'g': g,
            'eps_eff': eps_eff,
            'eps_device': eps_device,
            'eps_device_ci_lower': eps_device_lower,
            'eps_device_ci_upper': eps_device_upper,
            **gfs_result,
            'k_values': k_values,
            'p_mean': p_mean.tolist(),
            'p_ci': p_ci.tolist(),
            'p_fitted': fit['p_fitted'].tolist(),
        }
        summary_records.append(record)

        # Plot P(k) curve for this group
        plot_pk_curve(record, ideal_baselines[n], OUT_DIR)

    # ── Step 3: GFS summary table ──
    print("\n[Step 3] GFS Summary Table")
    print(f"{'Backend':<20} {'n':>3} {'ε_device':>10} {'k_ideal':>8} {'k*':>6} {'GFS':>8}")
    print("-" * 60)
    for r in summary_records:
        print(f"{r['backend']:<20} {r['n']:>3} {r['eps_device']:>10.5f} "
              f"{r['k_ideal']:>8} {r['k_star_int']:>6} {r['gfs']:>8.4f}")

    # Save full summary
    summary_file = OUT_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nFull summary saved to {summary_file}")

    # Final GFS comparison plot
    plot_gfs_comparison(summary_records, OUT_DIR)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_pk_curve(record: dict, ideal_baseline: dict, out_dir: Path):
    """Publication-quality P(k) curve: empirical, fitted, ideal."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    k_vals = record['k_values']
    k_fine = np.linspace(0.5, max(k_vals) + 0.5, 200)
    N = record['N']
    
    # Ideal curve
    p_ideal_fine = [p_ideal(k, N) for k in k_fine]
    ax.plot(k_fine, p_ideal_fine, 'k--', linewidth=1.5, label='Ideal (ε=0)', alpha=0.6)
    
    # Fitted theoretical curve
    p_fit_fine = [p_noisy(k, N, record['eps_eff']) for k in k_fine]
    ax.plot(k_fine, p_fit_fine, 'b-', linewidth=2,
            label=f'Theorem 1 fit (ε_eff={record["eps_eff"]:.4f})')
    
    # Empirical points with CI
    ax.errorbar(k_vals, record['p_mean'], yerr=record['p_ci'],
                fmt='ro', capsize=4, markersize=6, linewidth=1.5,
                label='Hardware (mean ± 95% CI)')
    
    # Mark k* and k_ideal
    ax.axvline(x=record['k_star_continuous'], color='blue', linestyle=':', alpha=0.7,
               label=f"k*={record['k_star_int']}")
    ax.axvline(x=record['k_ideal'], color='black', linestyle=':', alpha=0.4,
               label=f"k_ideal={record['k_ideal']}")
    
    # Random guessing baseline
    ax.axhline(y=1.0/N, color='gray', linestyle='--', alpha=0.5, linewidth=1,
               label=f'Random (1/N={1/N:.3f})')
    
    ax.set_xlabel('Grover iterations k', fontsize=12)
    ax.set_ylabel('Success probability P(k)', fontsize=12)
    ax.set_title(f'P(k) curve — {record["backend"]}, n={record["n"]}\n'
                 f'GFS = {record["gfs"]:.3f}', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    fname = out_dir / f"pk_curve_{record['backend']}_n{record['n']}.pdf"
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Plot saved: {fname}")


def plot_ideal_curves(ideal_baselines: dict):
    """Plot ideal P(k) curves for verification before hardware data arrives."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, (n, baseline) in zip(axes, ideal_baselines.items()):
        k_fine = np.linspace(0, max(baseline['k_values']) + 1, 300)
        N = 2 ** n
        p_fine = [p_ideal(k, N) for k in k_fine]
        
        ax.plot(k_fine, p_fine, 'k-', linewidth=2)
        ax.axvline(x=baseline['k_ideal'], color='red', linestyle='--',
                   label=f"k_ideal = {baseline['k_ideal']}")
        ax.axhline(y=1/N, color='gray', linestyle=':', alpha=0.6, label=f'1/N = {1/N:.3f}')
        ax.set_title(f'Ideal Grover, n={n} (N={N})')
        ax.set_xlabel('Iterations k')
        ax.set_ylabel('P(k)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = OUT_DIR / "ideal_baselines.pdf"
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Ideal baseline plot saved: {fname}")


def plot_gfs_comparison(summary_records: list, out_dir: Path):
    """Bar chart comparing GFS across backends and qubit counts."""
    if not summary_records:
        return
    
    labels = [f"{r['backend'][:12]}\nn={r['n']}" for r in summary_records]
    gfs_values = [r['gfs'] for r in summary_records]
    
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    bars = ax.bar(labels, gfs_values, color='steelblue', edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Perfect device (GFS=1)')
    ax.set_ylabel('Grover Fidelity Score (GFS)', fontsize=12)
    ax.set_title('GFS across backends — Definition 1', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, gfs_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fname = out_dir / "gfs_comparison.pdf"
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"GFS comparison plot saved: {fname}")


if __name__ == "__main__":
    run_analysis()


# ─── Extended Analysis Functions ──────────────────────────────────────────────

def load_extended_results() -> list:
    """Load all extended experiment results (seeds, backend, temporal)."""
    extended = []
    for results_file in LOG_DIR.glob("extended_*_results.jsonl"):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    extended.append(json.loads(line))
    print(f"Loaded {len(extended)} extended results.")
    return extended


def run_extended_analysis():
    """
    Extended analysis pipeline. Run after collect_extended_results.
    Produces:
      - Tighter CI plots (10+ seeds)
      - Multi-backend GFS comparison
      - Temporal ε_device drift chart
      - IBM calibration vs Algorithm 1 comparison table
    """
    print("\n" + "="*60)
    print("EXTENDED ANALYSIS — Multi-backend, Temporal, Tighter CI")
    print("="*60)

    extended = load_extended_results()
    if not extended:
        print("No extended results yet. Run submit_extended.py first.")
        return

    # Load original results too
    original = load_all_results()
    all_results = original + extended

    # ── Group by (backend, n, mode) ──
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        mode = r.get('mode', 'original')
        key = (r['backend_name'], r['n_qubits'], mode)
        groups[key].append(r)

    print(f"\nGroups found: {len(groups)}")
    for key in sorted(groups.keys()):
        print(f"  {key[0]}, n={key[1]}, mode={key[2]}: {len(groups[key])} points")

    summary_extended = []

    for (backend, n, mode), data in groups.items():
        if mode == 'temporal' or mode == 'seeds' or mode == 'original' or mode.startswith('backend_'):
            N = data[0]['N']
            g = int(np.median([d['g'] for d in data if 'g' in d] or
                              [d.get('transpiled_total_gates', 409) for d in data]))

            k_values = sorted(set(d['k_iterations'] for d in data))
            seeds = sorted(set(d['seed'] for d in data))

            p_matrix = np.full((len(seeds), len(k_values)), np.nan)
            for i, seed in enumerate(seeds):
                for j, k in enumerate(k_values):
                    matching = [d['p_empirical'] for d in data
                                if d['k_iterations'] == k and d['seed'] == seed]
                    if matching:
                        p_matrix[i, j] = matching[0]

            p_mean = np.nanmean(p_matrix, axis=0)
            p_std = np.nanstd(p_matrix, axis=0)
            n_valid = np.sum(~np.isnan(p_matrix), axis=0)
            p_ci = 1.96 * p_std / np.sqrt(n_valid)

            fit = fit_eps_eff(k_values, p_mean.tolist(), N)
            eps_eff = fit['eps_eff']
            eps_device = extract_eps_device(eps_eff, g)
            ci_result = bootstrap_ci(k_values, p_matrix, N)

            gfs_result = compute_gfs(N, eps_eff)

            timestamp = None
            timestamps = [d.get('submission_timestamp') for d in data if d.get('submission_timestamp')]
            if timestamps:
                timestamp = min(timestamps)

            record = {
                'backend': backend,
                'n': n,
                'N': N,
                'g': g,
                'mode': mode,
                'n_seeds': len(seeds),
                'timestamp': timestamp,
                'eps_eff': eps_eff,
                'eps_device': eps_device,
                'eps_device_ci_lower': extract_eps_device(ci_result['lower'], g),
                'eps_device_ci_upper': extract_eps_device(ci_result['upper'], g),
                **gfs_result,
                'k_values': k_values,
                'p_mean': p_mean.tolist(),
                'p_ci': p_ci.tolist(),
            }
            summary_extended.append(record)

            print(f"\n  [{mode}] {backend}, n={n}, {len(seeds)} seeds:")
            print(f"    ε_device = {eps_device:.6f} [{record['eps_device_ci_lower']:.6f}, {record['eps_device_ci_upper']:.6f}]")
            print(f"    k* = {gfs_result['k_star_continuous']:.3f} → {gfs_result['k_star_int']}")
            print(f"    GFS = {gfs_result['gfs']:.4f}")

    # ── Multi-backend GFS comparison plot ──
    backend_records = [r for r in summary_extended
                       if r['mode'].startswith('backend_') or r['mode'] == 'original']
    if len(set(r['backend'] for r in backend_records)) > 1:
        plot_multi_backend_gfs(backend_records, OUT_DIR)

    # ── Temporal drift plot ──
    temporal_records = [r for r in summary_extended
                        if r['mode'] in ['original', 'temporal']]
    if temporal_records:
        plot_temporal_drift(temporal_records, OUT_DIR)

    # ── Tighter CI comparison ──
    seed_records = [r for r in summary_extended
                    if r['mode'] in ['original', 'seeds']]
    if seed_records:
        plot_ci_comparison(seed_records, OUT_DIR)

    # ── IBM calibration comparison ──
    cal_file = LOG_DIR / "ibm_calibration_data.json"
    if cal_file.exists():
        compare_calibration(summary_extended, cal_file, OUT_DIR)

    # Save extended summary
    ext_summary_file = OUT_DIR / "extended_summary.json"
    with open(ext_summary_file, 'w') as f:
        json.dump(summary_extended, f, indent=2)
    print(f"\nExtended summary saved to {ext_summary_file}")


def plot_multi_backend_gfs(records, out_dir):
    """Bar chart comparing GFS across multiple backends."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, n in zip(axes, [3, 4]):
        n_records = [r for r in records if r['n'] == n]
        if not n_records:
            continue

        backends = [r['backend'] for r in n_records]
        gfs_vals = [r['gfs'] for r in n_records]
        eps_vals = [r['eps_device'] for r in n_records]
        eps_lower = [r['eps_device_ci_lower'] for r in n_records]
        eps_upper = [r['eps_device_ci_upper'] for r in n_records]

        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(backends)))
        bars = ax.bar(backends, gfs_vals, color=colors, edgecolor='black', linewidth=0.5)

        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.6, label='Perfect (GFS=1)')
        ax.set_ylabel('GFS', fontsize=12)
        ax.set_title(f'GFS by Backend — n={n} (N={2**n})', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        for bar, val, eps, lo, hi in zip(bars, gfs_vals, eps_vals, eps_lower, eps_upper):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'GFS={val:.3f}\nε={eps:.5f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fname = out_dir / "multi_backend_gfs.pdf"
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Multi-backend GFS plot saved: {fname}")


def plot_temporal_drift(records, out_dir):
    """
    Plot ε_device over time to show temporal fingerprinting.
    Shows drift in noise rate between original run and post-recalibration run.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, n in zip(axes, [3, 4]):
        n_records = sorted([r for r in records if r['n'] == n],
                          key=lambda x: x.get('timestamp', ''))
        if len(n_records) < 2:
            ax.set_title(f'n={n}: Need 2+ time points for drift analysis')
            continue

        times = [r.get('timestamp', f"Run {i+1}") for i, r in enumerate(n_records)]
        labels = [f"Run {i+1}\n({r['mode']})\n{r.get('n_seeds',5)} seeds"
                  for i, r in enumerate(n_records)]
        eps_vals = [r['eps_device'] for r in n_records]
        eps_lower = [r['eps_device_ci_lower'] for r in n_records]
        eps_upper = [r['eps_device_ci_upper'] for r in n_records]
        yerr_lower = [e - lo for e, lo in zip(eps_vals, eps_lower)]
        yerr_upper = [hi - e for e, hi in zip(eps_vals, eps_upper)]

        x = range(len(n_records))
        ax.errorbar(x, eps_vals,
                   yerr=[yerr_lower, yerr_upper],
                   fmt='o-', capsize=5, markersize=8, linewidth=2,
                   color='steelblue', ecolor='navy', label='ε_device ± 95% CI')

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('ε_device (per-gate error rate)', fontsize=11)
        ax.set_title(f'Temporal Fingerprint — n={n}, {n_records[0]["backend"]}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fname = out_dir / "temporal_drift.pdf"
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Temporal drift plot saved: {fname}")


def plot_ci_comparison(records, out_dir):
    """Show how CI tightens as seed count increases."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, n in zip(axes, [3, 4]):
        n_records = sorted([r for r in records if r['n'] == n],
                          key=lambda x: x.get('n_seeds', 5))
        if not n_records:
            continue

        seed_counts = [r.get('n_seeds', 5) for r in n_records]
        ci_widths = [(r['eps_device_ci_upper'] - r['eps_device_ci_lower']) * 1e6
                     for r in n_records]
        eps_vals = [r['eps_device'] * 1e6 for r in n_records]

        ax.plot(seed_counts, ci_widths, 'rs-', markersize=8, linewidth=2,
                label='95% CI width (\u00D710\u207B\u2076)')
        ax2 = ax.twinx()
        ax2.plot(seed_counts, eps_vals, 'bo--', markersize=6, linewidth=1.5,
                 alpha=0.6, label='ε_device (\u00D710\u207B\u2076)')
        ax2.set_ylabel('ε_device (\u00D710\u207B\u2076)', fontsize=11, color='blue')

        ax.set_xlabel('Number of seeds', fontsize=11)
        ax.set_ylabel('CI width (\u00D710\u207B\u2076)', fontsize=11, color='red')
        ax.set_title(f'CI Tightening vs. Seeds — n={n}', fontsize=11)
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    fname = out_dir / "ci_tightening.pdf"
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  CI tightening plot saved: {fname}")


def compare_calibration(summary_records, cal_file, out_dir):
    """
    Compare Algorithm 1 extracted ε_device against IBM published calibration.
    """
    with open(cal_file) as f:
        cal_data = json.load(f)

    print("\n[Calibration Comparison]")
    print(f"{'Backend':<20} {'n':>3} {'ε_device (Alg1)':>18} {'IBM CX mean':>14} {'IBM CX range':>22}")
    print("-" * 80)

    comparison = []
    for r in summary_records:
        backend = r['backend']
        if backend not in cal_data:
            continue
        cal = cal_data[backend]
        ibm_cx = cal.get('cx_error_mean')
        ibm_min = cal.get('cx_error_min')
        ibm_max = cal.get('cx_error_max')

        if ibm_cx:
            print(f"{backend:<20} {r['n']:>3} {r['eps_device']:>18.6f} "
                  f"{ibm_cx:>14.6f} [{ibm_min:.6f}, {ibm_max:.6f}]")

            comparison.append({
                'backend': backend,
                'n': r['n'],
                'eps_device_alg1': r['eps_device'],
                'eps_device_ci_lower': r['eps_device_ci_lower'],
                'eps_device_ci_upper': r['eps_device_ci_upper'],
                'ibm_cx_error_mean': ibm_cx,
                'ibm_cx_error_min': ibm_min,
                'ibm_cx_error_max': ibm_max,
                'ratio_alg1_to_ibm': r['eps_device'] / ibm_cx if ibm_cx else None
            })

    cal_summary = out_dir / "calibration_comparison.json"
    with open(cal_summary, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Calibration comparison saved: {cal_summary}")


# ── Entry point ──
if __name__ == "__main__":
    import sys
    if '--extended' in sys.argv:
        run_extended_analysis()
    else:
        run_analysis()
