# CVaR PSHA Bandit

Prove that an autonomous agent can learn the theoretically optimal Importance Sampling (IS) distribution for Conditional Value-at-Risk (CVaR) in a Probabilistic Seismic Hazard Analysis (PSHA) logic tree—using only a scalar reward.

## Experiments

### 1-node (GMM only)

One epistemic GMM node with three arms (`ln Y ~ N(μ, σ)`).

```powershell
python scripts/run.py --config experiments/config.yaml
```

Outputs: `results/`

### 3-node delayed-reward MDP

Classic epistemic chain **Source → Magnitude → GMM**.

```powershell
python scripts/run.py --config experiments/config_3node.yaml
```

Outputs: `results/3node/` (`cvar_convergence.png`, `ess_comparison.png`)

### 2D multi-site portfolio hazard MDP

Fault-line ruptures, \(K=10\) sites, Gutenberg–Richter magnitude bins, epistemic geometry + GMM (Manager) and mag + rupture location (Worker).

Portfolio loss \(L = \sum_k \mathrm{PGA}_k\) under a distance-attenuated GMPE footprint. Leaf reward:

```text
r_T = (prod_t w/q) * L * 1{L > v95}
```

Methods: Naive MC, Flat REINFORCE, **CVaR-CPO**, **Hierarchical** (~1920 joint paths).

```powershell
python scripts/run.py --config experiments/config_spatial.yaml
```

Outputs: `results/spatial/`

- `cvar_convergence.png`, `ess_comparison.png`
- `fault_mag_mass.png` — mag × rupture marginal mass vs \(q^*\)

## Setup

```powershell
cd C:\Users\kurt-\Projects\cvar-psha-bandit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Metrics

- **CVaR convergence** toward the large-sample true portfolio CVaR
- **ESS** = `(Σ W)² / Σ W²` to detect path-policy collapse
