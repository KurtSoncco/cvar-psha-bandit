# Background for the next EQ-JEPA iteration

Two inputs for the plan that follows: (1) what the JEPA literature has settled since the
L0-L3 build, and (2) whether "forecast energy instead of magnitude" is a usable idea.

Literature pull: arXiv API, 6 queries, 197 unique records, filtered to 2025-01-01 onward
(`jepa_seismicity_corpus_2025_2026.csv`, 167 records; 18 read in detail in
`jepa_seismicity_key_papers.csv`). Energy appraisal: the attached paper plus a numeric check
on this project's own corrected California catalog (19,152 events, M2.8+, 2000-2024).

---

## 1 JEPA literature: four things that bear directly on our open defects

### 1.1 Anti-collapse now has theory, and our EMA + stop-gradient setup is the thing it replaces

`LeJEPA` (arXiv:2511.08544, Balestriero & LeCun, Nov 2025) is the load-bearing reference. It
identifies the **isotropic Gaussian** as the embedding distribution that minimises downstream
prediction risk for a JEPA, and enforces it with **SIGReg** — a sketched-projection Gaussianity
test used as a loss term. The practical consequence for us is that the whole heuristic stack
(EMA teacher, stop-gradient, and the absence of any isotropy term) is exactly what LeJEPA
removes: one explicit regulariser, linear time and memory, no teacher-student asymmetry.

Two follow-ups sharpen it:

- `λ-JEPA / SACReg` (arXiv:2609.35288) shows anti-collapse applied **after a projection head**
  does not guarantee a non-collapsed **backbone**: the backbone can retain low effective rank
  while the objective looks healthy. Our diagnostic must therefore measure rank on the encoder
  output that L1-L4 actually consume, not on anything downstream of a projector.
- `No Gaussian Required` (arXiv:2608.17542) argues the opposite design: prescribing what the
  latent must look like is environment-independent, and a contrastive inverse-dynamics term
  can prevent collapse while letting the data set the geometry.

### 1.2 Isotropy is not automatically what we want, because L4 scores distances in the latent

`Anisotropic Representations Improve Planning in JEPA World Models` (arXiv:2609.37441): accurate
prediction plus non-collapsed representation still does **not** guarantee a latent geometry
aligned with the downstream planning cost — isotropic Gaussian regularisation can induce a
geometry that mis-weights terminal errors. This is a direct warning for our L4/L5 layers, where
the decision cost is evaluated on top of `z_t`. The implication is to treat isotropy as a
*collapse remedy* and report the latent geometry against the CVaR cost separately, rather than
assuming the two objectives coincide.

### 1.3 "Hierarchical" in the JEPA literature means temporal abstraction; ours is spatial

`H-JEPA` (arXiv:2610.06805, Oct 2026) is the first end-to-end hierarchical JEPA: a stack of
action-conditioned JEPAs in which **each level predicts farther ahead in its own learned latent
space**, with top-down planning. Our hierarchy is a spatial parent/child rate tree enforced by
a softmax allocation head (invariant I2); temporal hierarchy is currently a single
horizon-conditioned decode. H-JEPA is the template for making the horizon axis a real hierarchy,
which is also what invariant I2's temporal-additivity half needs.

`Hamiltonian JEPA` (arXiv:2609.33497) is a second structural idea worth noting: it stops asking
one embedding to serve both perception and control, splitting a wide isotropy-regularised
perceptual code from a narrow control state. Our `z_t` currently serves both forecasting (L0/L1)
and decision-making (L5).

### 1.4 One-step next-latent prediction is not a rollout — this is a correctness issue for us

`One-Step Next-Latent Prediction Is Not a World Model` (arXiv:2609.36227) makes a point that
applies to L0 as written: a one-step latent regression identifies a **conditional mean**, and a
mean is a transition kernel only in special cases (linear-Gaussian Markov latents). Open-loop
rollout of such a model is not a draw from the predictive distribution. Since L4 needs a
multi-horizon distribution to form a CVaR, any multi-step use of L0's latent has to predict a
*distribution* (innovation covariance at minimum), not a point in latent space.

### 1.5 JEPA on time series: four recent precedents

| Paper | What to take |
|---|---|
| `SC-JEPA` (arXiv:2602.04643) | Continuous self-distillation on time series is unstable and collapse-prone; multi-scale precursor modelling. Closest published analogue of our failure mode. |
| `Distributed JEPA` (arXiv:2609.17029) | Masked-segment latent prediction across heterogeneous **energy** time series — the architecture pattern if an energy channel is added. |
| `ER-JEPA` (arXiv:2607.01145) | Hierarchical SSL for multivariate time series with event reconstruction; lightweight, small-target-dataset regime, which is ours. |
| `Var-JEPA` (arXiv:2603.20111) | Variational reading of JEPA: context-to-target prediction mirrors a learned conditional prior. Useful framing if L0 becomes an explicit probabilistic point process. |

---

## 2 Neural seismicity forecasting: the baselines we now have to cite

Three 2026 papers overlap our open items so closely that they are both competitors and
references for the redesign:

- **`Fusion`, scaling-law-informed neural marked point process** (arXiv:2608.18791). A neural
  catalog-history representation fused with ETAS-derived temporal features and
  Gutenberg-Richter magnitude structure, with the input magnitude cutoff separated from the
  target threshold. This is the §9 point-process redesign, published, with the
  self-excitation built into the model rather than bolted on as our 2-parameter Hawkes patch.
- **`EarthquakeNet`, neural negative-binomial regression** (arXiv:2605.21437). Rejects the
  single-global-dispersion Poisson assumption (Central Asia, p < 1e-179) and estimates
  **per-cell** overdispersion endogenously, explicitly for tail risk. Our L0 already has a
  negative-binomial count head; this is how to use it, and it speaks directly to the N-test
  behaviour.
- **Conditional generative aftershock forecasting** (arXiv:2607.24109). Targets exactly what
  ETAS's isotropic spatial kernel misses — fault-controlled spatial patterns.

Two more for honest evaluation: `Neural Earthquake Forecasting with Minimal Information`
(arXiv:2509.14661) uses temporally stratified cross-validation and NN+Markov hybrids to ask
whether the neural part earns its keep — the same question our T-tests are asking; and
`EPBench` (arXiv:2505.15588) is a global regional-scale short-term prediction benchmark
(924,472 records) that would give us the cross-region evaluation we currently lack.

---

## 3 The energy-target idea

### 3.1 What the paper does

Kavitha & Raghukanth, *Regional level forecasting of seismic energy release*, Acta Geod Geophys
51:359-391 (2016), doi:10.1007/s40328-015-0131-7. Pipeline: ISC-GEM catalogue → 41 seismogenic
zones → per-event magnitudes converted to radiated energy with the Choy & Boatwright (1995)
relation → **annual** energy summed per zone → PCA across zones to group them (41 zones into 16
correlated groups) → ensemble empirical mode decomposition of each group's annual energy series
into intrinsic mode functions → a small ANN (5 inputs, one hidden layer of 5 nodes) modelling the
series in two parts (`Y_t` then the residual `Z_t`) → point forecast of 2015 annual energy per
zone. Verification is by RMS error, correlation coefficient, and a "performance parameter", on a
held-out tail of the series. Note there is a published Comment on this paper
(doi:10.1007/s40328-015-0157-x, Wu 2016) which I have not read — worth retrieving before leaning
on the result.

### 3.2 Why the recipe as written does not transfer

**(a) Converting magnitude to energy adds no information.** The Choy-Boatwright conversion is
affine in log space,

        log10 E_s = 1.5 M + const

so event-level energy is a strictly monotone relabelling of catalog magnitude. A forecast of
converted energy is the *same forecast* as a forecast of magnitude, pushed through a bijection.
Any information gain has to come from the aggregation or the loss function — not from the target
variable. (Real energy information would require measured radiated energy or moment: ComCat `Me`,
GCMT `M0`. Apparent stress varies by roughly an order of magnitude at fixed magnitude, and that
scatter is exactly what the conversion throws away.)

**(b) Summed energy is a maximum, not a sum.** Because `E ∝ 10^(1.5M)`, a window's total energy
is dominated by its single largest event. On our own corrected California catalog:

| statistic | monthly windows (n=300) | annual windows (n=25) |
|---|---|---|
| median share of window energy from the largest event | 0.42 | 0.50 |
| windows where the largest event is >50% of energy | 41% | 48% |
| windows where it is >90% | 6.3% | 8.0% |
| median share from the top 10% of events by count | 0.79 | 0.95 |

Across the whole 2000-2024 catalog, the single M7.2 El Mayor-Cucapah event carries **41%** of all
radiated energy, the ten largest events carry **88%**, and the largest 0.1% of events carry
**91%**. Correlation of window log-energy with the window's maximum magnitude is **0.98**,
against 0.75 for log-count. So an energy-release forecast is, to a very good approximation, a
largest-magnitude forecast with extra steps — a far harder and much less identifiable target
than a rate, and one on which RMSE and correlation coefficients are close to meaningless.

![Window energy tracks the largest event, not the event count. Left: total radiated energy per monthly window against the radiated energy of that window's single largest event (Choy-Boatwright conversion, corrected California catalog, M2.8+, 2000-2024), with the 1:1 line. Right: empirical CDF of the largest event's share of window energy, monthly and annual. Dotted line marks the 50% share.]({{artifact:art_75221946-9f78-4f9f-8372-db4faf63598c}})

**(c) The validation protocol would not survive our own standards.** Two specific problems: EMD
and EEMD are **non-causal** — decomposing the full series into IMFs and only then splitting
train/test leaks future information into the training features, which is the same class of error
as our hand-listed `issue_times` violating prospectivity; and RMSE / correlation / "performance
parameter" are not proper scoring rules, produce no predictive distribution, and are not
CSEP-testable, so nothing from this route could enter floatCSEP or produce a CVaR.

### 3.3 What is worth taking from it

The useful residue is that **energy is the right weighting, not the right target** — and that
matters to us more than to most forecasters, because our loss functional is a tail risk. Three
concrete forms:

1. **A moment-rate budget as a physical constraint on L0's total rate.** Our headline defect is a
   ~3.5x mean-rate deficit currently patched by a 2-parameter Hawkes-like multiplier fit to the
   data. The seismic moment rate implied by a forecast (Kostrov summation over the predicted
   magnitude-frequency distribution) can be compared against an independent tectonic/geodetic
   moment-rate budget for the region, which is standard practice in PSHA source models. That turns
   the rate correction from a fitted patch into a physics-anchored constraint — and it is a
   constraint the baselines must satisfy too, which removes the asymmetry in the current T-test
   comparison.
2. **An energy-weighted consistency test ("E-test") beside the N-test.** The N-test asks whether
   the forecast got the *number* of events right, weighting an M3 the same as an M7. For a CVaR
   objective the relevant question is whether it got the *moment release* right. Built with
   pycsep's existing simulate-and-rank pattern (as proposed for the depth D-test), this is a
   quantile of observed `ΣE` against ~1000 simulated `ΣE`, and it is the diagnostic that would
   actually correlate with downstream CVaR error.
3. **An energy channel as an auxiliary prediction target, not the forecast.** If L0's encoder is
   given measured `M0`/`Me` where available and asked to predict a latent moment-release channel
   alongside the rate field, the extra target up-weights large events in the representation
   without changing what the forecast object is. `Distributed JEPA` (arXiv:2609.17029) is the
   architectural precedent for a masked-latent energy channel.

---

## 4 References

JEPA / representation learning — arXiv:2511.08544 (LeJEPA), 2610.06805 (H-JEPA),
2609.35288 (λ-JEPA/SACReg), 2609.37441 (anisotropic planning), 2608.17542 (contrastive inverse
dynamics), 2609.36227 (one-step is not a world model), 2609.33497 (Hamiltonian JEPA),
2602.04643 (SC-JEPA), 2609.17029 (Distributed JEPA), 2607.01145 (ER-JEPA), 2603.20111 (Var-JEPA).

Seismicity forecasting — arXiv:2608.18791 (Fusion NPP), 2605.21437 (EarthquakeNet NB),
2607.24109 (conditional generative aftershocks), 2606.10069 (VQ-VAE + seismic statistical
features), 2509.14661 (minimal-information limits), 2505.15588 (EPBench), 2603.03079 (b-value DL
forecast evaluation).

Energy — doi:10.1007/s40328-015-0131-7 (Kavitha & Raghukanth 2016); doi:10.1007/s40328-015-0157-x
(Comment, Wu 2016); Choy & Boatwright (1995) energy-magnitude relation.
