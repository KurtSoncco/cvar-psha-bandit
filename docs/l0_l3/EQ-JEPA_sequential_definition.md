# A sequential scientific definition for the EQ-JEPA / CVaR-PSHA framework

*Connecting Hierarchical JEPA, floatCSEP, CVaR, and nonlinear SDOF into one falsifiable chain.
Written against `Filippo_Gatti_big_picture.pdf` (ch. 5, Perspectives), `EQ-JEPA_brainstorming.md`,
`EQ-JEPA_alignment_review.md`, and the `filippo` branch of `cvar-psha-bandit` at commit `5a7df6b`.*

Notation: `λ(c,m,Δ)` is the seismicity rate over cell `c`, magnitude bin `m`, horizon `Δ`;
`M` magnitude; `ε` the ground-motion-model residual in units of `σ_lnSa`; `z_gm` the latent of the
waveform generator; `μ_d` peak ductility; `Y` the loss ratio; `θ_a` the asset capacity parameters;
`α` is overloaded by convention and is disambiguated at each use (CVaR level `α_CVaR = 0.99`
throughout; post-yield stiffness ratio written `α_pv`).

---

## 0. The one-sentence definition

> **A hierarchical JEPA world model produces a horizon-coherent seismicity rate field; floatCSEP
> certifies that field against data before anything downstream is allowed to consume it; a
> conditional ground-motion generator turns certified rates into waveform ensembles; a nonlinear
> SDOF oscillator with a collapse mechanism turns waveforms into a loss distribution that has a
> tail and an atom; the elicitable pair (VaR, CVaR) is estimated on that distribution by importance
> sampling tilted towards the tail — not towards the hazard disaggregation; and only then does a
> constrained decision layer choose actions, which act on the asset's capacity and never on the
> seismicity model.**

![Layered architecture: six layers from seismicity to decision, with three dashed feedback loops]({{artifact:art_117ee2f5-d014-4f08-a98e-cbe696895884}})

Each arrow in that diagram is a **typed interface**, and each box is **separately falsifiable**.
That is the whole design claim: the framework is not a monolith trained end-to-end, it is a
sequence of layers each of which can be rejected on its own evidence.

---

## 1. The five invariants

These come out of the brainstorming note and the alignment review, and every design decision below
is downstream of them.

**I1 — Seismicity is never action-conditioned.** Actions enter only at the asset/exposure layer.
An inspection does not change the rate of earthquakes. Making actions part of the earthquake
dynamics would make the forecast unscoreable against the catalogue, which is the only ground truth
available.

**I2 — Hierarchical rate coherence holds by construction, not by penalty.** Both
`λ_parent = Σ_children λ_child` and `λ(Δ) = Σ_i λ(δ_i)` for `Δ = Σ_i δ_i`. A soft consistency
penalty leaves the model free to be incoherent where it matters most (the tail). Exact additivity is
cheap — a softmax allocation head — and removes a whole failure mode from the evaluation.

**I3 — Risk is reported and scored only through elicitable objects:** the pair `(VaR_α, CVaR_α)`
with the Fissler–Ziegel joint loss. CVaR alone is not elicitable and must never be a training or
model-selection criterion on its own; without this, "the model with the lower CVaR score" is a
meaningless ranking.

**I4 — Every layer is validated against data or a closed-form oracle before the next layer consumes
it.** This is what floatCSEP buys at L1 and what the Houng et al. analytic targets already buy for
the estimator side.

**I5 — The importance-sampling proposal is a variance-reduction device only;** it must not move the
reported risk. Self-normalised IS is biased at finite budget, so ESS / tail-ESS and a bootstrap
interval are reported alongside every estimate. Otherwise a "better" proposal and a "worse" hazard
become indistinguishable.

---

## 2. The layers, as definitions

### L0 — Seismicity layer: Hierarchical JEPA

State the model as a predictive latent dynamics, not as a generative catalogue model.

- **Context encoder** `z_t = E_θ(H_t)` over the observed history `H_t` — short-window event tokens
  plus long-window monthly rate/moment summaries.
- **Horizon-conditioned predictor** `ẑ_{t+Δ} = P_φ(z_t, Δ)`. The horizon `Δ` is an *argument*, not
  a configuration constant — this is what makes the object hierarchical in time rather than a
  single-horizon regressor.
- **Target encoder** `z̄_{t+Δ} = E_θ̄(F_{t,Δ})`, a momentum (EMA) copy of `E_θ` with stop-gradient,
  applied to the held-out future window.
- **Latent loss** `L_JEPA = ‖ẑ_{t+Δ} − z̄_{t+Δ}‖² + β·R_SIReg(z)`, with an explicit
  isotropy/variance regulariser. The deck's argument that the admissible target embedding geometry
  is the isotropic Gaussian makes SIReg (or VICReg-style variance–covariance terms) the principled
  choice, and it is the *only* thing standing between the momentum target and representation
  collapse.
- **Rate head** `λ(c,m,Δ) = Λ_Δ(z_t) · π(c,m | z_t, Δ)`, with `Λ_Δ = softplus(·)` the expected total
  count over `Δ` and `π` a softmax over cell × magnitude bins. I2 then holds exactly by
  construction. Temporal additivity additionally requires `Λ` to be built from a non-negative
  intensity integrated over sub-intervals, rather than predicted independently per horizon.
- **Count likelihood** negative binomial, not Poisson. Clustered seismicity is overdispersed; a
  Poisson head is misspecified in exactly the regime — aftershock sequences — that drives the tail,
  and it biases the CSEP N-test.

**Output:** `λ(c, m, Δ | t₀)` — one gridded forecast per (issue time, horizon), deterministic given
a seed.

### L1 — Validation layer: floatCSEP

floatCSEP is doing **two jobs**, and it is worth being explicit that the second one is not standard
practice.

1. **Forecast certification.** N-, S-, M-, CL-tests per forecast window, plus paired T-tests against
   baselines — the repo's `HistoricalRateBaseline`, `ETASLiteBaseline`, and a smoothed-seismicity
   reference. Rolling-origin issue times, so the experiment is genuinely prospective in structure.
2. **Representation-collapse audit.** A collapsed JEPA encoder produces a rate field that is nearly
   invariant to history and horizon. Such a field can still pass the N-test — it gets the long-run
   total right — while failing the S-test and CL-test, because it puts the mass in the
   climatological place rather than the current place. That makes the CSEP test battery a
   *behavioural* collapse detector complementing the geometric diagnostics (embedding rank,
   per-dimension standard deviation, sketched-projection Gaussianity) computed inside L0. This is
   the cheapest available answer to "how do we know the world model learned dynamics and not a
   prior?"

**Gate:** a rate field that fails L1 does not propagate. This is the dashed green loop in the
figure.

### L2 — Ground-motion layer: conditional waveform generator

Given a certified rate field, sample scenarios `(c, m)` and generate **waveform ensembles**
`{a_g(t)}` conditioned on `(M, R, site)` with an explicit latent `z_gm`. Section 4.2 is the argument
for why this layer cannot be collapsed into a scalar ground-motion model when the target is CVaR.
Staging: start with record selection from a conditional-spectrum bin — then `z_gm` is a record index
and everything downstream works unchanged — and substitute a MIFNO / SeismoALICE-type generator once
L3 and L4 are closed.

### L3 — Asset layer: nonlinear SDOF and the loss functional

    a_g(t)  →  μ_d = max_t |u(t)| / u_y  →  Y = min{Ȳ, k·μ_d^γ}   ∪   {Y = 1}
                                               └ repair ratio ┘     └ collapse atom ┘

with a hysteresis that can actually fail: bilinear with post-yield stiffness ratio `α_pv`, plus
cyclic strength/stiffness degradation. Collapse is **dynamic instability** — the backbone crossing
zero restoring force at `μ_d = (1 − α_pv)/|α_pv|` for `α_pv < 0` — not an imposed ductility cap. The
distinction is not cosmetic: imposing a cap on a linear oscillator manufactures a collapse atom that
no linear system can produce, and the resulting "tail" is then an artefact of the loss function
rather than a property of the structure.

The capacity parameters `θ_a = (η, α_pv, Ȳ, …)` are **where L5's actions act** — a retrofit changes
`η` or `α_pv`, a closure changes exposure, an inspection changes the posterior over `θ_a`. I1 is
enforced structurally by the fact that `λ` is an input to this layer and never an output of it.

### L4 — Tail-risk layer: elicitable (VaR, CVaR) by tail-targeted IS

With `ω = (t₀, c, m, ε or z_gm)`, `p(ω)` induced by L0–L2, and the excess `h_v(ω) = (Y(ω) − v)⁺`:

    CVaR_α = v_α + (1 − α)⁻¹ · E_p[h_{v_α}]
    (v_α, c_α) = argmin E[ S_FZ(v, c; Y) ]        S_FZ = Fissler–Ziegel joint score

estimated under a proposal `q`, scored by the Fissler–Ziegel joint loss (I3), with ESS / tail-ESS and
bootstrap intervals (I5). The repo's existing G-PMC/AIS machinery and `OnlineCVaRTracker` are the
implementation substrate; what is missing is the score and the correct proposal target (§4.3).

**A design constraint the pilot surfaced:** the CVaR level must be chosen relative to
`Pr(collapse)`. If `1 − α < Pr(collapse)`, then `VaR_α` is pinned to the collapse loss, the excess
term `h_{v_α}` vanishes identically, and `CVaR_α = 1` regardless of the repair-cost distribution —
the tail-sampling problem silently degenerates into a collapse-probability estimation problem.
Always report the pair `(α, Pr(collapse))` together.

### L5 — Decision layer: constrained control, not a bandit

    min_a c(a)   subject to   CVaR_α[ Y | a, λ ] ≤ r̄

solved as a **constrained finite-horizon MPC** over inspection / closure / retrofit, on the `θ_a`
state. The deck's objection to contextual bandits is right and worth restating as a design rule: a
bandit assumes the context is exogenous, but an inspection changes the state the next decision
conditions on, so the bandit formulation is structurally wrong for a maintenance loop — even though
it is fine for the *sampling* problem, which is where the current repo correctly uses it. Latent
planning and RL come after a constrained-MPC baseline exists and after a synthetic oracle can
validate it.

---

## 3. State of the code at `5a7df6b` — verified, not inferred

**What exists and is correct.** The `eq_jepa` module implements a two-level JEPA with an EMA target
branch, stop-gradient, and a softmax allocation head that *does* enforce exact additivity of child
rates to the parent total over cell × magnitude. `CatalogWindowDataset` is leakage-safe: it supplies
separate `future_*` keys, and `counts` is the **future**-window count tensor, so
`model.loss(batch, batch)` is a genuine forecast loss and not a retrodiction — worth stating because
the call signature invites the opposite reading. The CVaR side is a working importance-sampling
testbed benchmarked against closed-form targets, with `OnlineCVaRTracker` reporting a
self-normalised tail mean and ESS.

**The structural gap.** The repository currently contains two disconnected halves. `eq_jepa/*`
produces a gridded forecast file; `methods/*` consumes analytic PSHA targets. **Nothing reads a
JEPA rate field into the CVaR estimator**, and the layers that would join them — a ground-motion
model and a structural model — are absent: a keyword sweep for `sdof`, `oscillator`, `newmark`,
`fragility`, `floatcsep`, `fissler` across the whole tree returns nothing. The missing interface
*is* L2 + L3. That is the single highest-value thing to build, and it is also what makes "nonlinear
SDOF" not a nice-to-have but the load-bearing member (§4).

**Specific defects, ordered by how much they block the chain.**

1. `train.py` writes `forecast.dat` from `next(iter(dl))` on a **shuffled** DataLoader and averages
   rates over the batch's issue times. The emitted forecast is therefore not reproducible and is not
   a forecast for any named window — it cannot enter a CSEP experiment. Needs: one file per
   `(t₀, Δ)`, deterministic, seeded.
2. **No horizon conditioning.** `horizon_days` is a single scalar in the config and the predictor
   takes no `Δ` argument, so the "hierarchy" is spatial/magnitude only. Temporal additivity (I2) is
   not even expressible yet.
3. **No floatCSEP experiment.** The forecast *format* is written; there is no experiment
   configuration, no test invocation, and the baselines are not registered as CSEP forecast models.
   The README's compatibility claim is a prose claim only.
4. **No elicitable score anywhere.** No Fissler–Ziegel loss; `OnlineCVaRTracker` is an estimator,
   not a score. I3 is currently unimplementable.
5. **Anti-collapse rests entirely on EMA + stop-gradient**, with MSE on L2-normalised embeddings, no
   variance/covariance or isotropy term, and no collapse diagnostic. Collapse would currently be
   silent.
6. **Poisson-only count head** — see L0 above.
7. **`issue_times` is a hand-listed pair in the config** — no rolling-origin generation, so no
   time-dependent experiment is possible.
8. The target branch's `future_long_history` is all zeros except the final row, so the target
   encoder's long-history GRU sees a near-degenerate sequence; the asymmetry between context and
   target views is much larger than intended.
9. No magnitude-completeness (`M_c`) handling, no declustering option, and the free softmax over
   magnitude bins is unconstrained by a (tapered) Gutenberg–Richter law.

---

## 4. Pilot: why nonlinear SDOF is the load-bearing member

A small self-contained numerical experiment, run to check whether the nonlinear oscillator is
actually doing work in this architecture or is decoration.

**Setup.** `T₁ = 0.5 s`, `ζ = 5%`, yield strength coefficient `η = F_y/mg = 0.25`; 200 synthetic
records (Kanai–Tajimi-filtered white noise × growth–decay envelope) scaled onto a log-spaced
`Sa(T₁)` grid; hazard = truncated Gutenberg–Richter (`b = 1`, `M ∈ [5,8]`) crossed with a lognormal
ground-motion term (`σ_lnSa = 0.6`); saturating repair-cost function plus a collapse atom;
`α_CVaR = 0.99`. Central-difference integration of exact bilinear kinematic hardening, verified
`dt`-converged (identical results at `dt/4`) and verified against the exact linear identity
`μ_d = Sa/η`.

![Nonlinear SDOF pilot: loss exceedance curves, proposal disagreement map, and post-yield stiffness sweep]({{artifact:art_f9df16b0-ccbb-444c-bf7f-316037b55174}})

### 4.1 The mean is insensitive to hysteresis; the tail is not

| hysteresis | E[Y] | VaR_99 | CVaR_99 | Pr(collapse) |
|---|---|---|---|---|
| linear elastic | 0.0349 | 0.262 | 0.433 | 0 |
| ductile, `α_pv = +0.03` | 0.0359 | 0.294 | 0.542 | 3.1e-7 |
| degrading, `α_pv = −0.05` | 0.0417 | 0.620 | **0.904** | 3.3e-3 |

Going from a linear oscillator to a degrading one raises the **mean loss by 20%** and **CVaR_99 by
109%**, and only the degrading oscillator puts non-negligible probability on collapse. A linear
transfer-function surrogate can therefore be accurate for expected loss and badly wrong for the
quantity this framework is built to report. **The object CVaR measures is created by the degradation
mechanism** — which is the sharpest available justification for putting a nonlinear SDOF, rather
than a linear site-response surrogate, inside the loop.

### 4.2 Nonlinearity moves the tail variance into waveform space

Decomposing `Var_p[(Y − v)⁺]` into the part explained by the hazard variables `(M, ε)` and the part
carried by record-to-record variability at fixed `Sa`:

- **linear elastic: 0% waveform-borne.** For a linear oscillator `μ_d = Sa/η` exactly, so `Sa` is a
  sufficient statistic, the waveform is irrelevant, and a proposal that tilts only `(M, ε)` is
  *exactly* zero-variance.
- **nonlinear: 24–33% waveform-borne** across `α_pv ∈ [+0.05, −0.08]` — 25.8% at `α_pv = +0.03`,
  29.8% at `α_pv = −0.05`.

This is a hard variance floor: **no importance-sampling scheme confined to the hazard variables can
reduce that third of the variance**, because it lives in a coordinate that scalar-IM PSHA does not
have. It is the quantitative reason the ground-motion generator (L2) must be *inside* the sampling
loop with its latent `z_gm` exposed to the proposal, rather than being a fixed upstream
record-selection step. This is the orange feedback loop in the architecture figure.

### 4.3 Disaggregation-optimal is not tail-optimal — and the disagreement hides in ε

Three distinct proposals, all of the form `q ∝ p · w(M, ε)`:

    q_disagg   ∝ p · Pr(Y > v | M, ε)                  ← what PSHA disaggregation reports
    q_1stmom   ∝ p · E[(Y − v)⁺ | M, ε]
    q_opt      ∝ p · sqrt( E[((Y − v)⁺)² | M, ε] )     ← variance-optimal in this family

In the **full** sampling space the zero-variance proposal for an exceedance probability is
`p · 1{Y > v}`, while for a tail mean it is `p · (Y − v)⁺`: PSHA disaggregation and CVaR-optimal
sampling are provably different objects, and only the latter is zero-variance for CVaR. Inside a
**restricted** family that can tilt only `(M, ε)` — all a scalar-IM formulation can do — the
variance-minimising tilt is neither of those two; it is `p · sqrt(E[h²|·])`, a *third* object.
Measured variance penalty for using the disaggregation tilt instead: 1.02–1.38× depending on
`α_pv`. The naive first-moment tilt is also suboptimal here (7.65e-6 vs 5.53e-6 at
`α_pv = +0.03`).

The more important finding is *where* they disagree. The magnitude marginals of `q_disagg` and
`q_opt` are nearly identical — so a disaggregation plot, which is exactly an `M`–`R`–`ε` marginal
summary, will not reveal the problem. The density ratio `q_opt / q_disagg` spans a factor of **3.4**
across the cells carrying 99% of the optimal mass, and that disagreement is concentrated in a narrow
band in `ε` along the **collapse-onset contour** (`Sa = 0.74 g` here; panel b of the figure). The
tail-optimal sampler wants the scenarios that are *just* capable of triggering collapse — a set
defined by the structure's capacity, not by the hazard.

### 4.4 Caveats

Synthetic records, SDOF only, one period, one hazard configuration, an invented cost function, and
`α_CVaR = 0.99` fixed. The numbers are indicative of mechanism, not predictive of any real asset.
The sign and order of magnitude of the three effects are what should be carried forward; each
becomes a hypothesis to re-test with real records and an MDOF model (§5).

Underlying numbers: `pilot_results.csv` (per-hysteresis risk measures and variance decomposition)
and `pilot_sweep.csv` (the `α_pv` sweep).

---

## 5. Falsifiable hypotheses

**H1 — measured (§4.1).** The tail of the loss functional is generated by the hysteresis, not by the
loss function: replacing a degrading oscillator with a linear one changes `E[Y]` by order 20% but
`CVaR_99` by order 100%. *Test:* re-run with real records and an MDOF / IMK model; compare against a
published IDA curve.

**H2 — measured (§4.2).** Any IS proposal confined to `(M, ε)` has a variance floor of 20–35% of the
CVaR-relevant variance for a nonlinear asset, and 0% for a linear one. *Test:* implement a tilt over
`z_gm` and check whether the floor is broken. If it is not, L2 can stay outside the loop and the
architecture simplifies.

**H3 — measured (§4.3).** Disaggregation-optimal ≠ tail-optimal, and the disagreement is invisible
in the `M` marginal. *Test:* compare against the repo's existing analytic Houng et al. targets,
where both tilts are computable in closed form.

**H4 — untested.** floatCSEP consistency tests detect JEPA representation collapse: a collapsed
encoder passes the N-test but fails the S- and CL-tests. *Test:* deliberately collapse the encoder
(drop the isotropy term, or freeze the predictor) and run the test battery. A negative result would
mean the geometric diagnostics are not optional.

**H5 — untested.** CVaR-constrained action selection differs from expected-loss selection only when
`Pr(collapse | a)` differs across actions by more than `1 − α`. *Test:* synthetic oracle at L5 with
a known optimum.

---

## 6. Build order

Forecast-first, as the brainstorming note argues; the decision layer is last and gets a synthetic
oracle before it gets real data.

1. **M1 — make L0 scoreable.** Horizon-conditioned predictor `P_φ(z_t, Δ)`; negative-binomial count
   head; rolling issue times; deterministic per-window forecast writer (one file per `(t₀, Δ)`,
   seeded). *Deliverable:* a forecast bundle floatCSEP can ingest.
2. **M2 — stand up L1.** floatCSEP experiment with the two existing baselines plus a
   smoothed-seismicity reference; N/S/M/CL plus paired T-tests over a rolling origin.
   *Deliverable:* the first honest scoreboard for the JEPA forecast.
3. **M3 — close the collapse hole.** Add the isotropy / variance–covariance regulariser and the
   geometric diagnostics (embedding rank, per-dimension std, sketched-projection Gaussianity); run
   H4.
4. **M4 — build L3.** Nonlinear SDOF with cyclic degradation (the pilot integrator, extended to
   peak-oriented hysteresis), damage→loss with an explicit collapse atom, validated against a
   published IDA curve. *Deliverable:* the first end-to-end loss sample from a certified rate field.
5. **M5 — build L2.** Conditional record selection first (`z_gm` = record index), generative
   waveform model second. Run H2 at this point — its answer decides whether L2 stays in the loop.
6. **M6 — close L4.** Joint `(VaR, CVaR)` estimation with Fissler–Ziegel scoring; compare prior /
   disaggregation / learned tail-optimal proposals over `(M, ε, z_gm)`; report tail-ESS and bootstrap
   intervals. This is where the existing G-PMC/AIS code plugs in.
7. **M7 — L5.** Constrained MPC on a synthetic oracle, validated against the known optimum; RL only
   after that.

**Interface contracts** — the arrows in the figure, as data:

| from → to | object passed |
|---|---|
| L0 → L1 | `λ(c, m, Δ \| t₀)`, CSEP gridded format, one file per `(t₀, Δ)` |
| L1 → L2 | certified `λ` plus the `(c, m)` bins carrying the mass |
| L2 → L3 | waveform ensemble with weights; `z_gm` recorded per sample |
| L3 → L4 | `(ω, Y(ω))` pairs, `ω = (t₀, c, m, ε or z_gm)` |
| L4 → L5 | `(v_α, c_α)` plus the weighted tail-scenario set |
| L4 → L2 | proposal tilt over `z_gm` (dashed, orange) |
| L5 → L3 | capacity parameters `θ_a` (dashed, purple) |
| L1 → L0 | reject / retrain signal (dashed, green) |

---

## 7. Where this sits relative to the deck

The perspectives chapter asks for a continuously updated system that infers a partially observed
physical state, says when that state is uncertain, and acts. The chain above is a minimal,
*scoreable* instance of that ask, with three deliberate narrowings: the world model is confined to
seismicity (L0) so that it has a ground truth to be scored against; the agentic/orchestration layer
is deferred entirely, because an orchestrator that selects among uncertified tools inherits their
errors; and the "action" side is a constrained controller rather than a learned policy, so the
decision layer can be validated against a known optimum before it is trusted.

The deck's own framing of representational *crystallisation* as a failure mode that a continuously
updated twin must actively resist is, in this chain, not a metaphor but a testable quantity: it is
exactly what H4 measures, and floatCSEP is the instrument.

---

## 8. Update — M1/M2 executed, with two catalog bugs found and fixed

Since §3 was written, M1 and M2 were carried out end-to-end on a real California catalog
(USGS/ANSS, 2019–2024, 71 rolling monthly windows), closing defects §3.1, §3.3, §3.6 (partially),
and §3.7. This section reports what that run actually found, because two of the defects it
surfaced were not on the original list and change how §3's remaining items should be read.

**Bug A — aftershock contamination.** The independent-events filter was computed but never applied
before scoring, so floatCSEP was scored against the raw, clustered catalog. This alone produced
monthly-count dispersion of **379.5× Poisson** — a variance regime no count-head fix (negative
binomial or otherwise) can absorb, because it is a data-pipeline defect, not a model-misspecification
one.

**Bug B — declustering formula error.** Filtering to independent events used a hand-coded
Gardner–Knopoff window formula with swapped space/time coefficients for `M ≥ 6.5`. This fabricated
multi-year gaps of zero independent events that read as real seismic quiescence — the residual
dispersion after this "fix" was still **10.5× Poisson**, barely improved over Bug A, and a
Hawkes-like rate correction fit against it was chasing an artifact. Re-declustering with the vetted
`seismostats` port of OpenQuake HMTK's reviewed Gardner–Knopoff implementation collapsed dispersion
to **0.935× Poisson** — essentially the Poisson regime the L0 count head already assumes (a
negative-binomial head, defect §3.6, would only have mattered if this residual variance had
survived the catalog fix; it does not, at the monthly aggregation level used here).

**What the corrected L1 scoreboard shows.** With both bugs fixed, the universal N-test failure
originally reported is **not** an overdispersion problem — it is a genuine mean-rate deficit shared
by all three models (observed mean 16.8 events/window; L0 forecasts 4.8, GR 5.2, ETAS-lite 7.5).
A minimal, non-architectural fix — the 2-parameter Hawkes-like multiplicative boost from §L0/L4's
recent-activity conditioning, fit post-hoc by Poisson-likelihood MLE (amplitude 0.20, decay
τ = 5.1 days) — raises L0's N-test pass rate from 5.6% to 53.5% and its comparative T-test standing
from indistinguishable-or-behind both baselines to **clearly ahead in roughly a third of windows**
(24 better / 45 indistinguishable / 2 worse vs. ETAS-lite; 25/42/4 vs. GR), with very few losses.
M-test (spatial/magnitude shape) was strong throughout for all variants — the relative
spatial-magnitude structure was never the problem.

**Reading this against H4 and the build order.** This run did not execute H4 as designed (a
deliberate collapse ablation) — it is a data-correctness finding, not a representation-geometry
one — so H4 remains untested. It does sharpen the standing rule in §I4 ("every layer is validated
against data ... before the next layer consumes it"): the failure caught here was upstream of L0
entirely, in the catalog-preparation step that L0's training and L1's scoring both depend on, and it
survived one full round of "fix and re-score" before being caught on inspection of the diagnostic
numbers themselves (dispersion staying at 10.5× after a fix nominally meant to remove it). The
practical corollary: a diagnostic that doesn't move after a fix is targeted at it is evidence the
fix didn't address the mechanism, not evidence the mechanism was minor.

**Caveat carried forward, per §3's own standard.** The rate correction above was fit and applied to
L0 only; the T-test comparisons in this section therefore compare a rate-corrected L0 against
*uncorrected* GR/ETAS-lite baselines. Whether GR and ETAS-lite's own ~3× rate deficits are fixable
artifacts (in which case the comparison should be redone symmetrically) or a genuine skill
difference is open, and should be resolved before this section's T-test counts are treated as a
final ranking.
