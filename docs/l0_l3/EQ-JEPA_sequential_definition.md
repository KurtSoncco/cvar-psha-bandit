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

---

## 9. Extension — a continuous marked point process for L0, with depth

Prompted by a concrete gap found on inspection, not a hypothetical one: the gridded forecast
`.dat` files L0 writes already carry a depth-bin column pair (`z_0, z_1`, e.g. `0 700` — "somewhere
in the crust"), but pycsep's own gridded-forecast loader (`GriddedForecast.load_ascii`) builds its
spatial region from `data[:, :4]` — lon/lat only. The `z_0, z_1` columns are parsed off the line and
never used again. Depth is not under-resolved; it is **written as a placeholder and then silently
discarded** by the evaluation path currently in use. This section proposes replacing L0's output
representation to fix that, and traces the consequence through L1–L3.

### 9.1 Redefinition: L0 emits a continuous intensity, not a grid-and-bin rate

**Current.** `λ(c, m, Δ | t₀)` — a rate per (cell, magnitude-bin), via a softmax allocation head
`π(c, m | z_t, Δ)` over a fixed discrete grid (§2, L0).

**Proposed (L0′).** A continuous conditional intensity over exact position and depth:

    λ(t, x, y, z, m | z_t, Δ)        x, y continuous (lon/lat or projected coords), z = depth (km)

realized as a decoder conditioned on the same latent `z_t` that already exists — not a new encoder
or predictor. Candidate decoders, in order of how much they disturb the existing architecture:

1. **Mixture density network** — a mixture of anisotropic 3-D Gaussians/Student-t kernels in
   `(x,y,z)`, with magnitude-conditional mixture weights, read off `z_t`. Smallest delta: encoder,
   predictor, and EMA target branch are untouched; only the rate head changes.
2. **Normalizing flow** over `(x,y,z,m) | z_t` — more expressive, still an exactly-integrable
   density.
3. **Neural spatio-temporal point process / diffusion decoder** — samples catalogs directly rather
   than outputting a density. This is the natural home for the existing Hawkes-like rate correction
   (§8) to stop being a bolted-on post-hoc patch and become a *learned* part of the decoder.

**A free consequence for I2.** Exact additivity (`λ_parent = Σ_children λ_child`) is currently
*engineered* via a softmax allocation head specifically to avoid a soft penalty that could leave the
model incoherent in the tail. A continuous intensity gets this invariant for free: integration of a
density over any sub-region or sub-interval is automatically consistent with the integral over the
whole. I2 becomes a property of calculus rather than a head design choice.

### 9.2 L1: the validation track already exists in the installed library — mostly

pycsep ships `csep.core.forecasts.CatalogForecast` (an ensemble of `n_cat` simulated catalog
realizations, each with real per-event lon/lat/depth/magnitude) and `csep.core.catalog_evaluations`
with `number_test`, `spatial_test`, `magnitude_test`, `pseudolikelihood_test`, `calibration_test` —
the catalog-based analogs of N/S/M/CL, built for exactly a model that emits stochastic event sets
rather than a fixed rate grid. floatCSEP already wraps pycsep, so this is a forecast-writer and
config change, not new infrastructure (`forecast_type: catalog` instead of `gridded`).

**The gap, checked rather than assumed.** Even the catalog-based `spatial_test` re-grids internally
(`catalog.spatial_counts()` onto a 2-D `CartesianGrid2D`) to score spatial fit — **no
depth-consistency test exists anywhere in pycsep.** Given this project's history of two independent,
stacked bugs inside a *hand-rolled* geometric formula (§8, Gardner-Knopoff), the recommendation is
specific: build the depth test (call it **D-test**) by reusing pycsep's existing simulate-and-rank
pattern (`_simulate_catalog`, then rank the observed depth histogram/KDE against the same 1000
simulated catalogs already being generated for S/M/CL) rather than deriving a new depth-binned
likelihood from first principles.

### 9.3 L2: exact depth is not cosmetic here — it is a lever on an already-open gap

With real per-event depth instead of a fixed assumption, `R_hyp = √(R_epi² + z²)` replaces whatever
implicit nominal-depth stand-in the point-source simulator uses today — mattering most exactly where
it matters most for risk (shallow, near-site events). It is also a direct, testable lever on the one
quantitative gap already on record: the physics-only L2 residual (σ_lnSa ≈ 0.27–0.34) sits at
roughly half the empirical NGA-West2 total sigma (≈0.5–0.7), attributed to randomizing only
source/path/site *parameters*, not additional unmodeled variability. Real per-event depth variability
was structurally unavailable before §9.1; feeding it through is a new, falsifiable sub-hypothesis
(H7 below), not a guess that it will help.

### 9.4 L3 / portfolio: what exact locations actually unlock

The SDOF physics is 0-D and indifferent to location — but what it *receives* is not, and a
0.5°×0.5° grid cell (~50×50 km) currently forces every asset inside it toward identical treatment.
Exact hypocenter locations make a true spatial portfolio possible: each asset at its own coordinates,
its own `R_hyp`, with ground-motion residuals spatially correlated across sites (e.g. a Jayaram &
Baker-style empirical correlation model for `ε`). This is not optional machinery for later — **a
portfolio CVaR is undefined without it**, since tail risk depends entirely on whether nearby assets
fail together or independently, and a shared grid cell currently forces that answer by construction
rather than by data.

### 9.5 New falsifiable hypotheses

**H6 — untested.** A continuous mixture-density (or flow) decoder conditioned on `z_t` achieves S/M/CL-equivalent
calibration (via the catalog-based D/S/M tests, §9.2) at least as well as the discrete softmax head,
while removing grid-discretization error at cell boundaries. *Test:* train both heads on identical
data/latent, compare catalog-based test pass rates and spatial log-likelihood.

**H7 — untested.** Feeding real per-event depth into L2 (replacing the fixed-depth placeholder)
closes part of the σ_lnSa gap between the physics-only simulator (≈0.27–0.34) and empirical NGA-West2
total sigma (≈0.5–0.7). *Test:* rerun the existing L2 validation figure with depth-varying vs.
fixed-depth inputs, holding source/path/site randomization identical, and compare residual variance
directly.

**H8 — untested.** Spatial correlation of ground-motion residuals across a multi-site portfolio
(enabled only once exact locations replace grid cells) changes portfolio CVaR_99 by more than the
single-asset pilot's hysteresis effect (§4.1, +109%). *Test:* synthetic 2–3 asset portfolio at known
separation distances, compare correlated vs. independent-residual CVaR.

### 9.6 Build order addition

8. **M8 — swap L0's rate head.** Mixture-density decoder over `(x,y,z,m) | z_t`, trained against the
   same catalog and latent pipeline already in place. *Deliverable:* a catalog-sampling forecast
   (`CatalogForecast`-compatible) alongside the existing gridded one, for a direct head-to-head.
9. **M9 — stand up catalog-based L1.** `floatcsep` config switched to `forecast_type: catalog`;
   wire pycsep's `number_test`/`spatial_test`/`magnitude_test`/`pseudolikelihood_test`; implement the
   D-test (§9.2) by extending `_simulate_catalog`'s pattern to depth. *Deliverable:* first honest
   scoreboard for the point-process forecast, run H6.
10. **M10 — depth into L2.** Replace the fixed-depth placeholder in the stochastic point-source
    simulator with real per-event depth from the (now depth-resolved) catalog; rerun the σ_lnSa
    validation. *Deliverable:* test H7; if confirmed, a quantified, closed fraction of the
    empirical-sigma gap.

### 9.7 Further idealization improvements noted alongside this redesign, by layer

**L1.** A flat `M_c = 2.8` completeness cutoff over 2000–2024 is itself an idealization — real
network density changed over that span, so a time-varying `M_c(t)` surface would be more honest and
would change which early-period events enter declustering and scoring. The original plan's named
declustering method (Zaliapin–Ben-Zion, nearest-neighbor) operates natively in continuous
space-time-magnitude and would pair more naturally with L0′ than Gardner-Knopoff's window method
does — worth revisiting now that the representation is moving to continuous anyway. The standing
caveat that GR/ETAS-lite have not received the same Hawkes-like rate correction as L0 (§8) remains
open and is cheaper to close first, independent of this redesign.

**L2.** Report the GMPE residual as its standard three-component partition (inter-event `τ`,
inter-site `φ_s2s`, single-station `φ_ss`) instead of one aggregate `σ_lnSa`, so the empirical-vs-
physics gap is testable mechanism-by-mechanism rather than as one lumped number. Site/path
parameters are currently drawn from one nationwide distribution; USGS Vs30 grids (an already-reachable
science API) would let site terms be regionally conditioned instead.

**L3.** The pilot uses one oscillator (`T₁=0.5s`, `η=0.25`); a small suite of representative
building-class oscillators spanning a real portfolio turns L3 from an illustrative case into
something that can answer portfolio questions. Caveat 3 from the verification report — SDOF
validated against closed-form/qualitative IDA shape, not a specific published curve for a named
building class — remains open. A clean, separate falsifiable question worth adding alongside H1:
compare SDOF-based tail risk against a cheap 3-DOF shear-building model for identical input motions,
to isolate how much of CVaR is attributable to the single-degree-of-freedom idealization itself,
as distinct from the hysteresis mechanism H1 already measured.

## 10. Update — the next iteration executed: ten open defects addressed, and the target itself indicted

Everything in this section was run on the corrected catalog with the trained checkpoint from §8,
scored with `csep.core.poisson_evaluations` and `csep.core.catalog_evaluations` directly (the same
functions floatCSEP's `tests.yml` wraps) over the same 71 monthly test windows, 2019-01 to 2024-11.
Where a correction is fitted, it is fitted on the 108 pre-2019 windows and frozen before scoring.

### 10.1 Two of the previous section's headline numbers were measuring the plumbing

Regenerating forecasts one decode per issue time, at each window's own horizon, instead of through
`train.py`'s `write_forecast` (which averages rates over a shuffled-`DataLoader` batch mixing the
1/7/30-day horizons) reverses the sign of the rate error: L0 forecasts a mean of **58.8 events per
window against 17.1 observed**, an over-forecast of 3.4x, not the 3.5x under-forecast reported in
§8. The ratio is the declustering ratio. L0 was trained on the aftershock-inclusive catalog (19,152
events) and scored against the declustered independent catalog (5,491); 5491/19152 = 0.287, and the
Poisson-MLE constant lands at `c0 = 0.326`.

Generalising the correction from the two-parameter Hawkes form to

        mu_w = c0 * Lambda_w * (1 + amp * feat_w(tau))

with `c0` profiled out analytically (a multiplicative scale has a closed-form Poisson MLE) makes the
**excitation amplitude fit to zero for all three models**, with log-likelihood identical to
scale-only. The §8 "Hawkes-like rate correction" was fitting a constant. The open §8 caveat — that
the baselines had not received the same correction — is now closed by applying this one correction
symmetrically, and the answer is that symmetry does not rescue L0:

| model, rescaled on pre-2019 windows | N | S | M | CL |
|---|---|---|---|---|
| EQ-JEPA L0 (original checkpoint) | 85.9% | **8.5%** | 83.1% | **7.0%** |
| HistoricalRateGR | 95.8% | 97.2% | 93.0% | 83.1% |
| ETASLite | 90.1% | 98.6% | 81.7% | 98.6% |

L0 fails the **spatial** test in 91% of windows, and since the S-test rescales to the observed total
this is not a rate artifact. Paired T-tests: L0 beats HistoricalRateGR in **0 of 71** windows
(37 tied, 34 worse, mean information gain −0.52 per event) and ETASLite in 1 of 71. **H1 is refuted
on this run.**

![Symmetric rescaling and consistency scoring]({{artifact:art_3ef7c913-9fa0-48bc-8c7b-7b2133ec5070}})

### 10.2 The encoder was collapsed, and the input embedding — not the missing isotropy term — was why

Geometric diagnostics on the encoder output that L1–L4 actually consume (not a projector output;
per arXiv:2609.35288 a backbone can collapse while a projector-level objective looks healthy),
over 179 issue times:

- **participation ratio 1.09 of 128**; PC1 carries 95.5% of the variance, three PCs carry 99%
- mean pairwise cosine between issue times **0.958**, 42% of pairs above 0.99
- **no dead units** (per-dimension std 0.012–0.87) — the collapse is directional, so a
  per-dimension-std check alone would have missed it
- 85% of 512 sketched 1-D projections reject normality (median KS p = 2.3e-4)

The behavioural control settles it: replacing the latent with a constant (the pre-2019 mean) and
rescaling identically gives N 95.8 / S 9.9 / M 81.7 / CL 8.5 percent against the trained encoder's
85.9 / 8.5 / 83.1 / 7.0. **A frozen latent is as good as or better than the trained one on every
test.** The predicted collapse signature from §3 (N passes while S and CL fail) is confirmed, and
the stronger statement holds: the trained encoder contributes nothing measurable to the forecast.

![Latent collapse diagnostics and the frozen-encoder control]({{artifact:art_6ef6e1fe-3758-40d6-94d1-03dd024eeae5}})

A 2x2 ablation — {original, fixed} event embedding x {no, with} a LeJEPA-style SIGReg isotropy
penalty (arXiv:2511.08544; Epps-Pulley closed form, weight set by gradient-norm matching at
initialisation, not tuned) — all four arms retrained on declustered counts, isolates the cause:

| arm | participation ratio | mean cosine | Gaussianity rejected | N | S | M | CL |
|---|---|---|---|---|---|---|---|
| A original embedding | 1.13 | 0.91 | 78% | 94 | 89 | 86 | 49 |
| B original + SIGReg | 1.18 | 0.82 | 82% | 92 | 89 | 86 | 54 |
| C **fixed embedding** | **22.4** | **0.28** | **0%** | 89 | 90 | 90 | **76** |
| D fixed + SIGReg | 22.7 | 0.27 | 0% | 89 | 87 | 90 | 72 |

Three readings. First, **retraining the unchanged architecture on the declustered target takes the
S-test from 8.5% to 88.7%** — the catastrophic spatial failure was also the train/score catalog
mismatch, not a broken spatial head. Second, the §9-agreed encoder fix (per-feature normalisation,
Fourier features for lat/lon/depth, Time2Vec for `dt`, log-compressed long-history channels) is what
de-collapses the latent, and it buys real joint-likelihood quality (CL 49% → 76%). Third, **SIGReg
cannot rescue the original embedding** (1.13 → 1.18) and adds nothing once the embedding is fixed:
the binding constraint on collapse was the input representation, not the absence of a regulariser.
This vindicates the §9 decision to make the fix symmetric rather than decoder-only.

One tension to carry: the de-collapsed arms have better CL pass rates but *worse* paired T-tests
(0/40/31 at −0.48 versus arm A's 0/65/6 at −0.25). CL is self-consistency; the T-test is absolute
likelihood against a competitor. A sharper forecast can be consistent with its own uncertainty while
assigning less probability to the events that actually occurred than a smooth historical map does.

![The 2x2 ablation]({{artifact:art_53da9b18-f2d4-4183-8928-6d12516c7fc4}})

### 10.3 The point-process redesign (M8–M10): the magnitude side works, the spatial side does not

`model_pp.py` implements §9.1 at its least disruptive setting: the same JEPA encoder (fixed
embedding), and a decoder emitting

        lambda(t, x, y, z, m | z_t, Delta)
            = N(z_t, Delta) * f_t(t) * f_xy(x, y) * f_z(z) * f_m(m)

with every factor closed-form and exactly normalised over the modelling domain — a 24-component
axis-aligned Gaussian mixture in (lon, lat, depth), a truncated Gutenberg-Richter magnitude law with
learned `b` and a learned global upper bound, and `f_t` uniform within the window. Training is the
inhomogeneous marked-Poisson log-likelihood of the real event coordinates: no grid, no binning.
It self-calibrates (mean 18.0 events per window against 17.1 observed, `c0 = 0.97`).

**What it fixes — the moment budget becomes well-posed.** Implied moment rate **4.14e18 N·m/yr,
0.38x the observed release**, against 4x to 60x for every gridded forecast (§10.4). A truncated GR
with a learned `Mmax = 7.45` has finite moment content where an open [5.5, 8.5] bin does not.
Learned `b = 1.00` against the catalog's 0.84 is why it lands below rather than above.

**What it breaks — the spatial field.** On the gridded battery, S drops to 9.9% (against 90.1% for
the gridded head on the same encoder) and CL to 7.0%. On pycsep's catalog track: number 89%,
magnitude 96%, pseudolikelihood 75%, **spatial 31%** with median quantile 0.994 and a
calibration-test quantile of 2e-46. Both tracks say the forecast is too diffuse, and the mean
intensity map shows why: axis-aligned components produce cross-shaped streaks where California has
fault-aligned lineaments. **The fix direction is a spatial family that can be anisotropic** — full
covariance components, a normalising flow, or a continuous intensity modulated by a fixed
high-resolution smoothed-seismicity prior — not more components. H6 is therefore **not supported as
implemented**, with the failure localised to the spatial factor rather than to the point-process
formulation.

![Point-process forecast on both test tracks]({{artifact:art_a5229f22-67de-4cfe-a1e8-1762c624905e}})

### 10.4 Energy: the useful forms, and why the open top bin breaks the rest

Converting magnitude to energy adds no information — `log10 E_s = 1.5 M + const` is a bijection, so
a forecast of converted energy is a forecast of magnitude pushed through a monotone transform. Worse,
summing energy is effectively taking a maximum: on this catalog the single largest event carries a
median **42%** of a monthly window's radiated energy (**50%** annually), the largest M7.2 carries 41%
of the whole 2000–2024 total, the ten largest carry 88%, and window log-energy correlates 0.98 with
the window's maximum magnitude against 0.75 for log-count.

![Window energy tracks the largest event, not the count]({{artifact:art_75221946-9f78-4f9f-8372-db4faf63598c}})

What survives is energy as a **weighting and a constraint**, not a target:

- **Moment-rate budget.** Observed release is 1.08e19 N·m/yr (declustered) and 1.26e19 (full
  catalog). Every gridded forecast overshoots by 4x (L0) to 60x (ETASLite), almost entirely through
  the open top magnitude bin, whose Kostrov integral is dominated by its upper edge because
  `M0 ~ 10^(1.5m)` grows faster than `10^(-bm)` decays for `b < 1.5`. Truncating that bin at M7.5
  drops the implied rate 4–5x; `b = 1.0` halves it. **A gridded forecast with an open top bin has no
  well-posed moment content** — an independent argument for §9.1 that does not depend on depth.
  Note also that the M-test passes for every model while the moment budget is off by an order of
  magnitude: the two diagnostics are not redundant.
- **An E-test** (energy analog of the N-test, simulate-and-rank over 1000 synthetic catalogs with
  within-bin GR magnitudes) is implemented but has little discriminating power: pass rates 70–92%
  because the simulated energy distribution is dominated by whether a top-bin event is drawn. It does
  catch models that over-allocate to large magnitudes (21–30% of windows below the 2.5% quantile for
  the fixed-embedding arms and ETASLite).
- **An energy channel as an auxiliary target** remains untried and is the one form still worth
  building, using measured `Me`/`M0` rather than converted magnitude.

![Moment-rate budget and the E-test]({{artifact:art_9137d256-70e7-47f8-b58c-4054729b59a8}})

### 10.5 Depth: a working D-test, and the first quantified depth lever on CVaR

No depth test exists in pycsep, so one was written on pycsep's own simulate-and-rank pattern:
bin depth, take the forecast's depth distribution from its simulated catalogs, score the observed
depth histogram by Poisson joint log-likelihood against the forecast's expected histogram rescaled
to the observed count (exactly how the M-test treats magnitude), and rank that score against 1000
synthetic depth histograms from the same forecast. The point-process L0 **passes in 91.5% of
windows** (median quantile 0.32); forecast median depth 7.3 km against 8.1 km observed, though the
forecast's depth distribution barely varies window to window (IQR 0.5 km against 1.6 km observed).

Propagating real depths through L2–L4 for a Los Angeles site: using `R_hyp = sqrt(R_epi^2 + z^2)`
instead of `R_epi` cuts **CVaR_0.99 of loss from 0.0120 to 0.0080, a 33% reduction**, and mean loss
by 26%. Equivalently, treating hypocentres as surface sources inflates the tail by **49%**. The
effect is entirely near-field — only 4% of forecast events fall within 50 km, and those set the tail.
This is a **lower bound**: the ductility map used is the linear limit `mu_d = Sa/eta`, and §4.1 found
a degrading oscillator roughly doubles CVaR_0.99. H7 is therefore supported in the sense that depth
is a material lever, though it moves the tail by changing distance rather than by closing the
sigma gap.

![Depth test and the depth sensitivity of CVaR]({{artifact:art_c328d88e-3a3b-491c-904f-27eb13345c01}})

### 10.6 What actually moves the tail

Three levers, measured on the same site and the same forecast:

| lever | change in CVaR_0.99 |
|---|---|
| ground-motion sigma 0.33 → 0.70 (to the empirical range) | **+46%** |
| depth ignored (hypocentres treated as surface sources) | **+49%** |
| count dispersion, NB `r = 0.25` against Poisson | +7% |

The count-dispersion result is worth stating plainly, because L0 carries a negative-binomial
dispersion head whose purpose was tail risk. Of 1,760 cell/magnitude bins only 74 have 5 or more
observed events; of those, 57% fit at the Poisson limit and 5.4% reject Poisson — the nominal
false-positive rate — while the head predicts median `r = 2.0`. The head over-disperses against a
target that is Poisson. The aggregate window-total likelihood ratio does reject Poisson
(`r = 22.4`, p = 1e-4) but observed variance/mean is 0.92: that signal is the forecast's own
uncorrelated rate fluctuation being absorbed by the dispersion parameter. And it does not matter
anyway — the loss tail is mark-driven, set by *which* event occurs rather than how many.

The L2 sigma gap is the lever that does matter and cannot be closed here: the within-scenario
`sigma_ln(Sa)` is 0.331, so reaching the empirical total range 0.5–0.7 requires a missing
0.375–0.617 in quadrature from terms this physics randomisation does not touch (directivity, basin
response, model form). Documented, not closed; strong-motion records remain unavailable.

![Dispersion, sigma, and what moves the tail]({{artifact:art_c351fe3d-e95b-4aca-b8a4-7b6643a38c8a}})

### 10.7 The finding that reframes L0: the target has no monthly memory

The diagnostics that would have justified an H-JEPA-style multi-level temporal hierarchy
(arXiv:2610.06805) came back clean, so the hierarchy was not built:

- **Temporal additivity already holds.** Decoding one latent at 1, 7, 30, 90 and 365 days gives an
  implied daily rate of 0.583 at every horizon (max/min 1.046). Summing three consecutive one-month
  forecasts against a single three-month decode: median residual −0.6%, IQR ±10%. The time half of
  invariant I2 is satisfied empirically, not only by construction.
- **A conditional-mean decoder is not the limitation.** Randomized PIT values of observed counts
  under Poisson(total) are uniform (KS p = 0.58) with observed variance/mean 0.91. The objection that
  one-step latent regression identifies a mean rather than a transition kernel (arXiv:2609.36227)
  does not bite for window counts on a declustered catalog. The learned horizon correction does
  nothing for the same reason: declustering removes the Omori decay it was designed to capture.

And then the target itself: observed monthly counts have **lag-1 autocorrelation −0.08**, variance/mean
**0.92**, one of twelve ACF lags outside the 95% envelope (chance), and **no model has positive
correlation with them** — point-process L0 −0.05, gridded L0 −0.09, HistoricalRateGR −0.16,
ETASLite −0.17. At 90 and 365 days the level is right to 4–7% and the correlation is still zero.

![Horizon coherence, predictive calibration, and the absence of monthly memory]({{artifact:art_15edd77e-c6bf-4cf3-946b-506d76d73f48}})

This is not a California quirk. Across four regions from USGS ComCat, two magnitude thresholds and
three window lengths, declustered with the same vetted Gardner-Knopoff implementation:

| region | variance/mean, raw → declustered | out-of-sample Poisson log-lik gain per event, raw → declustered |
|---|---|---|
| California M2.8+ | 173 → 0.98 | −0.002 → 0.0001 |
| Japan M4.0+ | 443 → 1.78 | 0.017 → 0.006 |
| Chile M4.0+ | 98 → 1.60 | 0.047 → 0.005 |
| Italy–Greece M3.5+ | 36 → 5.99 | 0.221 → 0.212 |

In every raw catalog an exponential-decay excitation feature buys real out-of-sample likelihood;
after declustering the gain collapses to near zero and dispersion collapses to Poisson. The exception
is instructive — Italy–Greece M3.5+ retains variance/mean 6.0, lag-1 ACF 0.81 and nearly all its
predictability, i.e. Gardner-Knopoff's fixed windows do not remove swarm-type clustering there.

**Declustering removes precisely the signal L0 is being asked to learn.** H1 cannot be supported on
this target by any method, because after declustering and monthly aggregation there is nothing left
to predict beyond the rate. This is a target-definition result, not an architecture result.

![Where predictable signal lives]({{artifact:art_6509467a-c2e5-448d-93ba-3557ac42d8dc}})

### 10.8 Revised hypotheses

- **H1** (JEPA representation beats statistical baselines on L1 consistency) — **refuted on the
  declustered monthly target**, and now known to be unfalsifiable there: the target is Poisson.
  Retire it in this form; restate on a target that carries signal (§10.9, H9).
- **H6** (a continuous intensity passes the catalog-based track where the gridded form cannot) —
  **not supported as implemented**. The failure is localised to the spatial factor; the magnitude
  and depth factors pass.
- **H7** (real depth closes part of the L2 sigma gap) — **partially addressed**. Depth is a material
  lever on the tail (49%) but acts through distance, not through sigma; the sigma gap is unchanged and
  is independently worth 46%.
- **H8** (spatially correlated residuals across a portfolio change CVaR by more than the
  single-asset hysteresis effect) — **not tested**; unchanged.
- **H9 (new).** On the *undeclustered* catalog at sub-week horizons, a JEPA latent beats a fitted
  ETAS-style excitation feature in out-of-sample Poisson log-likelihood. The §10.7 measurements give
  the baseline to beat per region (0.017–0.22 nats per event) and the null (near zero after
  declustering), so this is falsifiable before any model is built.
- **H10 (new).** An anisotropic spatial decoder (full-covariance components or a normalising flow)
  recovers the gridded head's S-test pass rate while keeping the point-process form's well-posed
  moment content. Test: swap only the spatial factor in `model_pp.py`, hold everything else fixed,
  rescore both tracks.

### 10.9 Revised build order

11. **M11 — move the target before touching the architecture.** Define L0's target on the
    undeclustered catalog at 1-, 3- and 7-day horizons, with the excitation-feature benchmark of
    §10.7 as the baseline to beat. Deliverable: a target specification plus the per-region
    predictability table that makes H9 falsifiable. This precedes everything else.
12. **M12 — anisotropic spatial decoder.** Replace only `f_xy` in `model_pp.py` (full-covariance
    mixture first, normalising flow second) and rescore both tracks. Deliverable: H10.
13. **M13 — moment-rate constraint as a training term.** Penalise the Kostrov-implied moment rate
    against a regional budget during training rather than checking it afterwards; report the
    forecast's moment rate alongside N/S/M/CL as standard.
14. **M14 — close the L2 sigma gap or bound it.** Partition the residual into inter-event,
    inter-site and single-station components (§9.7) and, failing access to records, carry the
    0.375–0.617 missing-sigma interval explicitly into every CVaR the pipeline reports.
15. **M15 — retire the NB dispersion head or justify it on a target that is overdispersed.** On the
    declustered target it over-disperses and does not move the tail; on the undeclustered target of
    M11 it would have real work to do.

### 10.10 Status of the §9.7 idealization list

Addressed: the symmetric-baseline caveat (§10.1, closed), the open-top-bin moment problem (§10.4,
diagnosed), real depth into the forecast and into `R_hyp` (§10.5, done), magnitude-threshold
sensitivity (§10.7, four regions x two thresholds). Still open: time-varying `M_c(t)`, the
Zaliapin-Ben-Zion nearest-neighbour declustering alternative (now more clearly motivated — §10.7
shows how much the declustering choice determines what is left to predict), the three-component
GMPE residual partition, regionalised Vs30, the multi-oscillator portfolio, and the SDOF-vs-MDOF
comparison.
