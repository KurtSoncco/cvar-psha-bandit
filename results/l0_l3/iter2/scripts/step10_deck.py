"""Step 10: a deck for the next-iteration findings.

A new deck rather than a patch of EQ-JEPA_L0-L3_presentation.pptx: that deck documents the
first build and its numbers are superseded wholesale, and its result slides carry embedded
chart images that would all have to be regenerated anyway. The new figures replace them.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

SLIDES = [
    ("EQ-JEPA L0: what the next iteration found",
     "Ten open defects addressed · 71 monthly windows, California M2.8+ · "
     "every correction fitted pre-2019 and frozen\n"
     "Full write-up: §10 of EQ-JEPA_sequential_definition.md and "
     "L0_L3_verification_report_v2.md", None),
    ("Two headline numbers were measuring the plumbing",
     "• Decoding per window at its own horizon reverses the rate error: 58.8 forecast vs 17.1 "
     "observed — an over-forecast of 3.4x, not the 3.5x under-forecast on record\n"
     "• The ratio is the declustering ratio: trained on 19,152 raw events, scored on 5,491 "
     "independent ones; fitted constant c0 = 0.326\n"
     "• Admit a constant scale and the Hawkes excitation amplitude fits to zero for all three "
     "models — the correction was a constant\n"
     "• Applied symmetrically: L0 beats the historical-rate baseline in 0 of 71 windows. H1 refuted",
     "fig_step1_symmetric_scoring.png"),
    ("The encoder had collapsed — and a frozen latent scores the same",
     "• Participation ratio 1.09 of 128; PC1 = 95.5% of variance; mean pairwise cosine 0.958\n"
     "• No dead units (std 0.012–0.87) — the collapse is directional, invisible to a "
     "per-dimension check\n"
     "• 85% of 512 sketched projections reject Gaussianity\n"
     "• Control: replacing the latent with a constant matches or beats the trained encoder on "
     "all four CSEP tests",
     "fig_step2_collapse_diagnostics.png"),
    ("The input embedding, not the missing isotropy term, caused it",
     "• Retraining the unchanged architecture on declustered counts: S-test 8.5% → 88.7%\n"
     "• Fixed embedding (normalisation + Fourier for lat/lon/depth + Time2Vec for dt): "
     "participation ratio 1.13 → 22.4, CL pass 49% → 76%\n"
     "• SIGReg cannot rescue the original embedding (1.13 → 1.18) and adds nothing once the "
     "embedding is fixed\n"
     "• No arm beats the baseline; the best is tied in 65 of 71 windows, up from 37",
     "fig_step3_ablation.png"),
    ("Point process: magnitude side fixed, spatial side broken",
     "• Exactly normalised closed-form intensity λ(t,x,y,z,m); trained on the marked-Poisson "
     "likelihood of real coordinates; self-calibrates at 18.0 vs 17.1 events/window\n"
     "• Moment budget well-posed for the first time: 4.14e18 N·m/yr = 0.38x observed, against "
     "4–60x for every gridded forecast\n"
     "• Spatial field fails: S-test 9.9% gridded, catalog-track spatial 31%, calibration "
     "quantile 2e-46 — axis-aligned components cannot make fault-aligned lineaments\n"
     "• Fix: an anisotropic spatial family, not more components (H10)",
     "fig_step5_pointprocess.png"),
    ("Energy is a weighting and a constraint, not a target",
     "• log10 E = 1.5M + const is a bijection: forecasting converted energy is forecasting "
     "magnitude\n"
     "• A window's energy is its largest event: median 42% monthly, 50% annually; r(log E, Mmax) "
     "= 0.98 against 0.75 for log-count\n"
     "• Moment-rate budget: every gridded forecast overshoots 4–60x through the open [5.5, 8.5] "
     "bin — M-test passes while the budget is an order of magnitude off\n"
     "• The E-test works but barely discriminates; the moment-rate constraint is the useful form",
     "fig_step4_moment_rate_and_etest.png"),
    ("Depth: a test that works, and 49% of the risk tail",
     "• D-test written on pycsep's simulate-and-rank pattern (no depth test exists in the "
     "package): the point-process L0 passes in 91.5% of windows\n"
     "• Real hypocentral distance cuts CVaR₀.₉₉ from 0.0120 to 0.0080 — a 33% reduction; "
     "surface-source treatment inflates the tail 49%\n"
     "• Entirely near-field: only 4% of forecast events fall within 50 km, and those set the tail\n"
     "• A lower bound — the ductility map here is the linear limit μ = Sa/η",
     "fig_step6_depth.png"),
    ("What actually moves the tail",
     "• Ground-motion σ 0.33 → 0.70 (the empirical range): +46%\n"
     "• Depth ignored: +49%\n"
     "• Count dispersion, NB r = 0.25 vs Poisson: +7%\n"
     "• Of 1,760 cell/magnitude bins only 74 have ≥5 events; 57% fit at the Poisson limit and "
     "5.4% reject Poisson — the nominal false-positive rate — while the NB head predicts r ≈ 2.0",
     "fig_step8_dispersion_cvar.png"),
    ("The temporal machinery is fine; the target is the problem",
     "• Temporal additivity holds: implied daily rate 0.583 at horizons 1 to 365 days; three "
     "monthly decodes vs one three-month decode, median residual −0.6%\n"
     "• The Poisson predictive law is already calibrated (PIT KS p = 0.58, variance/mean 0.91) — "
     "so the conditional-mean decoder is not the limitation\n"
     "• Observed counts: lag-1 autocorrelation −0.08, no memory at any lag\n"
     "• No model correlates positively with them: L0 −0.05, gridded L0 −0.09, historical rate "
     "−0.16, ETAS-lite −0.17",
     "fig_step7_multihorizon.png"),
    ("Declustering removes the signal L0 is asked to learn",
     "• Four regions, two thresholds, three window lengths, fresh from USGS ComCat\n"
     "• Variance/mean raw → declustered: California 173 → 0.98, Japan 443 → 1.78, "
     "Chile 98 → 1.60, Italy–Greece 36 → 5.99\n"
     "• Out-of-sample log-likelihood gain per event from an excitation feature: 0.017–0.22 raw, "
     "0.0001–0.006 declustered\n"
     "• Exception: Italy–Greece M3.5+ keeps ACF 0.81 after declustering — fixed windows do not "
     "remove swarm clustering",
     "fig_step9_crossregion.png"),
    ("Where this leaves the build order",
     "M11 — move the target before touching the architecture: undeclustered catalog, 1/3/7-day "
     "horizons, with the measured excitation baselines as the bar to beat (H9)\n"
     "M12 — anisotropic spatial decoder; swap only f_xy and rescore both tracks (H10)\n"
     "M13 — moment-rate budget as a training constraint, not an after-the-fact check\n"
     "M14 — partition the L2 residual; carry the 0.375–0.617 missing-sigma interval into every "
     "CVaR\n"
     "M15 — retire the NB dispersion head, or move it to a target that is actually overdispersed",
     None),
]

def build(path="EQ-JEPA_next-iteration_findings.pptx"):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for i, (title, body, fig) in enumerate(SLIDES):
        slide = prs.slides.add_slide(prs.slide_layouts[6 if i else 6])
        tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.9))
        p = tb.text_frame.paragraphs[0]; p.text = title
        p.font.size = Pt(28 if i == 0 else 24); p.font.bold = True
        p.font.color.rgb = RGBColor(0x11, 0x2B, 0x4A)
        if body:
            w = Inches(12.2) if fig is None else Inches(5.2)
            bb = slide.shapes.add_textbox(Inches(0.6), Inches(1.45), w, Inches(5.4))
            tf = bb.text_frame; tf.word_wrap = True
            for j, line in enumerate(body.split("\n")):
                par = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                par.text = line
                par.font.size = Pt(16 if fig is None else 13)
                par.space_after = Pt(8)
        if fig:
            slide.shapes.add_picture(fig, Inches(6.1), Inches(1.5), width=Inches(6.9))
    prs.save(path)
    print(f"wrote {path} with {len(SLIDES)} slides")

if __name__ == "__main__":
    build()
