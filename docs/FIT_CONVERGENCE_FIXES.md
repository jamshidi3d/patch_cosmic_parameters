# Fit convergence fixes (2026-10-05)

Why the per-patch fits stalled at bounds or reported impossibly small
error bars, what was changed in `cosmo_patch/` and both notebooks, and
what is still worth checking. The physics pipeline (spectra, covariance,
CAMB theory, mode coupling) was **not** changed; every fix is in how the
chi^2 is minimized and how its errors are read off.

Evidence used below: `output/patched_all_varied/patched_params_all_varied.txt`
(the last 12-patch joint run with the multi-start code) and
`output/old_results/patched_params_all_varied.txt` (the run before it).

---

## 1. Point-source amplitudes are flat directions on a patch

**Symptom.** `A_ps_TT` ended at its upper bound 200 with error ~190 in 9 of
12 patches (or at 0 with error ~100-150 in the rest); `A_ps_EE` at 0 or 20.

**Cause.** The template is D_ell = A (ell/3000)^2. With the patch cuts
ell <= 1000 (TT) and ell <= 700 (EE) it contributes at most A/9 to TT and
0.05 A to EE -- for A_ps_TT = 200, about one bandpower sigma in the last
bin only. The patch data simply don't measure these amplitudes, so MIGRAD
drifts along them until it hits a bound.

**Fix.**
- **Primary (2026-10-06):** the patch fits now use the reference paper's
  multipole ranges (TT 32-2011, TE 32-1741, EE 32-1471, Delta ell = 30; see
  PIPELINE.md). At ell ~ 2000 the template is 0.44 A, so each patch's own
  data constrain the amplitudes, and they are fitted freely, as in the paper.
- **Optional:** Gaussian priors from the full-sky fit, switched on with
  `USE_PATCH_PRIORS = True` in the joint notebook (`PATCH_PRIORS`; A_ps_TT
  only in the TT notebook, where they are on). API:
  `fit_parameters(..., priors={name: (mean, sigma)})`, implemented as extra
  residual rows (`FitProblem`), so the prior's uncertainty propagates into
  every other error. Worth using only with a low patch ell_max, where the
  amplitudes are flat directions.
- No bounds on A_ps any more.

**To verify later**
- [x] On the full 12-patch run (paper ranges, no priors), A_ps_TT / A_ps_EE come out finite and data-constrained in every patch (A_ps_TT errors 26-62, A_ps_EE 3.7-10.7).
- [ ] If priors are ever used: check `chi2_prior` stays O(1), and that widening the prior x3 moves the cosmological parameters by much less than their errors.

## 2. Minuit's bounded-parameter transform collapses the errors at a bound

**Symptom.** Fake-tiny errors whenever a parameter touched a limit, e.g.
old patch 11: `ns +/- 3.2e-05`, `A_ps_TT = 200 +/- 7.6e-05`,
`H0 = 70.65 +/- 0.05`; current patch 4: `ns = 0.9` (bound) with
`H0 = 57.08 +/- 0.65`.

**Cause.** Minuit maps a parameter with `limits` through
P = lo + (hi-lo)(sin(theta)+1)/2. The slope dP/dtheta is zero at the bound,
so the external covariance (internal covariance times the slope squared)
collapses there -- for that parameter and, through correlations, the
others. The same effect makes the covariance inaccurate/non-posdef, which
is what triggered the multi-start rescue.

**Fix.** No Minuit `limits` at all. `bounds` are now wide *validity* limits
(`VALIDITY_BOUNDS`: H0 40-100, ombh2 0.005-0.05, omch2 0.01-0.5,
As 0.5e-9-5e-9, ns 0.7-1.3), implemented as a smooth quadratic chi^2 wall
outside the box (`WALL_STIFFNESS`), with the theory evaluated at the
clipped point. A fit that ends within 0.1 step of a bound is flagged
`at_bound:<names>` instead of silently reporting a fake error.

**To verify later**
- [ ] No patch in the full run is flagged `at_bound`; if one is, look at that patch's data rather than tightening/loosening bounds.

## 3. The physical bounds were narrower than a patch's sensitivity

**Symptom.** ns at its bound in patches 4 (0.9) and 6 (1.05); per-patch
sigma_ns is 0.02-0.14, so (0.9, 1.05) is a ~1-2 sigma box for a patch.

**Fix.** Same as 2 -- the old ranges were acting as hard priors. A patch is
now allowed to land wherever its data say (an outlying patch is the
interesting result of a directional analysis, not something to clip).

**To verify later**
- [ ] Look at patches whose ns / H0 land far from the full-sky value: is chi2/n_data reasonable there (no data problem)?

## 4. Poor conditioning, and HESSE on a noisy chi^2 (the main source of fake-tiny errors)

**Symptom.** Slow MIGRAD (`NegativeG2LineSearch`), `status=unresolved` /
`above_max_edm`, and error bars many orders of magnitude too small even
where nothing sits at a bound (old patch 11: `H0 +/- 0.05`, `ns +/- 3e-5`).

**Causes.**
- Parameter scales from 1e-9 (As) to 70 (H0); `fit_one_patch` never even
  forwarded `initial_step`, so the patch fits ran on iminuit's default
  step guesses.
- CAMB's chi^2 is not smooth to double precision. Measured on a patch-like
  synthetic chi^2 (~20; 4th differences along one parameter): step-like
  jumps of ~1e-4 in chi^2 along H0 and ~2e-6 along omch2, ns smooth to
  ~1e-9. d2chi2/dH0^2 from steps of 0.001-0.01 scatters 14.7..21.5 (true
  ~15.1); from steps >= 0.1 it is stable.
- **Minuit's HESSE cannot cope with that.** At the synthetic best fit,
  HESSE alone fails and returns its forced-positive-definite fallback:
  H0 +/- 3.8e-7, ombh2 +/- 1.4e-9, As +/- 1.4e-15 -- the same signature as
  the old fake errors. Setting `Minuit.precision` (1e-7 .. 1e-3) changed
  nothing. Run after MIGRAD, HESSE "succeeded" but gave errors 1.4-1.8x
  too small (H0 1.14 vs 2.09).

**Fix.**
- Minuit runs in O(1) coordinates x = (p - p0)/step (`DEFAULT_STEP`:
  H0 0.5, ombh2 1e-4, omch2 1e-3, As 1e-11, ns 5e-3, A_ps_TT 5, A_ps_EE 0.5),
  `strategy=0`, and gets `grad=` from a central-difference Jacobian with
  those fixed physical steps (`FitProblem.jacobian`) -- far above the noise.
- **Errors (both methods) are the Gauss-Newton curvature at the minimum**,
  Cov = (J^T C^-1 J + priors)^-1 -- for a Gaussian chi^2 the standard
  curvature estimate, exact for a model linear around the minimum.
  Measured: unchanged to 3 digits for Jacobian steps 0.5x, 1x, 2x
  `DEFAULT_STEP`.
- HESSE is opt-in, as a diagnostic: `fit_parameters(..., run_hesse=True)`
  adds `errors_hesse` and flags `hesse_failed` / `hesse_fisher_mismatch`.
- `camb_cl` is memoized, so Jacobian columns of the point-source
  amplitudes cost no Boltzmann solve.

**To verify later**
- [x] Full sky: Gauss-Newton errors (H0 0.655) match the old HESSE errors (0.629) -- the patch-level HESSE failure is the CAMB noise, not the Gauss-Newton approximation.
- [ ] If CAMB accuracy settings are ever changed, re-measure the noise floor (4th differences of chi^2 along H0 / omch2 at steps 1e-4..1e-2 of `DEFAULT_STEP`).
- [ ] Optional cross-check of the Gaussian errors on one patch: a 1D profile scan in H0 (refit others) should cross Delta chi^2 = 1 near +/- errors['H0'].

## 5. The multi-start rescue never changed the answer

**Symptom.** Every rescued patch has `chi2_spread` ~ 1e-4 across 6 starts
spread over the whole box: all starts reach the same minimum.

**Cause.** There were no local minima to escape -- the failures were 1-4
above, which restarting cannot fix.

**Fix.** Removed (`_diverse_starts`, `_fit_with_recovery`, `n_starts`,
`seed`, `force_multistart`). Each fit is one pass; its health is reported
in `status` (`ok` / `flagged`) and `flags` (e.g. `migrad_invalid`,
`covariance_not_posdef`, `at_bound:...`,
`lm_stalled`, and with `run_hesse=True` `hesse_failed` /
`hesse_fisher_mismatch:...`). Flagged patches are drawn in red in the
parameter plot.

## 6. Second minimizer: Levenberg-Marquardt

`cosmo_patch/levenberg_marquardt.py`, selected with
`FIT_METHOD = "levenberg_marquardt"` in either notebook
(`fit_parameters(..., method="levenberg_marquardt")`). The chi^2 is a sum of
squared whitened residuals, for which Gauss-Newton with Marquardt damping is
the textbook algorithm; it uses the same `FitProblem` (same priors, wall,
Jacobian), converges on EDM < 1e-3, and reports the Gauss-Newton (Fisher)
covariance. Outputs get a `_levenberg_marquardt` suffix, so both methods'
results can sit side by side.

**To verify later**
- [ ] Run Section 4 of the joint notebook with both `FIT_METHOD` values and compare the per-patch parameters (they should agree well within the errors).

## 7. Other changes

- `fit_one_patch` forwards `priors`, `initial_step` and `method`; patch
  fits start from the full-sky best fit.
- Per-patch cache JSON stores the new result keys; its `config` now
  includes `method` and `priors`, so caches written by the old code are
  recomputed automatically.
- Joint notebook: new cell after the per-patch TT figure draws the same
  4x3 figure for **TE** and **EE** (`patched_bandpowers_TE_<tag>.png`,
  `patched_bandpowers_EE_<tag>.png`). The TT-only notebook has no
  polarization data, so no TE/EE figures there.
- Section 5 (both notebooks): the H0-omch2 contour is drawn from the fit
  covariance (works for both methods, no extra CAMB solves) instead of
  `draw_mncontour`.
- `patch_cosmology_fit_colab.ipynb` was **not** updated; it still prints
  the removed `n_starts_tried` / `chi2_spread` keys.

**To verify later**
- [ ] Sync the Colab notebook from `patch_cosmology_fit.ipynb` before using it again.
- [ ] Re-run the full-sky fits (both notebooks): the joint one should reproduce H0 = 66.45 +/- 0.63 etc. (it was already well behaved).

---

## Test results (2026-10-05, WSL env: camb 1.6.5, iminuit 2.32, pymaster 2.7)

Re-fit with the joint-notebook Section 4 config (all varied: nside 2048,
ell_max TT/TE/EE = 1000/850/700, Delta_ell = 90, start at the full-sky best
fit, A_ps priors 47.49 +/- 8.52 and 5.16 +/- 1.36, `VALIDITY_BOUNDS`).
Patches 4, 6, 9 were `unresolved` in the old run; 7 was `ok`.

| patch | old status (old H0, ns) | method | new status | chi2 / 24 | H0 | ns | CAMB solves |
|---|---|---|---|---|---|---|---|
| 4 (f_sky 0.023) | unresolved, ns at bound (57.08 +/- 0.65, 0.900) | iminuit | ok | 13.24 | 52.1 +/- 5.3 | 0.847 +/- 0.042 | 157 |
| 6 (f_sky 0.034) | unresolved, A_ps_TT at bound (75.91 +/- 3.23, 1.050) | iminuit | ok | 25.40 | 77.1 +/- 6.8 | 1.068 +/- 0.060 | 161 |
| 9 (f_sky 0.076) | unresolved, both A_ps at bound (71.99 +/- 3.29, 1.021) | iminuit | ok | 61.52 | 70.81 +/- 3.78 | 1.0098 +/- 0.031 | 111 |
| 9 | | levenberg_marquardt | ok* | 61.52 | 70.79 +/- 3.78 | 1.0096 +/- 0.031 | 33 |
| 7 (f_sky 0.049) | ok, but A_ps_TT = 199.9 (71.84 +/- 2.28, 0.999) | iminuit | ok | 18.79 | 70.59 +/- 4.01 | 0.9881 +/- 0.035 | 102 |
| 7 | | levenberg_marquardt | ok | 18.79 | 70.58 +/- 4.01 | 0.9881 +/- 0.035 | 22 |

\* The first run of patch 9 with LM printed `covariance_not_posdef`. That flag
was wrong: the check ran `eigvalsh` on the raw covariance (entries
1e-21..1e1). It now runs on the correlation matrix. All four stored
covariances are positive definite (min. correlation eigenvalue 0.017-0.029).

- All fits are single-pass. Nothing sits at a validity bound, and A_ps sits on
  its prior (`chi2_prior` <= 0.04).
- The two methods agree to < 0.01 sigma on every parameter (errors identical:
  both use the Gauss-Newton curvature). LM needs 3-5x fewer CAMB solves:
  about 30 s against about 5 min for MIGRAD on one patch.
- Even the old "ok" patch 7 was distorted. With A_ps_TT on its bound, its
  H0 error was 2.28; with the prior and honest errors it is 4.01.
- Synthetic checks: on a mock TT+TE+EE patch with known truth, all pulls
  are < 1.4 sigma for both methods. The TT-only path (`fit_parameters_tt`,
  ell <= 2508) gives the same result with both methods (H0 68.70 / 68.73 +/- 2.23), pulls <= 1.1 sigma.
- The TE/EE per-patch figures were produced by the notebook's own new cell
  on these four patches.

**To verify later (from these results)**
- [ ] Patch 4 (smallest f_sky, 0.023) prefers ns = 0.85 +/- 0.04 and H0 = 52 +/- 5, about 3 sigma below the full sky, along the H0-omch2-ns degeneracy. Its chi2 is fine (13.2/24). Check its mask/foreground residuals before reading anything into it; the old fit hid this behind the ns >= 0.9 bound.
- [ ] Patch 9 has chi2 = 61.5 for 24 bandpowers (the old fit had 59.1, so this comes from the data, not the fit). Look at its per-spectrum residuals: which bins drive it?
- [x] Joint notebook: full 12-patch Section 4 run with the new code on 2026-10-06 (paper ell ranges, free A_ps): all 12 `status=ok`, no flags (`output/patched_all_varied/`, `output/runs/2026-10-06_all_varied/`). TT-only notebook not rerun yet.
