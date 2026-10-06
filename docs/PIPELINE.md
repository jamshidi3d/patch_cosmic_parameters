# Per-patch parameter-fitting pipeline

How `patch_cosmology_fit.ipynb` measures ΛCDM parameters on the full sky and in
12 sky patches, and how it differs from the reference analysis it reproduces,
Gimeno-Amo et al. 2025, *Exploring Statistical Isotropy in Planck Data Release 4*
([arXiv:2504.05597](https://arxiv.org/abs/2504.05597)). For the fitting-code
details (convergence, error bars, minimizers) see
[FIT_CONVERGENCE_FIXES.md](FIT_CONVERGENCE_FIXES.md).
For the theory behind each stage, see [docs/theory.md](theory.md) and
[docs/levenberg_marquardt.md](levenberg_marquardt.md).

## Pipeline

1. **Data.** Planck PR3 SMICA half-mission maps `hm1`, `hm2` (I, Q, U; Nside 2048, uK).
2. **Masks.** PR3 common confidence masks (intensity for T, polarization for Q/U),
   C1-apodized by 0.3°.
3. **Patches.** The 12 HEALPix Nside=1 base pixels (numbered 0–11; RING = NESTED at
   Nside 1), each intersected with the survey mask and re-apodized
   (`patch.make_superpixel_mask`, `patch.apodize_mask`). Per-patch f_sky (T: 2.3–7.8%,
   P: 2.4–7.8%) matches the paper's Table 1 to the quoted precision.
4. **Spectra.** NaMaster pseudo-Cℓ cross-spectra hm1 × hm2 (no noise bias): TT
   (spin 0 × 0), TE (0 × 2), EE (2 × 2), each decoupled with its own workspace;
   beam = 5′ Gaussian × HEALPix pixel window.
   Ranges: **TT 32–2011, TE 32–1741, EE 32–1471, Δℓ = 30** (66 / 57 / 48 bandpowers;
   bin [2, 31] dropped) — identical for the full sky and every patch.
5. **Covariance.** Analytic Gaussian (NaMaster `gaussian_covariance`) from a fiducial
   CAMB spectrum plus the per-split noise estimated from (hm1 − hm2)/2. Only
   same-bin terms are kept: TT, TE, EE variances and their TT–TE, TT–EE, TE–EE
   covariances in the same bin
   (`power_spectrum.compute_joint_tt_te_ee_covariance`).
6. **Model.** CAMB lensed spectra pushed through the data's own workspaces
   (`fitting.theory_vector`). Free: H0, Ωbh², Ωch², As, ns, plus the point-source
   amplitudes A_ps^TT, A_ps^EE (D_ℓ = A (ℓ/3000)²). Fixed: τ = 0.0602,
   Σmν = 0.06 eV, r = 0.01.
7. **Fit.** Gaussian χ², one pass, no restarts (`fitting.fit_parameters`):
   `FIT_METHOD = "iminuit"` (MIGRAD) or `"levenberg_marquardt"`. Errors come from the
   Gauss–Newton curvature at the minimum. Wide validity bounds act as a smooth χ²
   wall, never as Minuit limits. Every fit reports `status` / `flags`.
8. **Outputs.**
   - `output/fullsky_*`, including `fullsky_vs_paper.txt`: the comparison with the
     paper's Table 2.
   - `output/patched_<tag>/`:
     - per-patch result cache `patch_XX_<tag>.json` (an interrupted run resumes);
     - parameter dump and plot;
     - TT, TE and EE per-patch spectrum figures.

Switches (notebook config cells): `FIT_METHOD`, `fixed_params` (fixed at the full-sky
best fit), `USE_PATCH_PRIORS` (Gaussian A_ps priors from the full sky; off by default).

## Stored results (nothing needs recomputing)

| What | Where | Reused by |
|---|---|---|
| **Likelihood packages**: data vector, covariance + inverse, bandpower windows, bins, ells, errors, f_sky (≈1–5 MB each) | `output/likelihoods/fullsky.npz`, `output/likelihoods/patches_nside2048_lmax2011-1741-1471_dl30/patch_XX.npz` | `fitting.fit_one_patch(..., likelihood_cache=...)` (automatic in Section 4) and `fitting.load_likelihood(path)` → `(fit_data, extra)` for any refit |
| Per-patch fit results (best fit, errors, covariance, χ², status/flags, spectra) | `output/patched_<tag>/patch_XX_<tag>.json` | Section 4 reloads them when the fit config matches |
| Full-sky fit and comparison with the paper | `output/fullsky_result_<tag>.json`, `fullsky_params_<tag>.txt`, `fullsky_vs_paper.txt`, `fullsky_spectrum_<tag>.png` | Section 4 (start point, reference curves) |
| Figures | `output/patched_<tag>/patched_bandpowers{,_TE,_EE}_<tag>.png`, `patched_params_<tag>.png/.txt` | — |
| The executed notebook of each production run (all outputs inline) | `output/runs/<date>_<tag>/` | — |

- **What a likelihood package holds.** It contains everything NaMaster computes, which
  is the expensive step (minutes per patch, GBs of RAM). Instead of NaMaster
  workspaces, the theory is binned with the stored bandpower windows. The result is
  identical to machine precision (checked: relative difference ≲ 1e-15).
- **Refitting.** A refit with other fixed parameters, priors or minimizer only costs
  CAMB time. Changing `fixed_params` in Section 4 picks the packages up automatically.
- **When packages are recomputed.** A change to any data-side setting (ℓ ranges,
  binning, nside, beam, masks) recomputes them, because each package's metadata is
  checked on load.

**Memory note.** Patches use ℓmax = 2011, so each one builds full-size NaMaster
workspaces. Section 4 frees Sections 2/3's full-sky objects first, and each patch keeps
only its small windows. A run of Sections 2–4 in one kernel was OOM-killed at ~27 GB
before those two fixes.

## Differences from arXiv:2504.05597

| | Paper | This pipeline | Effect |
|---|---|---|---|
| Data | PR4 (NPIPE) SEVEM, **detector-split A × B** cross-spectra | PR3 SMICA **half-mission hm1 × hm2** | Different data, foreground cleaning and noise. Central values can shift by a fraction of σ, and errors are somewhat larger (PR3 is noisier). Not avoidable: the PR4 SEVEM A/B maps are not public (the PLA provides only the full map). |
| Covariance | Estimated from **600 PR4 E2E simulations** (same-bin terms) | **Analytic Gaussian** + half-difference noise (same-bin terms) | Misses non-Gaussian and real-noise effects; error bars differ by O(10%). |
| Debiasing | Spectra corrected by input − ⟨600 sims⟩; both columns reported | **None** | Compare with the paper's **"No debiasing"** column only. |
| Mean field, dipole PTEs | From the simulations | Not computed | Out of scope for now. |
| Minimizer | iMinuit with bounds | iMinuit (MIGRAD, gradient from fixed-step Jacobian, smooth bounds) or Levenberg–Marquardt; Gauss–Newton errors | Same minimum. Errors are not taken from HESSE, which is unreliable on CAMB's noisy χ² (see FIT_CONVERGENCE_FIXES.md). |
| Everything else | masks, apodization, patches, ℓ ranges, binning, model, fixed parameters, nuisance template | **same** | — |

**Reference values** (paper, Table 2, full sky, no debiasing):
- H0 = 66.78 ± 0.50
- Ωbh² = 0.02212 ± 0.00013, Ωch² = 0.1209 ± 0.0011
- ln(10¹⁰As) = 3.057 ± 0.0033, ns = 0.9598 ± 0.0036
- A_ps^TT = 55 ± 4, A_ps^EE = 0 ± 1

The paper gives **no per-patch data values** (only dipole directions and sim-based
PTEs), so the per-patch fits can be checked for health but not number-by-number.

**Full-sky result of this pipeline** (2026-10-06, after the covariance fix of
[power_spectrum_errors.md](power_spectrum_errors.md) §8.2; iminuit;
`output/fullsky_vs_paper.txt`): χ² = 178.9 for 171 bandpowers.

| | this pipeline | paper, no debias | Δ / σ_paper | σ ratio | (before the fix: value, σ ratio) |
|---|---|---|---|---|---|
| H0 | 65.97 ± 0.60 | 66.78 ± 0.50 | −1.62 | 1.20 | 66.45, 1.31 |
| Ωbh² | 0.02215 ± 0.00015 | 0.02212 ± 0.00013 | +0.19 | 1.19 | 0.02234, 1.36 |
| Ωch² | 0.1231 ± 0.0014 | 0.1209 ± 0.0011 | +1.96 | 1.26 | 0.1219, 1.31 |
| ln(10¹⁰As) | 3.064 ± 0.004 | 3.057 ± 0.0033 | +2.11 | 1.22 | 3.061, 1.26 |
| ns | 0.9563 ± 0.0044 | 0.9598 ± 0.0036 | −0.97 | 1.23 | 0.9633, 1.38 |
| A_ps^TT | 57.0 ± 4.2 | 55 ± 4 | +0.50 | 1.04 | 47, 2.14 |
| A_ps^EE | 4.4 ± 1.5 | 0 ± 1 | +4.4 | 1.51 | 5.2, 1.36 |

- **Point sources and baryons.** With the high-ℓ TT bandpowers correctly weighted, the
  point-source amplitude A_ps^TT matches the paper (0.5σ, same precision) and Ωbh²
  moves onto the paper's value.
- **Other ΛCDM parameters.** H0, Ωch², As and ns lie 1–2 σ_paper away along the usual
  H0–Ωch²–ns degeneracy.
- **Error bars.** Ours are 19–26% larger than the paper's (26–38% before the fix). That
  gap is what one expects from PR3 SMICA being noisier than PR4 SEVEM, and from the
  analytic covariance.
- **A_ps^EE.** It stays nonzero (4.4 ± 1.5, against the paper's 0 ± 1): a small-scale
  polarized residual specific to SMICA PR3 (see
  [point_source_amplitudes.md](point_source_amplitudes.md) §4).
- **Why the remaining differences are not fitting error.** Both minimizers find the same
  minimum, and the covariance is validated by simulation
  ([power_spectrum_errors.md](power_spectrum_errors.md) §8.3). What is left is the data
  difference (PR3 SMICA hm1×hm2 vs PR4 SEVEM A×B) and the analytic vs simulated
  covariance.

**Patches** (12/12 `status=ok`, `output/patched_all_varied/`):
- χ² is 130–202 for 171 bandpowers (median ≈ 174), as expected for a correct
  covariance.
- A_ps^TT per patch is 50–83 ± 12–30, consistent with the full sky.
- Parameter errors are 6–15% smaller than before the fix.

**To close the remaining gaps later**
- [ ] Obtain the PR4 SEVEM A/B maps (from the authors or NERSC) and rerun on them.
- [ ] Sim-based covariance and debiasing (needs PR4 E2E sims).
