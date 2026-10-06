# Per-patch parameter-fitting pipeline

How `patch_cosmology_fit.ipynb` measures ΛCDM parameters on the full sky and in
12 sky patches, and how it differs from the reference analysis it reproduces,
Gimeno-Amo et al. 2025, *Exploring Statistical Isotropy in Planck Data Release 4*
([arXiv:2504.05597](https://arxiv.org/abs/2504.05597)). For the fitting-code
details (convergence, error bars, minimizers) see
[FIT_CONVERGENCE_FIXES.md](FIT_CONVERGENCE_FIXES.md).
For the theory behind each stage, see [docs/theory.md](docs/theory.md) and
[docs/levenberg_marquardt.md](docs/levenberg_marquardt.md).

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

**Full-sky result of this pipeline** (2026-10-06, iminuit; `output/fullsky_vs_paper.txt`):
χ² = 178.2 for 171 bandpowers.

| | this pipeline | paper, no debias | Δ / σ_paper | σ ratio |
|---|---|---|---|---|
| H0 | 66.45 ± 0.66 | 66.78 ± 0.50 | −0.66 | 1.31 |
| Ωbh² | 0.02234 ± 0.00018 | 0.02212 ± 0.00013 | +1.66 | 1.36 |
| Ωch² | 0.1219 ± 0.0014 | 0.1209 ± 0.0011 | +0.88 | 1.31 |
| ln(10¹⁰As) | 3.061 ± 0.004 | 3.057 ± 0.0033 | +1.18 | 1.26 |
| ns | 0.9633 ± 0.0050 | 0.9598 ± 0.0036 | +0.96 | 1.38 |
| A_ps^TT | 47 ± 9 | 55 ± 4 | −1.9 | 2.1 |
| A_ps^EE | 5.2 ± 1.4 | 0 ± 1 | +5.2 | 1.4 |

- **ΛCDM parameters.** All five agree within 0.7–1.7 σ_paper (≲ 1.3σ of the combined error).
- **Error bars.** Ours are 26–38% larger, as expected from PR3 vs PR4 noise and the
  analytic covariance.
- **Nuisances.** These differ most. They absorb the foreground and point-source
  residuals of the specific cleaned map, and those residuals are not the same in
  SMICA PR3 and SEVEM PR4.
- **Why this is not fitting error.** The offsets are a data difference, not a
  minimizer problem: both minimizers find the same minimum, and Gauss–Newton and HESSE
  errors agree at full sky. Shifts of this size between PR3 and PR4 are typical.


**To close the remaining gaps later**
- [ ] Obtain the PR4 SEVEM A/B maps (from the authors or NERSC) and rerun on them.
- [ ] Sim-based covariance and debiasing (needs PR4 E2E sims).
