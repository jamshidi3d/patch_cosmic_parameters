# How the pipeline works: theory and implementation

This document covers the chain from Planck maps to cosmological parameters, on the
full sky and in 12 sky patches, as implemented in `cosmo_patch/` and the notebooks.
For each stage it gives the estimator or approximation used, and why. Companion
documents:
- [levenberg_marquardt.md](levenberg_marquardt.md): the LM minimizer;
- [power_spectrum_errors.md](power_spectrum_errors.md): the derivation and validation of the bandpower errors;
- [../PIPELINE.md](PIPELINE.md): settings, and the differences from
  arXiv:2504.05597;
- [../FIT_CONVERGENCE_FIXES.md](FIT_CONVERGENCE_FIXES.md): the history of the
  fitting fixes.

---

## 1. Data and masks

- **Maps.** Planck PR3 SMICA CMB maps from the two half-missions (`hm1`, `hm2`):
  temperature $T$ and Stokes $Q, U$, HEALPix $N_{\rm side}=2048$, converted to μK.
  The two halves see the same sky with **independent noise**.
- **Masks.** PR3 common confidence masks, one for intensity and one for polarization,
  with a C1 apodization of 0.3° (`patch.apodize_mask`). A hard mask edge would couple
  multipoles strongly and leak power. Tapering the edge keeps the coupling compact.
- **Patches.** The 12 base pixels of an $N_{\rm side}=1$ HEALPix grid, numbered 0–11
  (`patch.make_superpixel_mask`). Each is multiplied by the survey mask and
  re-apodized. The patch masks keep f_sky ≈ 2–8% of the sky, matching Table 1 of
  arXiv:2504.05597. Every patch then goes through *exactly* the same chain as the full
  sky, independently.

## 2. Power spectra: pseudo-Cℓ with mode decoupling (MASTER / NaMaster)

On a masked sky, the spherical-harmonic power of the masked map, the *pseudo-Cℓ*
$\tilde C_\ell$, is a mixture of the true $C_\ell$ over neighbouring multipoles:

$$
\langle \tilde C_\ell \rangle = \sum_{\ell'} M_{\ell\ell'}\, B_{\ell'}^2\, C_{\ell'} .
$$

- $M$ is the **mode-coupling matrix**, computed analytically from the mask's own power
  spectrum.
- $B_\ell$ is the beam (5′ Gaussian) times the HEALPix pixel window.
- For polarization, $M$ also mixes E and B (spin-2 fields; Q, U → E, B).

NaMaster (`pymaster`) bins the multipoles into bandpowers $b$, here Δℓ = 30 starting
at ℓ = 2, and inverts the *binned* coupling matrix:

$$
\hat C_b = \sum_{b'} \big(\mathcal{M}^{-1}\big)_{bb'}\, \tilde C_{b'} .
$$

This is the MASTER estimator (`power_spectrum.compute_power_spectrum`). Each spin
combination has its own workspace: TT (0×0), TE (0×2), EE (2×2).

**Cross-spectra between halves.** TT = $T^{hm1}\times T^{hm2}$,
TE = $T^{hm1}\times P^{hm2}$, EE = $P^{hm1}\times P^{hm2}$. Noise is independent
between the halves, so the cross-spectra carry **no noise bias**, and no noise model
is needed for the spectra themselves.

**Multipole ranges** (as in arXiv:2504.05597, for the full sky *and* each patch):
TT 32–2011, TE 32–1741, EE 32–1471, Δℓ = 30, giving 66 / 57 / 48 bandpowers.
- The first bin, ℓ = 2–31, is dropped.
- All fields share ℓmax = 2011 (NaMaster needs one ℓmax per workspace). Each spectrum
  is then truncated to its own range.

## 3. Bandpower windows: how theory is compared with data

The decoupled bandpowers are, in expectation, a linear functional of the true
spectrum:

$$
\langle \hat C_b \rangle = \sum_\ell \sum_k W_{b,k\ell}\, C^{(k)}_\ell ,
$$

where $k$ runs over the spin components:
- TT: [TT];
- TE: [TE, TB];
- EE: [EE, EB, BE, BB].

The **bandpower windows** $W$ come from `NmtWorkspace.get_bandpower_windows()`.
Theory must go through the same $W$, not just be averaged over each bin. That
accounts for the residual mixing left after the binned inversion; skipping it biases
the fit.

The code stores $W$ instead of the workspace (`fitting.bandpower_windows`). The binned
theory is then a matrix product (`fitting.theory_vector`):
- TB = EB = 0 in ΛCDM;
- TT → $W_{\rm TT}\,C^{TT}$;
- TE → $W_{\rm TE,0}\,C^{TE}$;
- EE → $W_{\rm EE,0}\,C^{EE} + W_{\rm EE,3}\,C^{BB}$.

This equals NaMaster's couple-then-decouple to machine precision (checked: ~1e-15).
It also reduces each patch's likelihood from GBs, the size of a full mode-coupling
matrix at ℓmax ≈ 2000, to a few MB.

## 4. Covariance

**Analytic Gaussian covariance.** NaMaster (`gaussian_covariance`) computes the
covariance of two pseudo-Cℓ cross-spectra $ab$ and $cd$ in the improved
narrow-kernel approximation. Schematically, for each pair of bins it takes

$$
\mathrm{Cov}\big(\hat C^{ab}_b, \hat C^{cd}_{b'}\big) \sim
\frac{C^{ac}C^{bd} + C^{ad}C^{bc}}{(2\ell+1)\,\Delta\ell\, f_{\rm sky}}
$$

and dresses it with mask-dependent coupling coefficients
(`NmtCovarianceWorkspace`). The input spectra are evaluated as follows:
- **Signal:** a fiducial CAMB ΛCDM spectrum (Planck 2018 parameters) for all legs.
- **Noise:** only on *auto* legs of the same half-mission (e.g. $C^{T^1T^1} = C^{TT} + N^{TT}$),
  with $N$ the **per-split** noise, also on the BB component. The cross legs $T^1T^2$
  are signal-only.
- **Beam convention:** every leg is passed to NaMaster at *map level*, i.e. multiplied
  by the beam × pixel window of its two fields. NaMaster's decoupling divides it out
  once. The derivation and the Monte Carlo check of these conventions are in
  [power_spectrum_errors.md](power_spectrum_errors.md).
- **Noise estimate** (`power_spectrum.estimate_noise_cl`): from the half-difference
  map $(m^{hm1}-m^{hm2})/2$, which cancels the sky and leaves pure noise. Its pseudo-Cℓ
  is doubled (half-difference power = per-split noise / 2), then divided by
  $\langle w^2\rangle$ (the f_sky correction) and by $B_\ell^2$.
- **T and P noise:** assumed uncorrelated.

**Joint TT+TE+EE (the paper's simplification).** Only *same-bin* terms are kept: the
variances of TT, TE, EE and their TT–TE, TT–EE, TE–EE covariances within each bin
(`compute_joint_tt_te_ee_covariance`). Each of the 6 blocks uses the coupling
coefficients of its own T/P field combination, because T and P masks differ.

The flattened NaMaster covariance is band-major. Component 0 sits at stride `ncls`,
which the code extracts explicitly. A covariance with condition number > 1e12 raises
an error rather than being inverted into garbage.

**TT-only notebook.** It keeps the full bin-to-bin TT covariance
(`compute_gaussian_covariance`).

## 5. Model

- **CAMB.** `fitting.camb_cl` returns *lensed* $C_\ell$ for TT, EE, BB, TE in μK²
  (`lens_potential_accuracy=1`; real data are lensed).
- **Free parameters:** H0, Ωbh², Ωch², As, ns.
- **Fixed parameters:** τ = 0.0602, Σmν = 0.06 eV, r = 0.01. A patch's polarization
  cannot constrain τ.
- **Residual point sources** are modelled as shot noise,
  $D_\ell = A_{\rm ps}\,(\ell/3000)^2$, with free amplitudes $A_{\rm ps}^{TT}$ and
  $A_{\rm ps}^{EE}$ added to TT and EE (`point_source_dl_template`). At the paper's ℓ
  ranges, each patch's data constrain them. Optional Gaussian priors from the full-sky
  fit exist (`USE_PATCH_PRIORS`, off) for cases with a low patch ℓmax, where the
  template is degenerate.
- **Memoization.** `camb_cl` is memoized on the exact parameter values. Re-evaluating
  a cosmology costs no new Boltzmann solve; this covers changing only $A_{\rm ps}$, and
  Minuit's gradient call at the point it just evaluated.

## 6. Likelihood

$$
\chi^2(p) = (d-t(p))^{\mathsf T} C^{-1} (d-t(p))
\;+\; \sum_{k\in\text{priors}} \Big(\frac{p_k-\mu_k}{\sigma_k}\Big)^2
\;+\; K\sum_k \Big(\frac{p_k-\mathrm{clip}(p_k)}{s_k}\Big)^2 .
$$

- The data term is a multivariate Gaussian in the bandpowers. `FitProblem` writes it
  as the squared norm of a whitened residual, with $C^{-1}=LL^{\mathsf T}$.
- The last term is a **smooth validity wall**, $K = 10^4$ per step-scale $s_k$.
  It keeps the fit inside the region where CAMB is trustworthy (H0 40–100, Ωbh²
  0.005–0.05, Ωch² 0.01–0.5, As 0.5–5×10⁻⁹, ns 0.7–1.3). The theory is evaluated at
  the clipped point.
- Unlike Minuit's own `limits`, the wall has no zero-slope transform. A parameter can
  never be pinned with a fake error; reaching a bound is flagged instead.
- **Fixed parameters** are simply removed from the free set.

## 7. Minimization

Both methods minimize the identical `FitProblem` in **one pass**, with no restarts.
Earlier multi-start runs always found the same minimum.

**Jacobian.** $J=\partial r/\partial p$ comes from central differences with **fixed
physical steps** (`DEFAULT_STEP`, ~0.2–1σ of a full-sky fit). CAMB's χ² has step-like
numerical noise (~1e-4 along H0 and Ωch²), so derivatives over tiny steps would
measure noise. Steps of this size are far above it.

**`method="iminuit"`** (default; the paper uses iMinuit):
- MIGRAD works in O(1) coordinates $x=(p-p_0)/s$.
- It is given the analytic gradient $g = 2J^{\mathsf T}r$, so it takes no tiny-step
  derivatives of its own.
- It runs with `strategy=0` (no internal numerical Hessians).
- Patch fits start from the full-sky best fit; the full-sky fit starts from a generic
  guess.

**`method="levenberg_marquardt"`.** Damped Gauss–Newton on $r$; see
[levenberg_marquardt.md](levenberg_marquardt.md). It reaches the same minimum with
3–6× fewer CAMB solves.

**Fit health.** Every fit reports `status` ("ok" or "flagged") and `flags`:
- invalid MIGRAD or EDM above tolerance;
- LM stalled or at its iteration limit;
- a validity bound reached;
- a singular Fisher matrix;
- with `run_hesse=True`, HESSE failing or disagreeing with the Gauss–Newton errors.

## 8. Error bars

For both methods, the parameter covariance is the **Gauss–Newton (Fisher)
curvature** at the minimum:

$$
\mathrm{Cov}(p) = (J^{\mathsf T}J)^{-1}
\quad(\text{priors included via their rows}).
$$

For a Gaussian likelihood this is the standard curvature estimate, exact when the
model is linear across about 1σ.

**HESSE is not used for the errors.** Minuit's HESSE takes numerical second
derivatives of the scalar χ², and on a patch these are dominated by CAMB's jumps.
- At a test minimum it failed outright, returning a forced positive-definite fallback
  with H0 ± 4×10⁻⁷. That is the origin of the "impossibly small" error bars seen
  before.
- After MIGRAD it underestimated errors by up to ~2×.
- On the full sky, where the curvature is large, HESSE and Gauss–Newton agree (H0 0.63
  vs 0.66). It remains available as a diagnostic (`run_hesse=True`).

## 9. Per-patch analysis and caching

`fitting.fit_one_patch` runs sections 2–8 for one patch mask.

**Likelihood packages.** The expensive part is spectra, workspaces and covariance:
minutes and GBs of RAM per patch at ℓmax 2011. It is stored as a likelihood package
(`save_likelihood`) containing:
- data vector, covariance and its inverse;
- bandpower windows, bin counts and effective ℓ;
- errors and f_sky;
- the data-side settings as metadata.

`fit_one_patch(..., likelihood_cache=...)` reloads the package whenever the metadata
match. Any refit (other fixed parameters, priors or minimizer) then costs only CAMB
time. Changing a data-side setting (ℓ ranges, binning, nside, beam, masks) triggers a
recompute.

**Result caches.** Per-patch fit results are cached as JSON keyed on the full fit
configuration, so an interrupted 12-patch run resumes. Results live under the
git-ignored `output/`.

**Memory.** Section 4 releases the full-sky NaMaster objects before the patches, and
each patch keeps only its windows (MBs, not GBs). Without this, a single kernel was
OOM-killed at ~27 GB.

## 10. Main assumptions and approximations

- **Gaussian likelihood** in the bandpowers, with an analytic Gaussian covariance:
  - fiducial-cosmology signal and half-difference noise;
  - no non-Gaussian, beam or calibration terms;
  - in the joint fit, same-bin TT/TE/EE correlations only.
- **τ, Σmν, r fixed.** Point sources are a pure ℓ² shot-noise term; no other
  foreground model is used.
- **Errors from local curvature.** They are reliable when the fit is good and the
  likelihood is close to Gaussian in the parameters across about 1σ.
- **Different data and covariance from the reference paper.** PR3 SMICA hm1×hm2
  instead of PR4 SEVEM A×B, and an analytic covariance instead of 600 E2E simulations
  (hence no debiasing). See [../PIPELINE.md](PIPELINE.md).
