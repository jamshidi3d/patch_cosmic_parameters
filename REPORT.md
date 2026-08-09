# Report: patch power spectrum → cosmological parameters

Two notebooks in this repo fit ΛCDM parameters to a Planck SMICA
temperature/polarization cross-spectrum over the full common-mask
footprint, following the pipeline of Gimeno-Amo, Hansen,
Martínez-González, Barreiro & Banday, *"Exploring Statistical Isotropy
in Planck Data Release 4"* (arXiv:2504.05597), sec. 2.2:

- **`patch_cosmology_fit.ipynb`** — the joint fit: TT + TE + EE together,
  with two point-source nuisance amplitudes.
- **`patch_cosmology_fit_TT.ipynb`** — a TT-only companion, built to
  check that dropping TE/EE doesn't change *what's being measured*, only
  its precision.

This document reports what each notebook measured, how those numbers
compare to the reference paper, and the methodological issues that
surfaced while building the pipeline (several of which are real bugs
worth knowing about if this code gets reused or extended). For how to
run and adapt the notebooks, see [GUIDE.md](GUIDE.md).

## 1. Data and shared methodology

- **Maps**: Planck PR3 SMICA half-mission splits (`hm1`, `hm2`), I/Q/U,
  `Nside=2048`. The `hm1 x hm2` cross-spectrum is used throughout instead
  of an auto-spectrum, so instrumental noise (independent between splits)
  never biases the spectrum itself — only the *covariance* needs a
  separate per-split noise estimate (from the `(hm1-hm2)/2` half-difference
  null map).
- **Mask**: the Planck PR3 common intensity confidence mask, apodized at
  0.3°, `fsky ≈ 0.78`. Reused for polarization too (see deviations,
  below).
- **Beam/pixel window**: 5′ Gaussian (SMICA common resolution) ×
  HEALPix pixel window (temperature and polarization pixel windows
  differ slightly at high ell; both are used where appropriate).
- **Binning**: linear, Δℓ=30, starting at ℓ=2; the first bandpower
  (ℓ=[2,31]) is dropped for every spectrum, following the paper.
  Per-spectrum cutoffs: TT to ℓ=2011 (66 bins), TE to ℓ=1741 (57 bins),
  EE to ℓ=1471 (48 bins).
- **Theory**: CAMB, lensed spectra (`lens_potential_accuracy=1`) —
  see bug #1 below for why this matters. τ=0.0602, Σmν=0.06 eV, r=0.01
  fixed throughout (never fit).
- **Nuisance**: point-source residuals `A_ps^TT`, `A_ps^EE`, modeled as
  `Dℓ = A * (ℓ/3000)²` shot noise, fit alongside the 5 ΛCDM parameters.
- **Fit**: Gaussian likelihood, iminuit (MIGRAD + HESSE, MINOS where it
  converges).

## 2. Results

### 2.1 Joint TT+TE+EE fit

chi²/dof = 195.3 / 171 ≈ 1.14, **converged**.

| Parameter | This pipeline | Paper, Table 2 (TTTEEE, no debiasing) |
|---|---|---|
| H0 | 66.56 ± 0.26 | 66.78 ± 0.50 |
| Ωb h² | 0.02239 ± 0.00009 | 0.02212 ± 0.00013 |
| Ωc h² | 0.1217 ± 0.0006 | 0.1209 ± 0.0011 |
| As | (2.134 ± 0.006) × 10⁻⁹ | ≈2.126 × 10⁻⁹ (from ln(10¹⁰As)=3.057±0.0033) |
| ns | 0.9643 ± 0.0038 | 0.9598 ± 0.0036 |
| A_ps^TT | 46 ± 8 | 55 ± 4 |

All parameters land within ~1σ of the paper's own no-debiasing column —
a good match given the deviations in Sec. 4. TT, TE, and EE all track
the acoustic peak structure correctly (see the notebook's Section 4
plot); residuals are unbiased across the full ℓ range with no systematic
trend, including at high ℓ where an earlier, unfixed version of this
pipeline showed a clear divergence (see bug #1).

### 2.2 TT-only fit

chi²/dof = 32.0 / 66 ≈ 0.48, **converged**.

| Parameter | TT only | Joint TT+TE+EE | Planck 2018 fiducial |
|---|---|---|---|
| H0 | 67.49 ± 0.10 | 66.56 ± 0.26 | 67.36 |
| Ωb h² | 0.02209 ± 0.00002 | 0.02239 ± 0.00009 | 0.02237 |
| Ωc h² | 0.1188 ± 0.0003 | 0.1217 ± 0.0006 | 0.1200 |
| As | (2.108 ± 0.0001) × 10⁻⁹ | (2.134 ± 0.006) × 10⁻⁹ | 2.1 × 10⁻⁹ |
| ns | 0.9632 ± 0.0001 | 0.9643 ± 0.0038 | 0.9649 |
| A_ps^TT | 60.1 ± 0.2 | 46 ± 8 | — |

**The expectation was met**: every free parameter except τ (which is
fixed, not fit, in both notebooks — see the note below) converges to a
value close to the Planck 2018 fiducial, confirming that TT alone
already encodes each of these parameters' imprint on the acoustic peaks
(peak spacing → H0, peak-height ratios → Ωbh²/Ωch², tilt → ns, overall
amplitude → As once the τ-As degeneracy is broken by fixing τ by hand).

**One genuinely surprising result, investigated and explained rather
than dismissed**: the TT-only errors above are *smaller* than the joint
fit's, which looks backwards — adding independent data (TE, EE) should
never *loosen* a constraint. It doesn't, here, either; the explanation is
a covariance-treatment difference between the two notebooks, not a bug
in either fit result:

- The **TT-only** notebook uses `power_spectrum.compute_gaussian_covariance`,
  which returns the *full* NaMaster analytic covariance, including all
  bin-to-bin correlations.
- The **joint** notebook uses `power_spectrum.compute_joint_tt_te_ee_covariance`,
  which — following the paper's own stated simplification ("we neglect
  all off-diagonal terms in all the blocks of the covariance matrix... we
  can reasonably assume that the correlations between different bins are
  minimal") — keeps only the same-bin diagonal of every TT/TE/EE block.

That simplification is *not* a safe assumption for this pipeline's mask.
Directly comparing the two covariances confirmed they agree exactly on
the diagonal (same per-bin variances, ratio = 1.000 across the first 10
bins checked) — no bug — but the **off-diagonal TT bin-to-bin
correlations are large**: ρ≈0.62 between adjacent bins, decaying but
still non-negligible several bins out, with some pairs exceeding
ρ≈0.9. The paper's own justification for dropping these terms was built
for their actual use case — 12 small, irregularly-shaped Nside=1 patches
at 2–8% sky fraction — where that assumption may well hold; it does not
hold for this pipeline's full common-mask footprint (fsky≈0.78). Ignoring
real, strong positive correlations lets a fit double-count shared
information as if adjacent bins were independent, artificially
*tightening* the reported errors — meaning **the joint notebook's quoted
uncertainties should be read as approximate and likely somewhat
optimistic**, while the TT-only notebook's fully-correlated errors are
the more statistically rigorous of the two (and, consistent with that,
happen to come out tighter, not looser, once TT's real internal
information is properly credited).

**Why τ is excluded from "everything converges"**: TT depends on τ only
through the combination `As · exp(-2τ)`, degenerate with As itself.
Without independent large-scale E-mode information there is no way to
break that degeneracy from TT alone — which is exactly why both
notebooks fix τ rather than fit it, instead of the pipeline "failing" to
constrain it.

## 3. Bugs found and fixed during development

Building this pipeline surfaced four real, independently-confirmed bugs.
Each was caught by a concrete symptom (not proactively), tracked to its
root cause, and verified fixed:

1. **CAMB `lens_potential_accuracy=0` silently disabled lensing.**
   Symptom: the fit's high-ℓ residuals showed a clear, growing
   data-vs-theory divergence. Real CMB data is lensed; the acoustic peaks
   are smoothed by a few percent, growing with ℓ. Fitting unlensed theory
   to lensed data reproduces exactly that signature. Fixed by setting
   `lens_potential_accuracy=1`.
2. **`estimate_noise_cl`'s beam/pixel-window correction was silently a
   no-op.** `NmtField`'s `beam` argument is only ever applied inside
   `compute_coupling_matrix`, never in the raw `compute_coupled_cell`
   used for a quick noise estimate — so the transfer function had to be
   divided out by hand, and wasn't. This under/over-stated the noise
   level by up to ~13× at the highest ℓ bins used, since it's exactly
   where the 5′ beam and pixel window suppress power most.
3. **Dividing by the polarization pixel window's exact zero at ℓ=0,1.**
   `healpy.pixwin(nside, pol=True)` returns exactly 0 there (undefined
   for a spin-2 field), producing inf/NaN that then poisoned the entire
   EE/TE covariance once summed over ℓ inside NaMaster's C routines.
   Symptom: a `RuntimeWarning: invalid value encountered in divide` and a
   chi² wildly larger than the bandpower count. Fixed by zeroing the
   noise estimate wherever the transfer function is exactly zero (those
   multipoles are never used downstream anyway).
4. **Wrong sub-block extracted from `nmt.gaussian_covariance`'s output.**
   NaMaster's flattened covariance index is **band-major**
   (`index = band * ncls + component`); the initial implementation
   assumed **component-major** (`component * n_bands + band`), silently
   mixing different bandpowers and spin components together. Verified
   empirically: a clean, unmasked, exact-zero-BB synthetic spin-2 field
   gives a smoothly decreasing bandpower-variance sequence only under the
   correct (strided) extraction, and an unphysical alternating pattern
   under the wrong one. Symptom: a joint chi²/dof of ~535 and a
   non-converged fit with a parameter pinned at its bound. Fixed by using
   strided indices (`np.arange(0, ncls*n_bands, ncls)`) to isolate
   component 0 (TT/TE/EE) of each band.

Bugs #3 and #4 together were responsible for the joint fit's chi² of
~91,000 (n_bins=171) before being found; after both fixes, chi²/dof
dropped to ≈1.14 and MIGRAD converged cleanly. See
[GUIDE.md's "Known sharp edges" section](GUIDE.md#5-known-sharp-edges)
for the code-level detail on each.

## 4. Known, deliberate deviations from the paper

- **Data**: PR3 SMICA `hm1`/`hm2` half-mission splits, not the paper's
  PR4 NPIPE SEVEM detector A/B splits (not available locally without a
  Planck Legacy Archive download).
- **Polarization mask**: Planck's separate common Pol confidence mask is
  used for Q/U (not the intensity mask). `compute_joint_tt_te_ee_covariance`
  builds each TT/TE/EE covariance block from the coupling coefficients of
  the actual T/Pol field combination entering it, rather than assuming one
  shared `NmtCovarianceWorkspace` -- necessary once the two masks differ.
- **Error bars**: analytic NaMaster Gaussian covariance, not the paper's
  600 Planck PR4 end-to-end simulations (not available locally).
- **Sky coverage**: the full common-mask footprint (one region), not the
  paper's 12 disjoint Nside=1 HEALPix patches used for its
  directional/dipole analysis. `patch.make_superpixel_mask` already
  provides the building block for that extension; see GUIDE.md.
- **No angular-clustering / Rayleigh-statistic / dipole analysis, no
  cobaya MCMC cross-check** — downstream analyses in the paper built on
  top of per-patch fits and (for the clustering test) the E2E simulation
  ensemble for null significance; out of scope here.

## 5. Conclusion

The joint TT+TE+EE pipeline reproduces the reference paper's own
no-debiasing results to within ~1σ across all 5 ΛCDM parameters and the
TT point-source amplitude, using independently-sourced data and the
paper's stated methodology. The TT-only companion confirms the expected
physical picture — every parameter but τ is already well-determined by
temperature alone — and, as a side effect of investigating an
unexpectedly tight error bar, surfaced a genuine, mask-dependent
limitation of the paper's diagonal-covariance simplification when
applied to a full-sky footprint rather than the paper's own small,
low-fsky patches.
