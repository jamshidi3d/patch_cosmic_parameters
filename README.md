# Guide: using `cosmo_patch` and the notebooks

Hands-on guide to the `cosmo_patch` package and the notebooks built on it. They
fit ΛCDM parameters to Planck TT/TE/EE power spectra on the full sky and,
independently, in the 12 HEALPix Nside=1 sky patches, following the pipeline of
Gimeno-Amo et al. 2025 ([arXiv:2504.05597](https://arxiv.org/abs/2504.05597)).

Documentation:

| Document | What it covers |
|---|---|
| [docs/theory.md](docs/theory.md) | How everything is implemented: pseudo-Cℓ / bandpower windows, covariance, model, likelihood, minimization, error bars, caching, assumptions |
| [docs/power_spectrum_errors.md](docs/power_spectrum_errors.md) | Physics and math of the bandpower error bars: cosmic variance, noise and split cross-spectra, beam, partial sky, NaMaster Gaussian covariance; the covariance fixes of 2026-10-06 and their MC validation |
| [docs/point_source_amplitudes.md](docs/point_source_amplitudes.md) | Physics of the A_ps nuisance amplitudes: unresolved radio/dusty galaxies, shot-noise ℓ² shape, why they bias ns/Ωbh²/H0 if ignored, what the fits find, their role on patches |
| [docs/levenberg_marquardt.md](docs/levenberg_marquardt.md) | Theory of the Levenberg–Marquardt fitting method (`method="levenberg_marquardt"`) |
| [PIPELINE.md](docs/PIPELINE.md) | The pipeline in one page, stored results, and the differences from arXiv:2504.05597 (with the full-sky comparison) |
| [FIT_CONVERGENCE_FIXES.md](docs/FIT_CONVERGENCE_FIXES.md) | Why the per-patch fits used to stall at bounds or report fake-tiny errors, what was changed, test results, checklist |
| [REPORT.md](docs/REPORT.md) | Earlier narrative write-up of the measurements |
| [COLAB.md](docs/COLAB.md) / [WSL.md](docs/WSL.md) | Running on Google Colab (`patch_cosmology_fit_colab.ipynb`) or on Windows via WSL2 (RAM management) |

## 1. Setup

```bash
conda install -c conda-forge namaster    # provides pymaster (compiled, needs cfitsio/GSL/FFTW)
pip install healpy camb iminuit scipy matplotlib nbformat nbclient
pip install -e .                          # installs cosmo_patch itself, editable
```

### Data

The notebooks expect these files in `input/` (Planck Legacy Archive, PR3/2018):

| File | Used for |
|---|---|
| `COM_CMB_IQU-smica_2048_R3.00_hm1.fits` | half-mission 1 map (I, Q, U) |
| `COM_CMB_IQU-smica_2048_R3.00_hm2.fits` | half-mission 2 map (I, Q, U) |
| `COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits` | intensity (temperature) sky mask |
| `COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits` | polarization sky mask (joint TT+TE+EE notebook only) |

They are read with `healpy.read_map`, converted `K_CMB -> uK_CMB` (`*1e6`), and kept
in `RING` ordering.

### Outputs

Everything the notebooks write goes to `output/`, which is **git-ignored**. It can be
regenerated, and it is also a cache, so keep it locally:

- `output/fullsky_*`: full-sky fit (`.json`, `.txt`, spectrum plot) and
  `fullsky_vs_paper.txt`;
- `output/patched_<tag>/`: per-patch result JSONs (resumable cache), parameter
  plot/dump, and TT, TE, EE per-patch spectrum figures;
- `output/likelihoods/`: **likelihood packages** (spectra, covariance, bandpower
  windows) for the full sky and every patch. A refit loads them instead of
  recomputing NaMaster;
- `output/runs/<date>_<tag>/`: executed copies of production runs.

## 2. Package layout

```
cosmo_patch/
├── patch.py               # mask building, apodization, patch validation
├── power_spectrum.py      # NaMaster fields, binning, pseudo-Cl, noise, covariance
├── fitting.py             # CAMB theory, bandpower-window binning, FitProblem,
│                          #   iminuit fit, one-patch pipeline, likelihood/result caches
└── levenberg_marquardt.py # alternative minimizer (method="levenberg_marquardt")
```

`patch` doesn't know about spectra, and `power_spectrum` doesn't know about
cosmological parameters. `fitting` holds the model and the fit. Its `fit_one_patch`
chains the other two modules for one sky region.

### `patch.py`

| Function | Use it for |
|---|---|
| `make_circular_mask(nside, center_lonlat_deg, radius_deg)` | a small circular sky patch |
| `make_superpixel_mask(nside, low_nside, low_pix)` | one of the `12 * low_nside**2` HEALPix superpixels: the building block of the 12-patch (`low_nside=1`) analysis |
| `apodize_mask(mask, apodize_deg, method="C1")` | taper a binary mask's edges before NaMaster sees it (**always do this**; a hard edge biases the pseudo-Cl through mode coupling) |
| `extract_patch(sky_map, mask)` | package `(map, mask)` with `f_sky` and shape validation; `sky_map` is `(npix,)` for T or `(2, npix)` `[Q, U]` for polarization |

### `power_spectrum.py`

| Function | Use it for |
|---|---|
| `make_bins(nside, bandpower_width, lmax=None)` | a linear `NmtBin` (width in ℓ, starting at ℓ=2) |
| `build_field(patch, spin, beam_function=None, pixel_window_function=None, lmax=None)` | wrap a patch dict into an `NmtField` (`spin=0` T, `spin=2` Q/U); beam × pixel window is deconvolved |
| `compute_power_spectrum(field_a, field_b, bins, workspace=None)` | mode-decoupled bandpowers of a cross- (or auto-) spectrum. Use independent splits to avoid noise bias |
| `estimate_noise_cl(map_a, map_b, mask, lmax, ..., spin=0)` | per-split noise Cℓ from the half-difference `(a - b)/2`, needed for the *covariance* |
| `compute_gaussian_covariance(workspace, field_a, cl_theory_guess, field_b=None, noise_cl_a=None, noise_cl_b=None)` | analytic Gaussian covariance of a **single** spectrum (full bin-to-bin; TT-only notebook) |
| `compute_joint_tt_te_ee_covariance(...)` | **joint** TT+TE+EE covariance with same-bin cross terms only (the paper's simplification) |
| `compute_errors(covariance)` | `sqrt(diag(covariance))` |

### `fitting.py`

| Function | Use it for |
|---|---|
| `camb_cl(H0, ombh2, omch2, As, ns, lmax, tau=0.0602, mnu=0.06, r=0.01)` | one CAMB solve: lensed `{"TT","EE","BB","TE"}` raw Cℓ in μK². Memoized on exact parameter values (`camb_solve_count()` reports solves) |
| `point_source_dl_template(lmax, amplitude, ell_pivot=3000.0)` | the `Dl = A (ell/3000)^2` point-source nuisance template, as raw Cℓ |
| `bandpower_windows(workspace, n_keep)` | a workspace's bandpower windows (bins 1..n_keep). Theory binned with them equals NaMaster couple+decouple to machine precision, at MBs instead of GBs |
| `decouple_theory(workspace, cl_theory)` | push theory through a live workspace (used for plots when the workspace is at hand) |
| `FitData` / `theory_vector` / `chi_square` | **joint TT+TE+EE** data bundle: data, inverse covariance, bandpower windows (pass workspaces and they are extracted, or pass `windows_*` directly), binned theory, χ² |
| `FitDataTT` / `theory_vector_tt` / `chi_square_tt` | the separate **TT-only** path (no TE/EE, no `A_ps_EE`) |
| `fit_parameters` / `fit_parameters_tt(data, initial_guess, bounds=None, fixed=None, priors=None, initial_step=None, method="iminuit", compute_minos=False, run_hesse=False)` | **one** minimizer pass, no restarts. `method="iminuit"` (MIGRAD with supplied gradient) or `"levenberg_marquardt"`. `bounds` = wide validity limits (smooth χ² wall, never Minuit `limits`); `priors={name: (mean, sigma)}` adds Gaussian priors. Errors = Gauss–Newton curvature; `run_hesse=True` adds HESSE as a diagnostic. Returns `best_fit`, `errors`, `covariance`, `free_params`, `chi2`/`chi2_prior`, `status` (`"ok"`/`"flagged"`), `flags`, `params_at_bound`, `n_iter`, `n_camb_solves`, `minuit` (iminuit only) |
| `FitProblem` | the least-squares problem both methods minimize: whitened residual + prior rows + wall rows, fixed-step central-difference Jacobian (`DEFAULT_STEP`), Gauss–Newton covariance |
| `fit_one_patch(maps_hm1, maps_hm2, mask_t, mask_pol, nside, lmax_tt, lmax_te, lmax_ee, bandpower_width, beam_function, pixel_window_t, pixel_window_pol, cl_fiducial, initial_guess, bounds=None, fixed=None, likelihood_cache=None, likelihood_meta=None, **fit_kwargs)` | full TT+TE+EE pipeline for one region (spectra, covariance, fit). With `likelihood_cache` it stores/reloads the likelihood package and skips NaMaster when the meta matches |
| `save_likelihood` / `load_likelihood(path, meta=None)` | write/read a likelihood package (`.npz`): returns `(FitData, extra)` |
| `save_patch_result` / `load_patch_result(path, config)` | per-patch fit-result cache (JSON), keyed on the fit configuration |
| `n_bins_upto(lmax_cut, bandpower_width=30)` | bandpowers kept per spectrum after dropping bin 0 |
| `levenberg_marquardt.fit_levenberg_marquardt(problem)` | the LM minimizer itself (see [docs/levenberg_marquardt.md](docs/levenberg_marquardt.md)) |

## 3. Running the notebooks

- **`patch_cosmology_fit.ipynb`** is the joint TT+TE+EE pipeline:
  - **Section 1** loads the maps and masks.
  - **Section 2** builds the full-sky spectra and joint covariance (~10–15 min).
  - **Section 3** runs the full-sky fit and compares it with the paper's Table 2 ("No
    debiasing"), writing `fullsky_vs_paper.txt`.
  - **Section 4** fits all 12 patches with the paper's settings (~6–8 min per patch the
    first time, then from cache), and draws the TT, TE and EE per-patch figures plus
    the parameter plot with the full-sky ±1σ band.
  - **Section 5** draws the H0–Ωch² ellipses from the fit covariance.
- **`patch_cosmology_fit_TT.ipynb`** is the TT-only companion: full sky (full
  bin-to-bin TT covariance) and its own 12-patch loop. The loop runs TT up to
  ℓ = 2508 with Δℓ = 90, an A_ps_TT prior from the full sky, and, as currently
  configured, `fixed_params = {"ns", "As"}`. It has not yet been aligned with the
  paper's settings.
- **`patch_cosmology_fit_colab.ipynb`** has Colab setup cells, then a verbatim copy of
  the joint notebook (keep it synced when the joint notebook changes).

Configuration switches sit in the notebooks' config cells:

| Switch | Meaning |
|---|---|
| `FIT_METHOD` | `"iminuit"` (default) or `"levenberg_marquardt"`; the LM outputs get a `_levenberg_marquardt` suffix |
| `fixed_params` (Section 4) | parameters held at the full-sky best fit in every patch; the output folder is tagged `fixed(...)` |
| `USE_PATCH_PRIORS` (Section 4) | Gaussian A_ps priors from the full-sky fit (off: the paper fits them freely) |
| `LMAX_*_PATCH`, `BANDPOWER_WIDTH_PATCH` | patch multipole ranges / binning (default: the paper's TT 2011, TE 1741, EE 1471, Δℓ = 30) |
| `VALIDITY_BOUNDS` | where CAMB is trusted (smooth wall, not a prior) |

Section 4 needs only Section 1 and the cached full-sky result, so after a first full
run you can restart, run Section 1, and jump to Section 4.

Headless execution (for Colab or WSL see [COLAB.md](docs/COLAB.md) / [WSL.md](docs/WSL.md)):

```python
import nbformat
from nbclient import NotebookClient

nb = nbformat.read("patch_cosmology_fit.ipynb", as_version=4)
NotebookClient(nb, timeout=None, kernel_name="python3").execute()
nbformat.write(nb, "output/runs/<date>_<tag>/patch_cosmology_fit.executed.ipynb")
```

To run only some sections, drop cells from `nb.cells` by `id` before `.execute()`.
Writing to a copy keeps the source notebook clean.

## 4. Adapting the pipeline

**Fixing parameters.** In a direct call, `fit_parameters(..., fixed={"ns", "As"})`
holds them at their `initial_guess` value; they are removed from the free set. In
Section 4, set `fixed_params`. Fixed values are seeded from the full-sky best fit, and
the stored likelihood packages are reused, so only CAMB time is spent.

**Refitting from stored likelihoods.** For example:

```python
fit_data, extra = fitting.load_likelihood("output/likelihoods/patches_.../patch_03.npz")
res = fitting.fit_parameters(fit_data, initial_guess, bounds, fixed={"ns"}, method="levenberg_marquardt")
```

**Changing the multipole range.** All fields and the one shared `NmtBin` use the
largest ℓmax, as NaMaster requires. Each spectrum's bandpowers are then truncated to
its own cut (`n_bins_upto`, `n_tt`/`n_te`/`n_ee` in `FitData`). Changing the ranges
changes the likelihood-package metadata, so the spectra are recomputed automatically.

**TE-only or EE-only fits** need their own small `FitData*` / `theory_vector_*` pair,
mirroring the TT-only path. `compute_gaussian_covariance` already works for any single
spin combination.

## 5. Known sharp edges

- **`NmtField`'s `beam` is only used in `compute_coupling_matrix`**, never in
  `compute_coupled_cell`. A coupled Cℓ computed by hand (as `estimate_noise_cl` does)
  must have the beam and pixel window divided out explicitly.
- **`hp.pixwin(nside, pol=True)` is exactly 0 at ℓ = 0, 1** for polarization.
  Dividing by it unguarded injects inf/NaN into every bin (`estimate_noise_cl` guards
  it).
- **`nmt.gaussian_covariance` output is band-major** (`index = band*ncls + component`).
  Take component 0 with stride `ncls`, not a leading slice.
- **CAMB's `lens_potential_accuracy=0` returns unlensed spectra.** Real data are
  lensed; use ≥ 1.
- **Don't use Minuit `limits` for these fits.** The sin transform has zero slope at the
  bound, so a parameter sitting on a limit gets a collapsed error (seen:
  `A_ps_TT = 200 ± 7.6e-05`, `ns ± 3e-05`). Bounds here are a smooth χ² wall, and a fit
  reaching one is flagged.
- **Don't trust HESSE on a CAMB χ².** CAMB's χ² has step-like numerical jumps of
  ~1e-4. HESSE's tiny-step second derivatives see them as curvature: on a patch it
  failed outright (H0 ± 4e-7) or underestimated errors by up to 2×. Errors come from
  the Gauss–Newton curvature with steps of ~0.2–1σ; HESSE is opt-in via
  `run_hesse=True`.
- **Point-source amplitudes are flat directions if a patch is cut at low ℓmax**
  (≤ A/9 of the signal at ℓ ≤ 1000). With the paper's ranges (ℓ up to 2011) the data
  constrain them. Otherwise use `USE_PATCH_PRIORS = True`.
- **MINOS dominates the cost** of a fit whose χ² is a CAMB solve. It is opt-in
  (`compute_minos=False`).
- **Memory at ℓmax 2011.** Each patch builds full-size NaMaster workspaces, so keep the
  full-sky ones out of memory (Section 4 frees them) and don't hold workspaces per
  patch (`FitData` keeps only windows). A single kernel holding both was OOM-killed at
  ~27 GB.
- **A figure open in Windows Photos can make `savefig` fail** (`OSError: [Errno 22]`)
  when the notebook overwrites it from WSL. Close the viewer before a long run. The fit
  results are saved before the figures, so a rerun only redraws.
- **Orphaned `nbclient` / Jupyter kernels** keep competing for CPU. Check
  `ps aux | grep ipykernel` if a run is unexpectedly slow.
