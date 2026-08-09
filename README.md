# Guide: using `cosmo_patch` and the notebooks

This is a hands-on guide to the `cosmo_patch` package and the two
end-to-end notebooks built on it. For a narrative writeup of what the
notebooks measured and how those results compare to the literature, see
[REPORT.md](REPORT.md). For a quick architectural overview, see
[README.md](README.md).

## 1. Setup

```bash
conda install -c conda-forge namaster    # provides pymaster (compiled, needs cfitsio/GSL/FFTW)
pip install healpy camb iminuit matplotlib nbformat nbclient
pip install -e .                          # installs cosmo_patch itself, editable
```

### Data

Both notebooks expect these files in `input/` (Planck Legacy Archive,
PR3/2018 unless noted):

| File | Used for |
|---|---|
| `COM_CMB_IQU-smica_2048_R3.00_hm1.fits` | half-mission 1 map (I, Q, U) |
| `COM_CMB_IQU-smica_2048_R3.00_hm2.fits` | half-mission 2 map (I, Q, U) |
| `COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits` | intensity (temperature) sky mask |
| `COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits` | polarization sky mask (joint TT+TE+EE notebook only) |

Both are read with `healpy.read_map`, converted `K_CMB -> uK_CMB` (`*1e6`)
immediately after loading, and left in `RING` ordering (healpy converts
automatically from the `NESTED` storage order in these files).

## 2. Package layout

```
cosmo_patch/
├── patch.py           # mask building, apodization, patch validation
├── power_spectrum.py  # NaMaster field/binning/Cl/covariance
└── fitting.py         # CAMB wrapper, chi^2, iminuit fit
```

Each module maps to one pipeline stage and doesn't know about the others'
concerns: `patch` doesn't know about spectra, `power_spectrum` doesn't
know about cosmological parameters, `fitting` doesn't know about masks
(it only ever touches the `NmtWorkspace` objects handed to it).

### `patch.py`

| Function | Use it for |
|---|---|
| `make_circular_mask(nside, center_lonlat_deg, radius_deg)` | a small circular sky patch |
| `make_superpixel_mask(nside, low_nside, low_pix)` | one of the `12 * low_nside**2` HEALPix superpixel regions -- this is the building block for a 12-patch (`low_nside=1`) analysis like Gimeno-Amo et al. 2025's |
| `apodize_mask(mask, apodize_deg, method="C1")` | taper a binary mask's edges before it ever reaches NaMaster -- **always do this**, a hard edge biases the pseudo-Cl through mode coupling |
| `extract_patch(sky_map, mask)` | package a `(map, mask)` pair into the dict the rest of the pipeline expects, with `f_sky` and shape validation. `sky_map` can be `(npix,)` for a scalar (T) field or `(2, npix)` `[Q, U]` for a spin-2 (polarization) field |

### `power_spectrum.py`

| Function | Use it for |
|---|---|
| `make_bins(nside, bandpower_width, lmax=None)` | a linear `NmtBin` (width in ell, starting at ell=2) |
| `build_field(patch, spin, beam_function=None, pixel_window_function=None, lmax=None)` | wrap a patch dict into an `NmtField`. `spin=0` for temperature, `spin=2` for polarization |
| `compute_power_spectrum(field_a, bins, field_b=None, workspace=None)` | the mode-decoupled bandpowers of one field, or the cross-spectrum of two. Pass `field_b` = an independent split to cancel noise bias without a separate debiasing step |
| `estimate_noise_cl(map_a, map_b, mask, lmax, ..., spin=0)` | per-split noise Cl from the half-difference `(map_a - map_b)/2` -- needed for the *covariance*, even though the cross-spectrum above is already noise-bias-free |
| `compute_gaussian_covariance(workspace, field_a, cl_theory_guess, field_b=None, noise_cl_a=None, noise_cl_b=None)` | analytic Gaussian covariance for a **single** spectrum (TT-only, or any one auto/cross spectrum) |
| `compute_joint_tt_te_ee_covariance(...)` | the **joint** TT+TE+EE covariance, keeping only same-bin cross terms between the three spectra (see its docstring for the full block-by-block derivation) |
| `compute_errors(covariance)` | `sqrt(diag(covariance))` |

### `fitting.py`

| Function | Use it for |
|---|---|
| `camb_cl(H0, ombh2, omch2, As, ns, lmax, tau=0.0602, mnu=0.06, r=0.01)` | one CAMB Boltzmann solve, returns `{"TT", "EE", "BB", "TE"}` (lensed, raw `Cl` convention -- not `Dl`) |
| `point_source_dl_template(lmax, amplitude, ell_pivot=3000.0)` | the `Dl = amplitude * (ell/3000)^2` point-source nuisance template, as a raw `Cl` |
| `decouple_theory(workspace, cl_theory)` | push a **list** of theory `Cl` arrays through the same mode-coupling workspace used for the data, so they land on the same bandpowers. `cl_theory` must match the workspace's own spin combination -- `[cl_tt]` for spin0×spin0, `[cl_te, cl_tb]` for spin0×spin2, `[cl_ee, cl_eb, cl_be, cl_bb]` for spin2×spin2 |
| `FitData` / `chi_square(data, H0, ombh2, omch2, As, ns, A_ps_TT, A_ps_EE)` / `fit_parameters(...)` | the **joint TT+TE+EE** fit path: dataclass bundling data + covariance + all 3 workspaces, its chi^2, and the iminuit runner |
| `FitDataTT` / `chi_square_tt(data, H0, ombh2, omch2, As, ns, A_ps_TT)` / `fit_parameters_tt(...)` | the **TT-only** fit path -- a separate, self-contained set of functions (not the joint ones with TE/EE left empty): one workspace, no TE/EE terms, no `A_ps_EE` |
| `fit_parameters`/`fit_parameters_tt(data, initial_guess, bounds=None, fixed=None, initial_step=None, compute_minos=False)` | runs iminuit MIGRAD+HESSE (MINOS opt-in, see below), returns best-fit values, errors, chi^2, convergence flag, and the raw `Minuit` object |

## 3. Running the notebooks

Two notebooks live at the project root:

- **`patch_cosmology_fit.ipynb`** -- joint TT+TE+EE fit over the full
  common-mask footprint, following the pipeline of Gimeno-Amo et al. 2025
  (arXiv:2504.05597). The data/covariance setup (building the TE and EE
  `NmtWorkspace`s and the 6-block joint covariance) is the bulk of it,
  on the order of ~10-15 minutes; the fit itself is a handful of minutes
  on top with `compute_minos=False` (the default -- see "Known sharp
  edges" below for why this matters).
- **`patch_cosmology_fit_TT.ipynb`** -- the TT-only companion. Data and
  covariance setup is under 2 minutes (measured: ~67s -- it skips the
  polarization fields and 5 of the 6 covariance blocks). The fit itself
  is the more variable part: TT alone has a weaker/more degenerate
  likelihood surface than the joint fit (no TE/EE to break the usual
  H0-Ωch²-ns degeneracies), which without the `initial_step` hint below can send
  MIGRAD into a slow negative-curvature recovery search.

Open either in Jupyter and run top to bottom, or execute headlessly:

```python
import nbformat
from nbclient import NotebookClient

nb = nbformat.read("patch_cosmology_fit.ipynb", as_version=4)
NotebookClient(nb, timeout=1800, kernel_name="python3").execute()
nbformat.write(nb, "patch_cosmology_fit.ipynb")
```

To iterate faster while developing (e.g. checking a change to an early
cell without paying for the full fit), execute only a prefix of the
notebook by slicing `nb.cells` down to the cell `id` you care about
before calling `.execute()`, then splice the executed cells back into the
full cell list before writing -- both notebooks in this repo were built
incrementally that way.

## 4. Adapting the pipeline

**Fixing a subset of parameters.** `fit_parameters(..., fixed={"ns", "A_ps_EE"})`
holds those parameters at their `initial_guess` value via iminuit's
native mechanism, instead of fitting them. Fixed parameters still need an
entry in `initial_guess` (that's the value they're held at) but don't
need bounds.

**Running TE-only or EE-only.** There's no ready-made `FitDataTE`/`FitDataEE`
-- the TT-only notebook needed a dedicated `FitDataTT`/`chi_square_tt`/
`fit_parameters_tt` (a separate, self-contained code path, not the joint
functions with TE/EE left empty; see fitting.py's module docstring), and
a TE-only or EE-only fit would need the same treatment: its own small
`FitData*`/`chi_square_*` pair mirroring `chi_square_tt`'s shape. Nothing
stops you from reusing `decouple_theory`/`camb_cl` as-is, though --
`compute_gaussian_covariance` also works for any *single* spin
combination (point `field_a`/`field_b` at the polarization fields with
`spin=2` for EE), so the covariance side is already one function call
away.

**Changing the multipole range.** `LMAX_TT`/`LMAX_TE`/`LMAX_EE` in the
joint notebook (and the single `LMAX` in the TT-only one) set each
spectrum's cutoff. Because NaMaster requires a workspace's fields and
bins to share exactly the same `lmax`, the joint notebook builds every
field at the *largest* of the three lmax values and one shared `NmtBin`,
then truncates each spectrum's *output* bandpowers down to its own cutoff
after decoupling (see `n_bins_upto` in Section 2 of the joint notebook,
and the note in `compute_joint_tt_te_ee_covariance`'s docstring). If you
change these values, that truncation arithmetic (and the `n_tt`/`n_te`/`n_ee`
counts fed into `FitData`) has to move with them.

**Looping over the 12 Nside=1 patches** (à la the source paper's
directional/dipole analysis, not built in either notebook here): use
`patch.make_superpixel_mask(nside, low_nside=1, low_pix=i)` for
`i in range(12)`, intersect with the common mask
(`mask * make_superpixel_mask(...)`), apodize, and repeat the pipeline
per patch. Expect a much smaller `f_sky` per patch (2-8%, per the source
paper's Table 1) -- covariance/error bars scale accordingly, and you may
need a coarser `bandpower_width` to keep bins from being strongly
mask-correlated at that sky fraction.

## 5. Known sharp edges

These bit us during development; worth knowing about before you hit them
again in a modified pipeline.

- **`NmtField`'s `beam` argument is only used in `compute_coupling_matrix`,
  never in `compute_coupled_cell`.** If you're computing a raw/coupled
  Cl by hand (as `estimate_noise_cl` does) rather than going through a
  full decoupled `compute_power_spectrum` call, the beam/pixel-window
  transfer function has to be divided out explicitly -- NaMaster won't do
  it for you at that stage.
- **`hp.pixwin(nside, pol=True)` is exactly 0 at ell=0,1** for the
  polarization component (undefined there for a spin-2 field). Dividing
  by it unguarded produces inf/NaN that then poisons *every* output bin
  once summed over ell inside NaMaster's C routines -- guard the division
  (`estimate_noise_cl` does, via `np.where(tf != 0, ..., 0.0)`).
- **`nmt.gaussian_covariance`'s flattened output is band-major**
  (`index = band * ncls + component`), not component-major. Slicing the
  first `n_bands` rows/columns to isolate "component 0" (TT/TE/EE) is
  wrong and silently mixes bands and spin components together. Use
  strided indices (`np.arange(0, ncls * n_bands, ncls)`) instead --
  `compute_joint_tt_te_ee_covariance`'s `block()` helper does this, and
  its docstring links to the empirical test that pinned this down (a
  clean, unmasked, exact-zero-BB synthetic field, where the correct
  extraction gives a smoothly decreasing bandpower-variance sequence and
  the wrong one gives an obviously-wrong alternating pattern).
- **CAMB's `lens_potential_accuracy=0` silently returns unlensed spectra.**
  Fitting unlensed theory to real (lensed) data shows up as growing
  residuals at high ell, where lensing smooths the acoustic peaks most.
  Always use `lens_potential_accuracy>=1` when comparing to real data.
- **A free nuisance parameter with zero effect on chi^2 is a flat
  direction Minuit can't usefully error-estimate.** This is why the
  TT-only notebook holds `A_ps_EE` `fixed=` rather than leaving it free
  with nothing for it to act on.
- **MINOS is the dominant cost of a fit with an expensive likelihood, and
  it used to run by default whether or not anything read its output.**
  Each `chi_square` call here is a full CAMB Boltzmann solve (~1-1.5s);
  MIGRAD+HESSE need on the order of tens to low hundreds of such calls,
  but MINOS runs a *separate* bisection profile-likelihood search for
  *every free parameter*, each needing many more calls of its own. In an
  instrumented run, plain `m.migrad()` converged in ~450s while the full
  `fit_parameters()` call (which also ran HESSE and MINOS for 6 free
  parameters) exceeded 1200s and had to be killed -- and neither notebook
  even reads `result["minos"]`, only the HESSE-based `errors`. Fixed by
  making MINOS opt-in (`compute_minos=False` by default); pass
  `compute_minos=True` only if you specifically want the asymmetric
  error bars. If a fit is unexpectedly slow, check what's actually being
  computed before assuming the model/covariance is at fault.
- **MIGRAD's first numerical gradient can land in a negative-curvature
  region if parameter scales are wildly different** (H0~70 vs As~1e-9),
  triggering an expensive `NegativeG2LineSearch` recovery. Pass
  `initial_step=` to `fit_parameters` (a rough expected-uncertainty-per-
  parameter dict) to give it a well-conditioned starting point; the
  TT-only notebook's weaker, more degenerate likelihood (no TE/EE to
  break parameter degeneracies) is noticeably more prone to this than
  the joint fit.
- **An `nbclient`/Jupyter kernel that times out mid-cell can be left
  running as an orphaned process**, silently competing for CPU with
  whatever you run next. If a rerun is inexplicably slower than a nearly
  identical previous run, check `ps aux | grep ipykernel` before assuming
  the code itself regressed.
