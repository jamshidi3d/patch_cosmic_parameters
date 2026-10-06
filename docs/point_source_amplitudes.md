# The point-source amplitudes $A_{\rm ps}^{TT}$ and $A_{\rm ps}^{EE}$: physics and why they matter

Every fit in this pipeline has two nuisance parameters besides the five ΛCDM ones:

$$
D^{\rm ps}_\ell = A_{\rm ps}\Big(\frac{\ell}{3000}\Big)^2
\quad\Longleftrightarrow\quad
C^{\rm ps}_\ell = \frac{2\pi A_{\rm ps}}{3000^2}\,\frac{\ell}{\ell+1}\simeq \text{const},
$$

added to the TT ($A_{\rm ps}^{TT}$) and EE ($A_{\rm ps}^{EE}$) theory spectra in μK²
(`fitting.point_source_dl_template`), following arXiv:2504.05597. This note explains:
- what physical signal they stand for;
- why that signal has this shape;
- why the fit needs them;
- what they do on small sky patches.

## 1. What they represent: unresolved extragalactic sources

Besides the CMB, the microwave sky contains millions of **compact extragalactic
sources**, too small to be resolved by Planck's 5′ beam:
- **Radio sources:** AGN, blazars, radio galaxies. Synchrotron emission, bright at low
  frequencies (≲ 100 GHz), often a few percent linearly polarized.
- **Dusty star-forming galaxies:** the sources of the cosmic infrared background (CIB).
  Thermal dust emission, rising steeply with frequency and dominant at ≳ 200 GHz,
  polarized only at the ~1% level.

The brightest of them (above roughly 100–200 mJy at the CMB frequencies) are
**detected and masked**; the Planck common masks include point-source holes. The far
more numerous faint sources below the detection threshold stay in the map. Component
separation (SMICA here, SEVEM in the paper) cannot remove them fully: their spectra
differ from source to source, and they are not a single coherent component. So the
cleaned CMB map keeps a **residual of unresolved sources**. $A_{\rm ps}$ is its
amplitude, in the cleaned map's own units.

The thermal and kinetic Sunyaev–Zel'dovich effects (from galaxy clusters) and the
*clustered* part of the CIB also add small-scale power. This model does not include
them separately: whatever of their residual has a similar ℓ-shape is absorbed into
$A_{\rm ps}$ as well. "Point-source amplitude" is thus best read as **an effective
small-scale foreground residual**.

## 2. Why the shape is $C_\ell = $ constant (shot noise)

Take sources scattered over the sky as a **Poisson process**: each one independent,
number counts $dN/dS$ per steradian per unit flux density $S$. Converting flux to
temperature with the derivative of the blackbody, $\partial B_\nu/\partial T$, the
temperature field of the sources is a sum of delta functions. Its power spectrum has
two parts.

**(a) A Poisson (shot-noise) term.** Each source's contribution is uncorrelated with
every other, so the spectrum is **white**, independent of ℓ:

$$
C^{\rm Poisson}_\ell = \Big(\frac{\partial B_\nu}{\partial T}\Big)^{-2}
\int_0^{S_{\rm cut}} S^2\,\frac{dN}{dS}\,dS .
$$

The integral runs up to the masking threshold $S_{\rm cut}$. Masking more sources
(lower $S_{\rm cut}$) lowers $A_{\rm ps}$, and the integral is dominated by the
brightest *unmasked* sources.

**(b) A clustering term.** Galaxies trace large-scale structure, which adds a
component that falls with ℓ (for the CIB, $D_\ell\propto\ell^{\sim0.8}$ rather than
$\ell^2$). It matters for the CIB at high frequency and is neglected here.

A constant $C_\ell$ means $D_\ell=\ell(\ell+1)C_\ell/2\pi\propto\ell^2$, hence the
template. The pivot ℓ = 3000 is a convention: $A_{\rm ps}$ is the residual's $D_\ell$
at ℓ = 3000, where Planck foreground amplitudes are traditionally quoted. The sources
are seen through the same beam as the CMB, so the template is beam-deconvolved like
the CMB theory.

**In polarization.** Sources are polarized at a few percent (radio) or ~1% (dust), with
random orientations, so E and B get equal shares. Their EE shot-noise power is
therefore of order $\langle\Pi^2\rangle/2$ times the TT one, i.e. well below
$10^{-3}$ of it. A pure point-source $A_{\rm ps}^{EE}$ should be ≲ 0.1 μK², tiny. The
fitted $A_{\rm ps}^{EE}$ is mainly a catch-all for other small-scale polarized residuals
(§5).

## 3. Why the fit needs them

The CMB spectrum and the source residual have opposite behaviour at small scales:
- **CMB:** $D^{\rm CMB}_\ell$ falls exponentially beyond ℓ ~ 1000 because of
  **Silk damping**: photon diffusion erases anisotropies below the diffusion length at
  recombination.
- **Point sources:** $D^{\rm ps}_\ell$ *rises* like ℓ².

Their ratio grows fast. With the fitted $A_{\rm ps}^{TT}\approx 57$ μK²:

| ℓ | $D^{\rm CMB}_\ell$ [μK²] | $(\ell/3000)^2$ | $D^{\rm ps}_\ell$ [μK²] | ps / CMB |
|---|---|---|---|---|
| 500 | ≈ 2500 | 0.028 | 1.6 | 0.06% |
| 1000 | ≈ 1050 | 0.111 | 6.3 | 0.6% |
| 1500 | ≈ 680 | 0.25 | 14 | 2% |
| 2000 | ≈ 230 | 0.45 | 26 | 11% |
| 3000 | ≈ 40–50 | 1 | 57 | > 100% |

At ℓ ≈ 2000, the edge of the TT range used here, unmodelled sources would be an excess
of ~10%, against bandpower errors of a few percent on the full sky. Ignoring them biases
the cosmology through the parameters that shape the damping tail:
- **$n_s$:** the tilt. Extra small-scale power mimics a bluer spectrum. In the patch fits
  the correlation between $A_{\rm ps}^{TT}$ and $n_s$ is **−0.51**: if $A_{\rm ps}$ is
  too low, $n_s$ comes out too high.
- **$\Omega_bh^2$ and $H_0$:** these set the relative heights of the acoustic peaks and
  the damping scale. Correlations with $A_{\rm ps}^{TT}$ are −0.39 and −0.25 in the
  patch fits.
- **$A_s$:** less directly, through the overall amplitude.

Fitting $A_{\rm ps}$ alongside the cosmology lets the data separate the ℓ²-rising
residual from the CMB's damping-tail shape. It also propagates the remaining degeneracy
into honest, larger error bars on $n_s$, $\Omega_bh^2$ and $H_0$. Fixing it at a wrong
value would bias them; dropping it would bias them and underestimate their errors.

## 4. What the fits find

**Full sky** (TT+TE+EE, ℓ ≤ 2011 for TT, after the covariance fix of 2026-10-06):

| | this pipeline (PR3 SMICA hm1×hm2) | arXiv:2504.05597 (PR4 SEVEM A×B, no debias) |
|---|---|---|
| $A_{\rm ps}^{TT}$ | 57.0 ± 4.2 μK² | 55 ± 4 μK² |
| $A_{\rm ps}^{EE}$ | 4.4 ± 1.5 μK² | 0 ± 1 μK² |

$A_{\rm ps}^{TT}$ agrees with the paper to 0.5σ, with the same precision. Two different
component-separation methods, data releases and splits leave about the same
unresolved-source residual in TT. That is what one expects: both use the same PR3 masks,
so they share $S_{\rm cut}$. Before the covariance fix, the over-inflated high-ℓ errors
let $A_{\rm ps}^{TT}$ float to 47 ± 9 (see
[power_spectrum_errors.md](power_spectrum_errors.md) §8.2).

$A_{\rm ps}^{EE}$ differs at the ~3σ level. As argued in §2, real point sources cannot
produce 4 μK² in EE. The nonzero value points to a small-scale polarized residual
specific to SMICA PR3: Galactic dust, the different noise properties of PR3 vs PR4, or
the EE covariance approximations. SEVEM PR4 shows none.

## 5. On small sky patches

**Can a patch constrain them?** That depends on how far in ℓ the patch spectra reach:

| patch ℓmax (TT) | template weight $(\ell_{\max}/3000)^2$ | consequence |
|---|---|---|
| 1000 | 0.11 | $A_{\rm ps}$ is a **flat direction**: in the earlier ℓ ≤ 1000 setup it ran to its bound in nearly every patch and corrupted the error bars ([../FIT_CONVERGENCE_FIXES.md](FIT_CONVERGENCE_FIXES.md) §1) |
| 2011 (current, as in the paper) | 0.45 | measurable per patch: e.g. patch 0, $A_{\rm ps}^{TT}=52\pm14$ μK² |

With a low patch ℓmax, the physically motivated remedy is a **Gaussian prior from the
full-sky fit** (`USE_PATCH_PRIORS = True`). The residual source level is a property of
the source population and the mask threshold, so it should be nearly the same
everywhere.

**Should $A_{\rm ps}$ vary between patches?**
- **Expected to be nearly uniform.** Extragalactic source counts are statistically
  isotropic, and the masking threshold and component-separation weights are the same
  over the sky.
- **What can make it vary:** Poisson fluctuations in the number of bright unmasked
  sources (a patch is ~7% of the sky); real large-scale structure in the CIB; and
  regionally varying **Galactic** residuals near the mask edges, which the fit absorbs
  into $A_{\rm ps}$ (especially $A_{\rm ps}^{EE}$).
- **So a patch with an outlying $A_{\rm ps}$** is a hint of a local foreground residual
  rather than of new cosmology. It is a useful diagnostic next to that patch's ΛCDM
  parameters, because via the $n_s$–$A_{\rm ps}$ correlation such a residual also
  shifts $n_s$, $\Omega_bh^2$ and $H_0$.

**Why it matters for directional (isotropy) tests.** The patch analysis looks for
direction-dependent cosmological parameters. A foreground residual that differs between
patches, if not absorbed by $A_{\rm ps}$, would show up as a spurious directional
variation of $n_s$ or $H_0$. Fitting $A_{\rm ps}$ per patch, as here and in the paper,
is what separates the two.

## 6. Limitations of this model

- **One power law.** It captures shot noise only, with no separate clustered-CIB
  ($\ell^{0.8}$), tSZ or kSZ templates. Planck's own high-ℓ likelihoods use a
  multi-component, frequency-dependent foreground model. With a single cleaned map, as
  here, one effective amplitude is the standard simplification, and it is adequate for
  ℓ ≲ 2000.
- **No TE term.** Source polarization is uncorrelated with temperature on average, so
  TE gets nothing.
- **Unbounded amplitudes.** They can go slightly negative within errors (a statistical
  fluctuation, or an over-subtracted residual), which is allowed.
