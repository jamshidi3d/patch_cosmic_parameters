# Errors on the CMB power spectrum: where they come from and how they are computed

This document derives the uncertainty on a measured CMB bandpower from first
principles, and explains how `cosmo_patch` computes it (`power_spectrum.py`):
- **§1–§6:** the physics and statistics;
- **§7:** the implementation, step by step;
- **§8:** verification against simulations, including **three implementation issues**
  found while writing this document and **fixed on 2026-10-06** (§8.2), with the
  validation of the fix (§8.3);
- **§9:** the approximations that remain.

Notation:
- $a_{\ell m}$: spherical-harmonic coefficients;
- $C_\ell$: the true (theory) angular power spectrum of the sky;
- $N_\ell$: noise power;
- $B_\ell$: beam × pixel window;
- $w(\hat n)$: the mask (weight map);
- $b$: a bandpower (bin of width $\Delta\ell$);
- $D_\ell=\ell(\ell+1)C_\ell/2\pi$.

---

## 1. The sky is one realization of a random field: cosmic variance

The temperature anisotropy is expanded as
$\Delta T(\hat n)=\sum_{\ell m} a_{\ell m} Y_{\ell m}(\hat n)$.
In ΛCDM with Gaussian initial conditions, the $a_{\ell m}$ are **Gaussian random
variables** with zero mean and

$$
\langle a_{\ell m}\, a^*_{\ell' m'}\rangle = C_\ell\,\delta_{\ell\ell'}\delta_{mm'} .
$$

$C_\ell$ is a property of the *ensemble* of universes. We observe one sky, and from it
we estimate $C_\ell$ by averaging over the $2\ell+1$ values of $m$:

$$
\hat C_\ell = \frac{1}{2\ell+1}\sum_{m=-\ell}^{\ell} |a_{\ell m}|^2 .
$$

This estimator is unbiased, $\langle\hat C_\ell\rangle = C_\ell$. Its variance follows
from Wick's (Isserlis') theorem for Gaussian variables,
$\langle x_1x_2x_3x_4\rangle = \langle x_1x_2\rangle\langle x_3x_4\rangle +
\langle x_1x_3\rangle\langle x_2x_4\rangle + \langle x_1x_4\rangle\langle x_2x_3\rangle$.
Using $a_{\ell,-m}=(-1)^m a^*_{\ell m}$ (the field is real):

$$
\mathrm{Var}(\hat C_\ell) = \frac{2}{2\ell+1}\, C_\ell^2 .
$$

Equivalently, $(2\ell+1)\hat C_\ell/C_\ell$ is $\chi^2$-distributed with $2\ell+1$
degrees of freedom. This is **cosmic variance**: an irreducible uncertainty from having
only $2\ell+1$ independent modes per multipole on one sky. It is no instrumental
defect. It dominates at low ℓ, at ℓ = 2 for example ±63% on $C_2$.

## 2. Instrument noise, and why cross-spectra between splits are used

A map is signal plus noise, $a_{\ell m} = s_{\ell m} + n_{\ell m}$. The noise is
independent of the signal, with power $N_\ell$.

**Auto-spectrum.** $\langle\hat C_\ell\rangle = C_\ell + N_\ell$. This is biased, so
$N_\ell$ would have to be known to percent precision and subtracted. Its variance is

$$
\mathrm{Var}(\hat C^{\rm auto}_\ell) = \frac{2}{2\ell+1}(C_\ell+N_\ell)^2 .
$$

**Cross-spectrum of two independent splits.** Planck's two half-missions observe the
same sky with **independent noise**: $a^{(i)}_{\ell m} = s_{\ell m}+n^{(i)}_{\ell m}$,
with $\langle n^{(1)}n^{(2)*}\rangle=0$. The cross-spectrum

$$
\hat C^{12}_\ell = \frac{1}{2\ell+1}\sum_m \mathrm{Re}\big(a^{(1)}_{\ell m}a^{(2)*}_{\ell m}\big)
$$

has $\langle\hat C^{12}_\ell\rangle = C_\ell$ exactly: **no noise bias**. That is why
the pipeline uses TT = hm1×hm2 (and similarly for TE and EE). Noise still adds
*scatter*. Wick's theorem with the leg spectra $C^{11}=C+N_1$, $C^{22}=C+N_2$,
$C^{12}=C$ gives

$$
\mathrm{Var}(\hat C^{12}_\ell) = \frac{(C_\ell+N_{1,\ell})(C_\ell+N_{2,\ell}) + C_\ell^2}{2\ell+1} .
$$

For equal split noise this is $[(C+N)^2+C^2]/(2\ell+1)$: between the auto-spectrum of
one split, $2(C+N)^2$, and pure cosmic variance, $2C^2$. Here $N$ is the noise **of one
split**. The full-mission map, the average of the two halves, has noise $N/2$.

**General rule.** For fields $a, b, c, d$, at most four of them, with all pairwise
spectra $C^{xy}_\ell$:

$$
\mathrm{Cov}\big(\hat C^{ab}_\ell, \hat C^{cd}_{\ell'}\big) =
\frac{C^{ac}_\ell C^{bd}_\ell + C^{ad}_\ell C^{bc}_\ell}{2\ell+1}\,\delta_{\ell\ell'} .
\tag{1}
$$

A leg pairing the *same* split (e.g. $T^{(1)}T^{(1)}$) contains that split's noise. A
leg pairing *different* splits is signal only. Every covariance used in this pipeline
is an instance of (1).

## 3. Polarization: TE and EE

$Q$ and $U$ form a spin-2 field, decomposed into E and B modes with their own
$a^{E}_{\ell m}, a^{B}_{\ell m}$. In ΛCDM, $C^{TB}=C^{EB}=0$, and $C^{BB}$ (lensing
plus $r=0.01$) is small. The pipeline measures

$$
TT = T^{(1)}\!\times T^{(2)},\qquad TE = T^{(1)}\!\times E^{(2)},\qquad EE = E^{(1)}\!\times E^{(2)} .
$$

Applying (1) to these crosses gives the six blocks of the joint covariance, per
multipole (T and P noise taken as uncorrelated):

| Block | $(2\ell+1)\times$ covariance |
|---|---|
| TT–TT | $(C^{TT}+N^T)^2 + (C^{TT})^2$ |
| TE–TE | $(C^{TT}+N^T)(C^{EE}+N^E) + (C^{TE})^2$ |
| EE–EE | $(C^{EE}+N^E)^2 + (C^{EE})^2$ |
| TT–TE | $(C^{TT}+N^T)\,C^{TE} + C^{TT}C^{TE}$ |
| TT–EE | $2\,(C^{TE})^2$ |
| TE–EE | $C^{TE}(C^{EE}+N^E) + C^{TE}C^{EE}$ |

TT, TE and EE are measured on the same sky, so they are **correlated**. The
cross-blocks are not small; for example, TT–EE is set by $C^{TE}$. The joint
likelihood must include them. The pipeline does, at the same multipole bin.

## 4. Beam and pixel window: why errors grow at small scales

The telescope and the pixelization smooth the sky. The harmonic coefficients *in the
map* are

$$
a^{\rm map}_{\ell m} = B_\ell\, s_{\ell m} + n_{\ell m},
\qquad B_\ell = b_\ell\, p_\ell ,
$$

where $b_\ell=\exp[-\tfrac12\ell(\ell+1)\sigma_b^2]$ is the Gaussian beam
(FWHM 5′, $\sigma_b=\mathrm{FWHM}/\sqrt{8\ln 2}$) and $p_\ell$ is the HEALPix pixel
window. The noise is *not* smoothed by the beam, because it enters after it.

There are two consistent ways to write the error.

1. **Map level.** Signal power $B^2_\ell C_\ell$, noise power $N^{\rm map}_\ell$:
   $\mathrm{Var}(\hat C^{\rm map,12}_\ell) = [(B^2C+N^{\rm map})^2+(B^2C)^2]/(2\ell+1)$.
2. **Sky level**, after dividing the map spectrum by $B^2_\ell$, as the estimator does.
   Define the deconvolved noise $\tilde N_\ell = N^{\rm map}_\ell/B_\ell^2$. Then
   $\mathrm{Var}(\hat C^{12}_\ell)=[(C+\tilde N)^2+C^2]/(2\ell+1)$, i.e. the map-level
   variance divided by $B^4_\ell$.

Both give the same answer. The physics: $B_\ell$ falls like a Gaussian, so the
deconvolved noise $\tilde N_\ell$ grows like $e^{\ell^2\sigma_b^2}$. For SMICA at
5′, $B_\ell^2$ is 0.80 at ℓ = 700, 0.44 at ℓ = 1370 and 0.18 at ℓ = 2000. This is why
the TT error bars blow up beyond ℓ ~ 1500, and EE's (lower signal) sooner.

The two descriptions must not be **mixed**. A covariance code that expects map-level
spectra and divides by the beam itself must not be given sky-level spectra (see §8.2).

## 5. Partial sky: masks, mode coupling, and the effective number of modes

The Galaxy and point sources are masked, and the patches cover 2–8% of the sky. Then
the $a_{\ell m}$ of the masked map, $\tilde a_{\ell m}=\int w\,\Delta T\,Y^*_{\ell m}$,
are no longer independent across ℓ. Their power, the **pseudo-$C_\ell$**, mixes
multipoles:

$$
\langle\tilde C_\ell\rangle = \sum_{\ell'} M_{\ell\ell'}\,B^2_{\ell'}C_{\ell'},\qquad
M_{\ell\ell'} = \frac{2\ell'+1}{4\pi}\sum_{L}(2L+1)\,W_L
\begin{pmatrix}\ell&\ell'&L\\0&0&0\end{pmatrix}^2 ,
$$

where $W_L$ is the power spectrum of the mask. For spin-2 fields the analogous
matrices also mix E and B. $M$ is a convolution kernel in ℓ with a width
$\delta\ell\sim\pi/\theta_{\rm mask}$:
- a few multipoles for the full-sky mask;
- tens of multipoles for an $N_{\rm side}=1$ patch, which spans about 60°.

Apodization keeps the kernel compact.

**The MASTER estimator** (Hivon et al. 2002), implemented by NaMaster, first bins
$\tilde C_\ell$ into bandpowers. It then inverts the *binned* coupling matrix, beam
included:

$$
\hat C_b = \sum_{b'}\big(\mathcal M^{-1}\big)_{bb'}\,\tilde C_{b'},\qquad
\mathcal M_{bb'} = \sum_{\ell\in b}\sum_{\ell'\in b'}\tfrac{1}{\Delta\ell}\,M_{\ell\ell'}B^2_{\ell'} .
$$

This makes $\langle\hat C_b\rangle$ an unbiased estimate of the binned sky spectrum,
up to the bandpower window functions.

**How many modes survive: the Knox approximation.** A mask keeping a fraction
$f_{\rm sky}$ of the sky leaves roughly $f_{\rm sky}(2\ell+1)$ independent modes per
ℓ. A bin of width $\Delta\ell$ then has about
$\nu_b = (2\ell_b+1)\,\Delta\ell\,f_{\rm sky}$ modes, giving the standard
order-of-magnitude formula (Knox 1995):

$$
\sigma(\hat C^{12}_b)\simeq\sqrt{\frac{(C_b+\tilde N_b)^2 + C_b^2}{(2\ell_b+1)\,\Delta\ell\,f_{\rm sky}}} .
\tag{2}
$$

For a weighted mask, the relevant sky fraction for variances is
$f_{\rm sky}^{\rm eff} = \langle w^2\rangle^2/\langle w^4\rangle$. That is 0.753 (T)
and 0.771 (P) for the full-sky mask here, and 0.076 for patch 9.

Two consequences:
- **Patch errors are larger.** A patch error bar is $\sqrt{0.75/0.075}\approx 3.2$
  times the full-sky one, for the same ℓ and Δℓ. Patch 9's TT bins carry ~3.6% errors
  at ℓ ≈ 700 against 1.1% on the full sky.
- **Neighbouring bins are correlated.** Bins closer than the kernel width share modes.
  For a patch (δℓ of a few tens) with Δℓ = 30, adjacent-bin correlations are not
  negligible.

## 6. The analytic Gaussian covariance used here (NaMaster)

Eq. (2) ignores the shape of the mask. NaMaster's `gaussian_covariance` (García-García
et al. 2019, "improved narrow-kernel approximation") computes the covariance of the
**pseudo**-spectra:
- It applies eq. (1) at the level of the masked fields. Each pair of legs is weighted
  by a coupling matrix built from the **product masks** of the two fields in that leg,
  e.g. $w_a w_c$ and $w_b w_d$ (`NmtCovarianceWorkspace`).
- It assumes the $C_\ell$ vary slowly across the kernel width.

The result is then propagated through the same decoupling as the data:

$$
\mathrm{Cov}(\hat C_b,\hat C_{b'}) = \sum_{q,q'}(\mathcal M^{-1})_{bq}\,
\mathrm{Cov}(\tilde C_q,\tilde C_{q'})\,(\mathcal M^{-1})_{b'q'} .
$$

Because that decoupling divides by $B^2_\ell$ (it is built into $\mathcal M$), the
**input spectra must be those of the fields as they are in the maps**: beam-convolved
signal $B^2C$ plus map-level noise. This is checked by simulation in §8.1.

NaMaster returns the covariance with a band-major flattened index
(`band*ncls + component`). The TT, TE or EE component of each band is taken with
stride `ncls`.

## 7. What the pipeline does, step by step

1. **Signal model for the covariance.** A fiducial CAMB spectrum, Planck 2018
   parameters, lensed, μK², in the *sky-level* convention (no beam). It is computed
   once (`cl_fiducial`), so the covariance does not change with the fit parameters.
   That is the standard choice, since the data constrain the parameters well enough.
2. **Noise model** (`power_spectrum.estimate_noise_cl`):
   - The half-difference map $(m^{(1)}-m^{(2)})/2$ cancels the sky, CMB and
     foregrounds alike, and keeps only noise.
   - Its pseudo-$C_\ell$ is divided by $\langle w^2\rangle$, the standard correction
     that turns the pseudo-power of a weighted white-noise field into the full-sky
     noise power.
   - That power is $(N_1+N_2)/4$, so it is **doubled** to give the per-split noise
     $(N_1+N_2)/2$.
   - It is then divided by $B^2_\ell$, giving a *sky-level* per-split noise
     $\tilde N_\ell$.
   - Done separately for T (intensity mask) and E (polarization mask). B-mode noise is
     taken equal to E-mode noise.
3. **Leg spectra.** They are built at sky level, then converted to **map level** by
   multiplying with the transfer functions of the leg's two fields ($B_T^2$, $B_TB_P$
   or $B_P^2$; `transfer_t`, `transfer_pol`):
   - auto legs (same split): $(C+\tilde N)\,B_aB_b$;
   - cross legs (different splits): $C\,B_aB_b$;
   - EE legs carry the four spin-2 components $[EE, EB, BE, BB]$, and noise enters both
     EE and BB of the auto legs.
   - NaMaster's decoupling then divides by the beam once.
4. **Covariance blocks.**
   - **Joint fit:** the six blocks of §3 via `nmt.gaussian_covariance`, each with the
     coupling coefficients of its own T/P mask combination. Only the **same-bin**
     terms are kept (the simplification of arXiv:2504.05597), so the matrix is
     block-diagonal per bin (3×3).
   - **TT-only fit:** keeps the full bin-to-bin TT covariance
     (`compute_gaussian_covariance`).
   - The first bin, ℓ = 2–31, is dropped.
   - A covariance with condition number > 1e12 is rejected.
5. **Error bars on the plots:** $\sigma_b=\sqrt{\mathrm{Cov}_{bb}}$
   (`compute_errors`), shown as $D$ errors $\ell_b(\ell_b+1)\sigma_b/2\pi$ at the
   effective multipole $\ell_b$ of each bin.
6. **In the fit:** the full matrix enters
   $\chi^2=(d-t)^{\mathsf T}C^{-1}(d-t)$, cross-blocks included. The *parameter*
   errors then follow from its curvature ([theory.md](theory.md) §8).

## 8. Verification, and issues found

### 8.1 Monte-Carlo test of the covariance conventions

Setup (`nside` 256, ℓ ≤ 600, Δℓ = 30):
- an $N_{\rm side}=1$ patch mask, apodized;
- a 40′ beam (chosen so that $B^2_\ell$ spans 1 → 0 over the range);
- two splits with white noise.

300 CMB+noise realizations went through the same NaMaster cross-spectrum. The empirical
bandpower variance was compared with `gaussian_covariance` fed two ways:

| ℓ | $B^2_\ell$ | MC var ÷ Gaussian (inputs **sky-level**, as the pipeline does) | MC var ÷ Gaussian (inputs **beam-convolved**) |
|---|---|---|---|
| 106 | 0.75 | 0.47 | 0.82 |
| 166 | 0.49 | 0.21 | 0.85 |
| 226 | 0.27 | 0.076 | 1.03 |
| 286 | 0.12 | 0.015 | 0.98 |
| 346 | 0.04 | 0.002 | 0.94 |
| 466 | ≈0 | ≈0 | 1.01 |

- The **beam-convolved** inputs reproduce the simulated scatter to within the MC noise
  (±8% for 300 sims).
- The sky-level inputs overestimate the variance by $1/B^4_\ell$.
- The bandpowers themselves are unbiased: their mean ÷ theory is 1.00 ± 0.01 for ℓ ≥ 60.

In the same simulations, the half-difference noise estimate came out at
**0.498 × the true per-split noise**.

### 8.2 Issues found, and fixed

These issues were found while preparing this document and **fixed on 2026-10-06**
(`power_spectrum.py`). The earlier fits carried them: the full sky and the 12 patches of
the 2026-10-06 run (kept in `output/old_results/before_covariance_fix_2026-10-06/`).

**(A) The beam was divided out twice in the covariance (largest effect).**
`gaussian_covariance` received sky-level spectra ($C$ and $\tilde N$), and the workspace
then divided by $B^2_\ell$ again. So σ was too large by $1/B^2_\ell$:

| | ℓ ≈ 700 | ℓ ≈ 1370 | ℓ ≈ 2000 |
|---|---|---|---|
| expected $1/B^2_\ell$ (5′ beam × pixel window) | 1.25 | 2.3 | 5.7 |
| measured σ(old) ÷ σ(Knox, eq. 2), full-sky TT | 1.42 | 2.30 | 5.13 |
| same, patch 9 TT | 1.43 | 2.35 | 5.35 |

Consequences:
- The high-ℓ bandpowers were under-weighted.
- χ² was too low where the beam matters.
- The parameter errors were inflated.

*Fix:* every leg spectrum is multiplied by the transfer functions of its two fields
before NaMaster sees it (`transfer_t`, `transfer_pol` in
`compute_joint_tt_te_ee_covariance`, which are required arguments now; `transfer_a`,
`transfer_b` in `compute_gaussian_covariance`).

**(B) The noise level was half the per-split noise.** The half-difference
$(m^{(1)}-m^{(2)})/2$ has noise power $(N_1+N_2)/4 = N/2$, not $N$. The MC gave 0.498.
- The effect was largest in polarization, which is noise-dominated much earlier: with
  the correct noise, σ is larger by ×1.5–2.0 in EE, ×1.2–1.6 in TE, and ×1.6 in TT at
  ℓ ≈ 2000.
- *Fix:* `estimate_noise_cl` returns twice the half-difference power, i.e. the per-split
  noise.

**(C) There was no B-mode noise in the EE blocks.** On a cut sky, B-mode noise leaks
into the E bandpower variance.
- *Fix:* the auto legs' BB component now carries noise (`noise_bb`, default
  $\tilde N^E$).

**Net effect of the old code.** (A) inflated the errors and (B) shrank them:
- TT was too large by about ×1.4 at ℓ ≈ 700, ×2.2 at ℓ ≈ 1370 and ×3.2 at ℓ ≈ 2000;
- EE was up to ~25% too *small* at intermediate ℓ.

### 8.3 Validation of the fix

The fixed public functions were tested by Monte Carlo:
- `compute_joint_tt_te_ee_covariance` with `estimate_noise_cl` noise, against 300
  simulations;
- TT, TE and EE cross-spectra between two noisy splits;
- 40′ beam with pixel windows;
- different T and P patch masks.

Results:
- The noise estimate is **1.008 (T) and 1.009 (E)** × the true per-split noise.
- The bandpower variances agree with the simulated scatter at every ℓ to within the MC
  uncertainty (±0.08), including where $B^2_\ell\to 0$:

| ℓ | var MC ÷ analytic: TT | TE | EE |
|---|---|---|---|
| 76 | 1.01 | 1.07 | 0.99 |
| 196 | 1.02 | 1.13 | 1.02 |
| 316 | 0.97 | 1.08 | 0.98 |
| 436 | 1.02 | 1.06 | 1.20 |
| 556 | 1.19 | 0.99 | 1.17 |

- The same-bin TT–TE, TT–EE and TE–EE correlation coefficients agree with the analytic
  ones within the MC resolution (±0.06), with a few ~2σ deviations among 27, as expected
  by chance.

### 8.4 The first bin: Gaussian approximation is conservative

In the same simulations, the ℓ ≈ 46 bin had an MC variance of only **0.35×** the
Gaussian prediction, *with either input convention*. In the real data, the first kept
TT bin's error is 3.7× (full sky) and 8× (patch 9) the Knox value.
- **Likely cause:** the narrow-kernel approximation fails where $C_\ell$ changes steeply
  across the mask kernel, and where the dropped ℓ = 2–31 bin leaks power.
- **Effect:** the lowest bandpowers are given *larger* errors than they should. This is
  conservative, but it down-weights them.

## 9. Remaining approximations

Approximations kept even after the fix:
- **Gaussianity.** The $a_{\ell m}$ are treated as Gaussian. Lensing (the
  lensing-induced trispectrum and super-sample terms), point sources and residual
  foregrounds add non-Gaussian covariance, mostly at high ℓ in TT.
- **Homogeneous, white-ish noise.** One noise spectrum per mask. Planck's noise depth
  varies with the scanning (deep near the ecliptic poles, shallow along the ecliptic),
  so per-patch noise is captured only through the half-difference map *of that patch*,
  and not its anisotropy within the patch. T and P noise are assumed uncorrelated.
- **Fiducial cosmology.** The covariance is evaluated at Planck 2018 parameters, not at
  each fit's best fit. The effect is small when the fits stay near ΛCDM.
- **Same-bin approximation** (joint fit). Correlations between different bins are
  dropped. They are small on the full sky but real for patches (§5).
- **No systematic error budget.** No beam, calibration or transfer-function
  uncertainties.

**How the reference paper differs.** arXiv:2504.05597 estimates the covariance from
600 end-to-end simulations of the PR4 data. Those capture the beam, the real noise
(including its anisotropy and correlations), the non-Gaussian terms and the estimator's
exact behaviour, and they also provide the debiasing. The analytic covariance here is a
cheaper stand-in, accurate to the level validated in §8.3.

## References

- Knox L., 1995, *Phys. Rev. D* 52, 4307: mode counting, eq. (2).
- Hivon E. et al., 2002, *ApJ* 567, 2: MASTER pseudo-$C_\ell$ estimator.
- Efstathiou G., 2004, *MNRAS* 349, 603: pseudo-$C_\ell$ covariances, cross-spectra.
- Alonso D., Sanchez J., Slosar A., 2019, *MNRAS* 484, 4127: NaMaster.
- García-García C., Alonso D., Bellini E., 2019, *JCAP* 11, 043: Gaussian covariance
  (narrow-kernel approximation).
- Gimeno-Amo C. et al., 2025, arXiv:2504.05597: the reference analysis.
