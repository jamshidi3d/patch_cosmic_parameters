"""
power_spectrum.py
------------------
Pseudo-Cl power spectrum estimation on a masked HEALPix patch using
NaMaster (pymaster), plus its Gaussian analytic covariance.

The three objects NaMaster needs, in order:
  1. NmtField   -- the (masked) map turned into spherical-harmonic space
  2. NmtBin     -- how multipoles are grouped into bandpowers
  3. NmtWorkspace -- the mode-coupling matrix that deconvolves the mask's
                     effect from the raw pseudo-Cl

The workspace is expensive to compute and is reused both for decoupling the
data spectrum and for pushing theory spectra through the same mode coupling
before comparing to data (see fitting.py) -- that consistency is what keeps
the chi^2 unbiased.
"""

from __future__ import annotations

import numpy as np
import pymaster as nmt


def make_bins(nside: int, bandpower_width: int, lmax: int | None = None) -> "nmt.NmtBin":
    """
    Build a linear bandpower binning scheme.

    Parameters
    ----------
    nside : int
    bandpower_width : int
        Number of ell modes per bandpower. Should be large enough to exceed
        the mode-coupling scale of the mask (roughly pi / patch_size_rad);
        too narrow a binning will leave bandpowers strongly correlated.
    lmax : int, optional
        Maximum ell to compute up to. Defaults to 3*nside - 1 (HEALPix limit).

    Returns
    -------
    bins : nmt.NmtBin
    """
    lmax = lmax if lmax is not None else 3 * nside - 1
    return nmt.NmtBin.from_lmax_linear(lmax, bandpower_width, is_Dell=False)


def build_field(
    patch: dict,
    spin: int = 0,
    beam_function: np.ndarray | None = None,
    pixel_window_function: np.ndarray | None = None,
    lmax: int | None = None,
) -> "nmt.NmtField":
    """
    Wrap a patch dict (from patch.extract_patch) into an NmtField.

    Parameters
    ----------
    patch : dict
        Output of patch.extract_patch: needs 'map' and 'mask'.
    spin : int
        0 for scalar fields (temperature, convergence). Use 2 for polarization
        (pass [Q, U] as the map in that case).
    beam_function : ndarray, optional
        Instrumental beam transfer function B_ell, indexed from ell=0 (e.g.
        healpy.gauss_beam). If None, no beam correction is applied.
    pixel_window_function : ndarray, optional
        HEALPix pixel window transfer function, indexed from ell=0 (e.g.
        healpy.pixwin(nside, lmax=lmax)). Matters increasingly at high ell /
        high nside, where finite pixel size smooths the map. If None, no
        pixel-window correction is applied.
    lmax : int, optional
        Maximum ell for this field's harmonic transform. NaMaster requires
        beam/pixel-window arrays to reach the map's own HEALPix limit
        (3*nside-1) unless this is set, so pass the same lmax used for
        make_bins()/camb_cl() here whenever beam_function or
        pixel_window_function is truncated to that lmax. If None, defaults
        to NaMaster's own choice (3*nside-1).

    Both are folded into NmtField's `beam` argument as their product -- this
    is what NaMaster actually deconvolves when the workspace's mode-coupling
    matrix is built from this field, so the resulting decoupled bandpowers
    come out already beam- and pixel-window-corrected. NaMaster does not
    apply the pixel window automatically, unlike the beam -- hence passing
    both here rather than relying on NmtField's beam alone.

    Returns
    -------
    field : nmt.NmtField
    """
    if (
        beam_function is not None
        and pixel_window_function is not None
        and beam_function.shape != pixel_window_function.shape
    ):
        raise ValueError(
            f"beam_function and pixel_window_function shape mismatch: "
            f"{beam_function.shape} vs {pixel_window_function.shape}"
        )

    transfer_function = None
    if beam_function is not None and pixel_window_function is not None:
        transfer_function = beam_function * pixel_window_function
    elif beam_function is not None:
        transfer_function = beam_function
    elif pixel_window_function is not None:
        transfer_function = pixel_window_function

    map_list = [patch["map"]] if spin == 0 else patch["map"]
    return nmt.NmtField(patch["mask"], map_list, spin=spin, beam=transfer_function, lmax=lmax)


def compute_power_spectrum(
    field_a: "nmt.NmtField",
    bins: "nmt.NmtBin",
    field_b: "nmt.NmtField | None" = None,
    workspace: "nmt.NmtWorkspace | None" = None,
) -> dict:
    """
    Compute the mode-decoupled pseudo-Cl bandpowers for a field, or the
    cross-spectrum between two fields sharing the same mask/footprint.

    Parameters
    ----------
    field_a : nmt.NmtField
        From build_field().
    bins : nmt.NmtBin
        From make_bins().
    field_b : nmt.NmtField, optional
        A second field (e.g. an independent split/half-mission map) to
        cross-correlate with field_a. Cross-correlating two noise-independent
        splits cancels the noise bias that an auto-spectrum would carry.
        If None, computes the auto-spectrum of field_a with itself.
    workspace : nmt.NmtWorkspace, optional
        Precomputed workspace (mode-coupling matrix). If None, one is built
        here -- this is the slow step (O(lmax^3)), so reuse it across calls
        with the same mask/binning rather than rebuilding every time.

    Returns
    -------
    result : dict with keys
        'ell'       : effective multipole of each bandpower
        'cl'        : decoupled Cl bandpowers
        'workspace' : the NmtWorkspace (return it so callers can reuse it,
                      e.g. to push theory spectra through the same coupling
                      in fitting.py)
    """
    if field_b is None:
        field_b = field_a

    if workspace is None:
        workspace = nmt.NmtWorkspace()
        workspace.compute_coupling_matrix(field_a, field_b, bins)

    cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
    cl_decoupled = workspace.decouple_cell(cl_coupled)[0]  # spin-0: single spectrum

    return {
        "ell": bins.get_effective_ells(),
        "cl": cl_decoupled,
        "workspace": workspace,
    }


def compute_gaussian_covariance(
    workspace: "nmt.NmtWorkspace",
    field_a: "nmt.NmtField",
    cl_theory_guess: np.ndarray,
    field_b: "nmt.NmtField | None" = None,
    noise_cl_a: np.ndarray | None = None,
    noise_cl_b: np.ndarray | None = None,
) -> np.ndarray:
    """
    Analytic Gaussian covariance of the bandpowers, following the standard
    NaMaster covariance workspace approach.

    This needs a fiducial theory Cl (full multipole range, unbinned) as an
    input -- the Gaussian covariance formula depends on the true underlying
    spectrum, not just the data. In practice, use a reasonable fiducial
    (e.g. a CAMB spectrum at fiducial cosmology) rather than the noisy data
    Cl itself, or iterate: fit once, recompute covariance at the best fit,
    refit.

    Parameters
    ----------
    workspace : nmt.NmtWorkspace
        Same workspace used to decouple the data Cl.
    field_a : nmt.NmtField
        The (first) field the spectrum was computed from (needed to rebuild
        the covariance workspace's mode-coupling machinery).
    cl_theory_guess : ndarray, shape (lmax+1,)
        Fiducial signal-only theory Cl over the full multipole range (not
        binned). Used for the field_a x field_b cross-leg, and (plus noise)
        for the two auto-legs.
    field_b : nmt.NmtField, optional
        A second field, matching whatever was passed to
        compute_power_spectrum as field_b. If None, uses field_a for both
        (auto-spectrum covariance).
    noise_cl_a, noise_cl_b : ndarray, shape (lmax+1,), optional
        Per-field noise power spectra, added only to that field's auto-leg
        (C^aa = cl_theory_guess + noise_cl_a, C^bb = cl_theory_guess +
        noise_cl_b; the field_a x field_b cross-leg stays signal-only, since
        independent splits have no noise cross-correlation). Needed whenever
        field_a/field_b are noisy and not identical (e.g. two independent
        half-mission splits cross-correlated to cancel noise bias in the
        *spectrum* -- the noise still has to enter the *covariance*, or the
        Gaussian covariance formula silently underestimates the variance at
        the ell's where noise dominates over signal, inflating chi^2 and
        biasing the fit). Omit (or leave None) only for true noise-free
        fields, or when field_b is field_a itself.

    Returns
    -------
    covariance : ndarray, shape (n_bins, n_bins)
    """
    if field_b is None:
        field_b = field_a
    if noise_cl_a is None:
        noise_cl_a = np.zeros_like(cl_theory_guess)
    if noise_cl_b is None:
        noise_cl_b = np.zeros_like(cl_theory_guess)

    cl_aa = cl_theory_guess + noise_cl_a
    cl_bb = cl_theory_guess + noise_cl_b

    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_a, field_b)

    covariance = nmt.gaussian_covariance(
        cov_workspace,
        0, 0, 0, 0,  # spins of the two fields being correlated (spin-0 x spin-0)
        [cl_aa],
        [cl_theory_guess],
        [cl_theory_guess],
        [cl_bb],
        workspace,
    )
    return covariance


def estimate_noise_cl(
    map_a: np.ndarray,
    map_b: np.ndarray,
    mask: np.ndarray,
    lmax: int,
    beam_function: np.ndarray | None = None,
    pixel_window_function: np.ndarray | None = None,
    spin: int = 0,
) -> np.ndarray:
    """
    Estimate a split's noise power spectrum from the half-difference of two
    independent noise realizations of the same sky (e.g. half-mission maps).

    (map_a - map_b) / 2 cancels the shared CMB signal and any other
    common-to-both-splits sky component, leaving (n_a - n_b) / 2 -- a pure
    noise map. Its auto-spectrum is (N_a + N_b) / 4, which is the per-split
    noise N if the two splits have comparable noise levels (the usual case
    for symmetric half-mission splits).

    This uses a quick fsky-corrected pseudo-Cl (coupled Cl / mean(mask**2))
    rather than a full NaMaster decoupling -- adequate for a noise curve
    that only needs to be smooth and roughly right, e.g. as an input to
    compute_gaussian_covariance, not for a precision spectrum in its own
    right.

    Parameters
    ----------
    map_a, map_b : ndarray
        The two independent split maps (same units, same footprint).
        Shape (npix,) for spin=0 (temperature), or a [Q, U] pair (shape
        (2, npix)) for spin=2 (polarization).
    mask : ndarray
        The (apodized) weight mask shared by both splits.
    lmax : int
        Maximum ell to estimate up to.
    beam_function, pixel_window_function : ndarray, optional
        Same transfer functions passed to build_field() for map_a/map_b --
        keeps this noise estimate on the same beam/pixel-window convention
        as cl_theory_guess before it's added to it.
    spin : int
        0 for temperature (returns the TT noise Cl), 2 for polarization
        (returns the EE noise Cl -- component 0 of NaMaster's
        [EE, EB, BE, BB] spin2 x spin2 ordering).

    Returns
    -------
    noise_cl : ndarray, shape (lmax+1,)
    """
    if spin == 0:
        half_diff = [(map_a - map_b) / 2.0]
    else:
        half_diff = [(a - b) / 2.0 for a, b in zip(map_a, map_b)]

    transfer_function = None
    if beam_function is not None and pixel_window_function is not None:
        transfer_function = beam_function * pixel_window_function
    elif beam_function is not None:
        transfer_function = beam_function
    elif pixel_window_function is not None:
        transfer_function = pixel_window_function

    # NmtField's `beam` is only ever applied inside compute_coupling_matrix,
    # never in compute_coupled_cell below -- so it has no effect here and
    # the beam/pixel-window suppression has to be divided out by hand to
    # bring this onto the same raw (unbeamed) convention as cl_theory_guess.
    field_diff = nmt.NmtField(mask, half_diff, spin=spin, lmax=lmax)
    cl_coupled = nmt.compute_coupled_cell(field_diff, field_diff)[0]  # TT or EE: component 0
    fsky = np.mean(mask ** 2)
    noise_cl = cl_coupled[: lmax + 1] / fsky
    if transfer_function is not None:
        # healpy's polarization pixel window is exactly 0 at ell=0,1
        # (undefined for a spin-2 field there) -- dividing by it would
        # inject inf/NaN that then poisons every bin once this array is
        # summed over in couple_cell/gaussian_covariance. Those ells are
        # never used downstream (every fit here starts at ell=32), so
        # just zero them instead of dividing by zero.
        tf = transfer_function[: lmax + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            noise_cl = np.where(tf != 0, noise_cl / tf ** 2, 0.0)
    return noise_cl


def compute_joint_tt_te_ee_covariance(
    workspace_tt: "nmt.NmtWorkspace",
    workspace_te: "nmt.NmtWorkspace",
    workspace_ee: "nmt.NmtWorkspace",
    field_t_a: "nmt.NmtField",
    field_t_b: "nmt.NmtField",
    field_pol_a: "nmt.NmtField",
    field_pol_b: "nmt.NmtField",
    cl_tt: np.ndarray,
    cl_te: np.ndarray,
    cl_ee: np.ndarray,
    cl_bb: np.ndarray,
    noise_tt: np.ndarray,
    noise_ee: np.ndarray,
    n_tt: int,
    n_te: int,
    n_ee: int,
) -> np.ndarray:
    """
    Joint Gaussian covariance of the stacked [TT, TE, EE] bandpowers,
    following the same-bin-only simplification of Gimeno-Amo et al. 2025
    (arXiv:2504.05597, sec. 2.2): "we neglect all off-diagonal terms in
    all the blocks of the covariance matrix, focusing only on the
    variances and covariances between TT, TE, EE within the same bin."
    That means every one of the 6 TT/TE/EE block-pairs only contributes
    its diagonal (bin i of spectrum A with bin i of spectrum B); the
    result is a matrix that is block-diagonal *per ell-bin* (up to 3x3),
    not a dense matrix.

    Conventions (fixed throughout, matching the rest of the pipeline):
    TT = hm1_T x hm2_T, TE = hm1_T x hm2_Pol, EE = hm1_Pol x hm2_Pol --
    always a cross between the two splits, never an auto-spectrum of a
    single split, so the spectra themselves stay noise-bias-free. The
    per-split noise still has to enter this *covariance*, exactly as in
    compute_gaussian_covariance. Temperature-noise and polarization-noise
    are assumed uncorrelated even within the same split (the standard
    assumption for this kind of Gaussian covariance forecast -- T and P
    noise come from different linear combinations of detector timelines).

    A single NmtCovarianceWorkspace is reused for all 6 blocks: its
    coupling coefficients depend only on the fields' masks, and this
    pipeline uses the same apodized mask for temperature and polarization.

    Parameters
    ----------
    workspace_tt, workspace_te, workspace_ee : nmt.NmtWorkspace
        From compute_power_spectrum for the TT (spin0 x spin0), TE
        (spin0 x spin2), and EE (spin2 x spin2) bandpowers respectively.
        All three must share the *same* underlying NmtBin (NaMaster
        requires a workspace's fields and bins to have identical lmax, so
        TT/TE/EE can't each use their own truncated lmax here -- they
        share one bins object at the largest lmax, and are truncated to
        their own bandpower count via n_tt/n_te/n_ee below instead).
    field_t_a, field_t_b : nmt.NmtField
        The two splits' temperature fields (spin 0).
    field_pol_a, field_pol_b : nmt.NmtField
        The two splits' polarization fields (spin 2).
    cl_tt, cl_te, cl_ee, cl_bb : ndarray, shape (lmax+1,)
        Fiducial signal-only theory spectra, full multipole range.
    noise_tt, noise_ee : ndarray, shape (lmax+1,)
        Per-split noise spectra (same convention as compute_gaussian_covariance).
    n_tt, n_te, n_ee : int
        Final bandpower count to keep for each spectrum (after dropping
        bin 0), matching the arrays passed into FitData.

    Returns
    -------
    covariance : ndarray, shape (n_tt + n_te + n_ee, n_tt + n_te + n_ee)
        Stacked [TT, TE, EE] joint covariance, each spectrum's bin 0
        (ell=[2,31]) already dropped to match FitData.cl_data.
    """
    zero = np.zeros_like(cl_tt)
    ctt_auto = cl_tt + noise_tt
    ctt_cross = cl_tt
    cte = [cl_te, zero]
    cee_auto = [cl_ee + noise_ee, zero, zero, cl_bb]
    cee_cross = [cl_ee, zero, zero, cl_bb]

    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_t_a, field_t_b)

    def block(spins, cla1b1, cla1b2, cla2b1, cla2b2, wa, wb, n_keep):
        cov = nmt.gaussian_covariance(
            cov_workspace, *spins, cla1b1, cla1b2, cla2b1, cla2b2, wa, wb
        )
        # NaMaster's flattened covariance index is band-major
        # (index = band*ncls + component), *not* component-major --
        # verified empirically (a clean no-mask, zero-BB synthetic EE
        # field gives a smoothly decreasing bandpower-variance sequence
        # only under a stride-ncls extraction, never under a leading
        # [:n_bands, :n_bands] slice, which mixes bands and components).
        # Component 0 (TT/TE/EE) of each band sits at indices
        # 0, ncls, 2*ncls, ... .
        idx_a = np.arange(0, wa.wsp.ncls * wa.wsp.bin.n_bands, wa.wsp.ncls)
        idx_b = np.arange(0, wb.wsp.ncls * wb.wsp.bin.n_bands, wb.wsp.ncls)
        same_bin_diag = np.diag(cov[np.ix_(idx_a, idx_b)])
        return same_bin_diag[1:1 + n_keep]  # drop bin 0, keep this pair's overlap

    var_tt = block((0, 0, 0, 0), [ctt_auto], [ctt_cross], [ctt_cross], [ctt_auto],
                   workspace_tt, workspace_tt, n_tt)
    cov_tt_te = block((0, 0, 0, 2), [ctt_auto], cte, [ctt_cross], cte,
                      workspace_tt, workspace_te, min(n_tt, n_te))
    cov_tt_ee = block((0, 0, 2, 2), cte, cte, cte, cte,
                      workspace_tt, workspace_ee, min(n_tt, n_ee))
    var_te = block((0, 2, 0, 2), [ctt_auto], cte, cte, cee_auto,
                   workspace_te, workspace_te, n_te)
    cov_te_ee = block((0, 2, 2, 2), cte, cte, cee_cross, cee_auto,
                      workspace_te, workspace_ee, min(n_te, n_ee))
    var_ee = block((2, 2, 2, 2), cee_auto, cee_cross, cee_cross, cee_auto,
                   workspace_ee, workspace_ee, n_ee)

    n_total = n_tt + n_te + n_ee
    cov = np.zeros((n_total, n_total))

    tt_sl = slice(0, n_tt)
    te_sl = slice(n_tt, n_tt + n_te)
    ee_sl = slice(n_tt + n_te, n_total)

    cov[tt_sl, tt_sl] = np.diag(var_tt)
    cov[te_sl, te_sl] = np.diag(var_te)
    cov[ee_sl, ee_sl] = np.diag(var_ee)

    n = len(cov_tt_te)
    cov[0:n, n_tt:n_tt + n] = np.diag(cov_tt_te)
    cov[n_tt:n_tt + n, 0:n] = np.diag(cov_tt_te)

    n = len(cov_tt_ee)
    cov[0:n, n_tt + n_te:n_tt + n_te + n] = np.diag(cov_tt_ee)
    cov[n_tt + n_te:n_tt + n_te + n, 0:n] = np.diag(cov_tt_ee)

    n = len(cov_te_ee)
    cov[n_tt:n_tt + n, n_tt + n_te:n_tt + n_te + n] = np.diag(cov_te_ee)
    cov[n_tt + n_te:n_tt + n_te + n, n_tt:n_tt + n] = np.diag(cov_te_ee)

    return cov


def compute_errors(covariance: np.ndarray) -> np.ndarray:
    """Diagonal sqrt(covariance) -- the per-bandpower 1-sigma error bars."""
    diag = np.diag(covariance)
    if np.any(diag <= 0):
        raise ValueError("non-positive variance in covariance diagonal -- check inputs")
    return np.sqrt(diag)
