"""
fitting.py
----------
Fit cosmological parameters to a masked-sky TT+TE+EE power spectrum:

  params --[CAMB]--> theory Cl --[NaMaster mode coupling]--> binned theory
  chi^2 = (data - binned theory)^T Cov^-1 (data - binned theory)
  params_best = argmin chi^2   (iminuit / MIGRAD)

Keeping the CAMB call and the NaMaster decoupling as separate, composable
functions matters: it lets you unit-test each independently (does CAMB
return the right theory Cl for known parameters? does decoupling reproduce
a known bandpower window?) before trusting the combination.

Follows the pipeline of Gimeno-Amo et al. 2025 (arXiv:2504.05597): joint
TT+TE+EE Gaussian likelihood, tau/mnu/r fixed, point-source residual
amplitudes (A_ps_TT, A_ps_EE) fit as nuisance parameters alongside the 5
LambdaCDM parameters.

Two independent fit paths live here, sharing only camb_cl/decouple_theory/
point_source_dl_template: FitData/chi_square/fit_parameters for the joint
TT+TE+EE fit, and FitDataTT/chi_square_tt/fit_parameters_tt for a
TT-only fit (section 4, below) -- kept separate rather than folding
TT-only as an optional code path through the joint functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import camb
from iminuit import Minuit


# ---------------------------------------------------------------------------
# 1. Theory: parameters -> CAMB Cl
# ---------------------------------------------------------------------------

def camb_cl(
    H0: float,
    ombh2: float,
    omch2: float,
    As: float,
    ns: float,
    lmax: int,
    tau: float = 0.0602,
    mnu: float = 0.06,
    r: float = 0.01,
) -> dict:
    """
    Compute the full set of CAMB angular power spectra for a given
    parameter set, in one Boltzmann solve.

    Parameters
    ----------
    H0 : float
        Hubble constant, km/s/Mpc.
    ombh2, omch2 : float
        Physical baryon and CDM density parameters (Omega * h^2).
    As : float
        Primordial scalar amplitude (e.g. ~2.1e-9).
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole to compute.
    tau : float
        Optical depth to reionization. Fixed (not fit) -- a single
        full-sky patch's polarization doesn't reach the large angular
        scales where tau is actually constrained. Default matches the
        Planck PR4 E2E simulation input (arXiv:2504.05597, sec. 2.2).
    mnu : float
        Sum of neutrino masses, eV. Fixed, same reasoning as tau.
    r : float
        Tensor-to-scalar ratio. Fixed; negligible impact at these scales
        and below Planck's sensitivity anyway.

    Returns
    -------
    cl : dict of ndarray, each shape (lmax+1,)
        Keys "TT", "EE", "BB", "TE". C_ell in the same convention
        NaMaster expects (Cl, not Dl = ell(ell+1)Cl/2pi -- CAMB's
        raw_cl=True output). Lensed spectra (lens_potential_accuracy=1) --
        real CMB data is lensed, so fitting unlensed theory to it biases
        the fit and shows up as growing residuals at high ell where
        lensing smooths the acoustic peaks most.
    """
    params = camb.CAMBparams()
    params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=mnu)
    params.InitPower.set_params(As=As, ns=ns, r=r)
    params.WantTensors = r > 0
    params.set_for_lmax(lmax, lens_potential_accuracy=1)

    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit="muK", raw_cl=True)

    total = powers["total"][: lmax + 1]  # shape (lmax+1, 4)
    idx = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}
    return {name: total[:, i] for name, i in idx.items()}


def point_source_dl_template(
    lmax: int, amplitude: float, ell_pivot: float = 3000.0
) -> np.ndarray:
    """
    Unresolved-compact-object (point source) residual, modeled as
    shot-noise: Dl = amplitude * (ell/ell_pivot)^2, converted to raw Cl
    (arXiv:2504.05597, sec. 2.2). `amplitude` is the residual's Dl value
    at ell = ell_pivot, in the same muK^2 convention as the rest of the
    pipeline.

    Returns
    -------
    cl_ps : ndarray, shape (lmax+1,)
        Zero at ell < 2 (undefined there).
    """
    ell = np.arange(lmax + 1)
    cl_ps = np.zeros(lmax + 1)
    cl_ps[2:] = amplitude * (ell[2:] / ell_pivot) ** 2 * 2 * np.pi / (ell[2:] * (ell[2:] + 1))
    return cl_ps


# ---------------------------------------------------------------------------
# 2. Push theory through the same mode coupling as the data
# ---------------------------------------------------------------------------

def decouple_theory(workspace, cl_theory: list[np.ndarray]) -> np.ndarray:
    """
    Apply the NaMaster mode-coupling workspace to a set of theory Cl's so
    they land on the same bandpowers as the (mode-decoupled) data
    spectrum.

    This step is required -- comparing a raw CAMB Cl directly to a
    mask-decoupled data Cl silently biases the fit, since the data
    bandpowers are a mask-convolved, binned version of the true sky.

    Parameters
    ----------
    workspace : nmt.NmtWorkspace
        The same workspace used to decouple the data (see power_spectrum.py).
    cl_theory : list of ndarray, each shape (lmax+1,)
        Full-resolution theory Cl's, one per component the workspace's
        field-spin combination expects, in NaMaster's own ordering:
        [TT] for spin0 x spin0, [TE, TB] for spin0 x spin2, [EE, EB, BE,
        BB] for spin2 x spin2. The wanted spectrum (TT, TE, or EE) is
        always component 0 in each case.

    Returns
    -------
    binned_theory : ndarray, shape (n_bins,)
        The component-0 (TT / TE / EE) decoupled bandpowers.
    """
    coupled = workspace.couple_cell(cl_theory)
    decoupled = workspace.decouple_cell(coupled)[0]
    return decoupled


# ---------------------------------------------------------------------------
# 2b. Shared Minuit-running machinery, with multi-start recovery for a fit
#     that lands in a local minimum or pins a parameter at its bound
#     without ever exploring the rest of the bounded parameter space.
#     Used by both fit_parameters (joint) and fit_parameters_tt (TT-only)
#     below -- the chi^2 definitions stay separate (see module docstring),
#     only the "build a Minuit, run it, decide whether to trust it"
#     plumbing is shared, the same spirit as the shared camb_cl /
#     decouple_theory / point_source_dl_template above.
# ---------------------------------------------------------------------------

def _fmin_diagnostics(m: Minuit) -> dict:
    """
    Pull the iminuit FunctionMinimum flags that distinguish a trustworthy
    MIGRAD+HESSE result from one that only looks converged. `m.valid`
    alone misses cases like a parameter pinned at its bound with a
    suspiciously tiny HESSE error -- HESSE's parabolic-error assumption is
    invalid right at a boundary, so a "valid" fit can still report a fake
    precision there.

    Returns
    -------
    dict with keys 'valid', 'accurate_covar', 'posdef_covar',
    'hesse_failed', 'call_limit', 'above_max_edm', and 'params_at_limit'
    (the free parameter names sitting at one of their `limits`, to within
    a relative tolerance -- `m.fmin.has_parameters_at_limit` only says
    *whether*, not *which*).
    """
    fmin = m.fmin
    params_at_limit = []
    for p in m.parameters:
        if m.fixed[p]:
            continue
        lo, hi = m.limits[p]
        val = m.values[p]
        scale = (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else max(abs(val), 1.0)
        tol = 1e-6 * max(scale, 1e-12)
        if np.isfinite(lo) and abs(val - lo) < tol:
            params_at_limit.append(p)
        elif np.isfinite(hi) and abs(val - hi) < tol:
            params_at_limit.append(p)
    return {
        "valid": fmin.is_valid,
        "accurate_covar": fmin.has_accurate_covar,
        "posdef_covar": fmin.has_posdef_covar,
        "hesse_failed": fmin.hesse_failed,
        "call_limit": fmin.has_reached_call_limit,
        "above_max_edm": fmin.is_above_max_edm,
        "params_at_limit": params_at_limit,
    }


def _is_unreliable(m: Minuit) -> bool:
    """
    True if this fit shouldn't be trusted as-is: MIGRAD didn't converge,
    the covariance is inaccurate or not positive-definite (so HESSE errors
    are meaningless), HESSE itself failed, MIGRAD hit its call limit, the
    EDM is still above tolerance, or a free parameter is sitting at its
    bound (where HESSE's parabolic assumption breaks down).
    """
    d = _fmin_diagnostics(m)
    return (
        not d["valid"]
        or not d["accurate_covar"]
        or not d["posdef_covar"]
        or d["hesse_failed"]
        or d["call_limit"]
        or d["above_max_edm"]
        or bool(d["params_at_limit"])
    )


def _diverse_starts(
    initial_guess: dict,
    bounds: dict | None,
    fixed: set | None,
    n_starts: int,
    seed: int,
) -> list[dict]:
    """
    Build n_starts initial-guess dicts spanning the bounded parameter
    space, to rescue a fit that has landed in a local minimum or at a
    bound without ever exploring the rest of the space -- a small jitter
    around the original guess doesn't reliably fix that, so this
    stratifies each free, bounded parameter's *entire* range into
    n_starts bins (a manual Latin Hypercube: independently permute the
    bin order per parameter, no new dependency) and draws one point per
    bin.

    Start 0 is always `initial_guess` unchanged, so it's never wasted. A
    parameter with no entry in `bounds` (nothing to span) or that is in
    `fixed` keeps its initial_guess value in every start.
    """
    fixed = fixed or set()
    bounds = bounds or {}
    rng = np.random.default_rng(seed)

    spannable = [p for p in initial_guess if p in bounds and p not in fixed]
    perms = {p: rng.permutation(n_starts) for p in spannable}

    starts = [dict(initial_guess)]
    for k in range(1, n_starts):
        guess = dict(initial_guess)
        for p in spannable:
            lo, hi = bounds[p]
            frac = (perms[p][k] + rng.uniform()) / n_starts
            guess[p] = lo + frac * (hi - lo)
        starts.append(guess)
    return starts


def _run_once(
    cost_fn,
    guess: dict,
    bounds: dict | None,
    fixed: set | None,
    initial_step: dict | None,
    use_simplex: bool = False,
) -> Minuit:
    """
    Build one Minuit instance at `guess` and run SIMPLEX (optional) +
    MIGRAD + HESSE. Shared by fit_parameters and fit_parameters_tt -- the
    only thing that differs between the two fit paths is `cost_fn` itself.

    use_simplex : bool
        Run derivative-free SIMPLEX before MIGRAD. Only worth the extra
        cost for a diverse/far-from-optimum start (see _diverse_starts) --
        a random start point is much more likely to hand MIGRAD's
        numerical gradient a poorly conditioned first step than the
        original, physically motivated guess is.
    """
    m = Minuit(cost_fn, **guess)
    m.errordef = Minuit.LEAST_SQUARES  # chi^2 convention: errordef = 1

    if bounds:
        for name, (lo, hi) in bounds.items():
            m.limits[name] = (lo, hi)
    if initial_step:
        for name, step in initial_step.items():
            m.errors[name] = step
    if fixed:
        for name in fixed:
            m.fixed[name] = True

    if use_simplex:
        m.simplex()
    m.migrad()
    m.hesse()
    return m


def _fit_with_recovery(
    cost_fn,
    initial_guess: dict,
    bounds: dict | None,
    fixed: set | None,
    initial_step: dict | None,
    n_starts: int,
    seed: int,
    force_multistart: bool = False,
) -> dict:
    """
    Run cost_fn's MIGRAD+HESSE fit at `initial_guess`; if the result looks
    unreliable (see _is_unreliable), rescue it with a genuine multi-start
    search over the bounded parameter space rather than a cheap local
    retry -- a fit stuck at a bound or in a local minimum usually hasn't
    explored the rest of the space, so restarting from the same point (or
    a small jitter of it) doesn't reliably fix it.

    Caveat this does *not* cover: MIGRAD can converge "cleanly" -- valid,
    accurate positive-definite covariance, no parameter at a bound -- to a
    local minimum that just happens to have none of those tells (e.g. a
    narrow decoy dip well inside the bounds). No flag-based check can catch
    that; only actually searching elsewhere can. That's what
    `force_multistart=True` is for: it runs the same diverse-start search
    unconditionally, even when `initial_guess` alone already looks fine,
    at the cost of paying for every extra start every time. Off by default
    (the flag-triggered path already fixes every failure mode actually
    observed in this pipeline's output -- see fitting.py's module notes);
    turn it on for a final, paranoid production run rather than routine
    iteration.

    Returns a dict with keys 'minuit', 'status' ('ok'/'recovered'/
    'unresolved'), 'n_starts_tried', 'params_at_limit', 'chi2_spread'.
    """
    m0 = _run_once(cost_fn, initial_guess, bounds, fixed, initial_step)
    if not force_multistart and not _is_unreliable(m0):
        return {
            "minuit": m0,
            "status": "ok",
            "n_starts_tried": 1,
            "params_at_limit": [],
            "chi2_spread": 0.0,
        }

    starts = _diverse_starts(initial_guess, bounds, fixed, n_starts, seed)
    candidates = [m0]  # start 0 (== initial_guess) already run above
    for guess in starts[1:]:
        candidates.append(
            _run_once(cost_fn, guess, bounds, fixed, initial_step, use_simplex=True)
        )

    fvals = [m.fval for m in candidates]
    chi2_spread = max(fvals) - min(fvals)

    reliable = [m for m in candidates if not _is_unreliable(m)]
    if reliable:
        winner = min(reliable, key=lambda m: m.fval)
        status = "ok" if (winner is m0 and not _is_unreliable(m0)) else "recovered"
    else:
        winner = min(candidates, key=lambda m: m.fval)
        status = "unresolved"

    return {
        "minuit": winner,
        "status": status,
        "n_starts_tried": len(candidates),
        "params_at_limit": _fmin_diagnostics(winner)["params_at_limit"],
        "chi2_spread": chi2_spread,
    }


# ---------------------------------------------------------------------------
# 3. chi^2 and the fit
# ---------------------------------------------------------------------------

@dataclass
class FitData:
    """
    Bundles everything the joint TT+TE+EE chi^2 needs so Minuit only sees
    the free parameters.

    `cl_data` and `cov_inv` are the stacked [TT, TE, EE] data vector and
    its inverse covariance, with each spectrum's first bin (ell = [2, 31],
    discarded following arXiv:2504.05597 sec. 2.2 -- the sky fraction
    there is too small/uncertain) already dropped -- see
    power_spectrum.compute_joint_tt_te_ee_covariance.

    For a TT-only fit, use FitDataTT/chi_square_tt/fit_parameters_tt
    instead -- a separate, self-contained code path (no TT+TE+EE and
    TT-only logic mixed into shared functions).
    """

    cl_data: np.ndarray
    cov_inv: np.ndarray
    workspace_tt: object   # nmt.NmtWorkspace, spin0 x spin0
    workspace_te: object   # nmt.NmtWorkspace, spin0 x spin2
    workspace_ee: object   # nmt.NmtWorkspace, spin2 x spin2
    lmax: int               # shared field lmax (>= the largest of the 3 bin lmaxes)
    n_tt: int               # bandpower count to keep per spectrum (after
    n_te: int                # dropping bin 0) -- workspace_tt/te/ee all
    n_ee: int                # share one NmtBin, so decouple_cell over-produces
    tau: float = 0.0602
    mnu: float = 0.06
    r: float = 0.01


def chi_square(
    data: FitData,
    H0: float,
    ombh2: float,
    omch2: float,
    As: float,
    ns: float,
    A_ps_TT: float,
    A_ps_EE: float,
) -> float:
    """
    chi^2 between the masked-sky TT+TE+EE data and joint theory at the
    given parameters. tau, mnu, r are held fixed (see FitData); A_ps_TT
    and A_ps_EE are point-source nuisance amplitudes fit alongside the 5
    LambdaCDM parameters.
    """
    cl = camb_cl(
        H0=H0, ombh2=ombh2, omch2=omch2, As=As, ns=ns,
        tau=data.tau, mnu=data.mnu, r=data.r, lmax=data.lmax,
    )
    cl_tt = cl["TT"] + point_source_dl_template(data.lmax, A_ps_TT)
    cl_ee = cl["EE"] + point_source_dl_template(data.lmax, A_ps_EE)
    zero = np.zeros(data.lmax + 1)

    tt_binned = decouple_theory(data.workspace_tt, [cl_tt])[1:1 + data.n_tt]
    te_binned = decouple_theory(data.workspace_te, [cl["TE"], zero])[1:1 + data.n_te]
    ee_binned = decouple_theory(data.workspace_ee, [cl_ee, zero, zero, cl["BB"]])[1:1 + data.n_ee]

    theory_binned = np.concatenate([tt_binned, te_binned, ee_binned])
    residual = data.cl_data - theory_binned
    return float(residual @ data.cov_inv @ residual)


def fit_parameters(
    data: FitData,
    initial_guess: dict,
    bounds: dict | None = None,
    fixed: set | None = None,
    initial_step: dict | None = None,
    compute_minos: bool = False,
    n_starts: int = 6,
    seed: int = 0,
    force_multistart: bool = False,
) -> dict:
    """
    Run the chi^2 minimization with iminuit.

    Parameters
    ----------
    data : FitData
        Data + covariance + workspaces bundle.
    initial_guess : dict
        e.g. {"H0": 67.0, "ombh2": 0.0224, "omch2": 0.12, "As": 2.1e-9,
              "ns": 0.965, "A_ps_TT": 50.0, "A_ps_EE": 0.0}
    bounds : dict, optional
        e.g. {"H0": (50, 90), "ombh2": (0.01, 0.03),
              "omch2": (0.05, 0.3), "As": (1e-9, 4e-9), "ns": (0.9, 1.05),
              "A_ps_TT": (0, 200), "A_ps_EE": (0, 20)}
        Physically-motivated bounds keep MIGRAD from wandering into
        unphysical or numerically unstable regions of parameter space.
    fixed : set, optional
        Names (subset of initial_guess) to hold fixed at their
        initial_guess value instead of fitting, e.g. {"ns", "ombh2"}.
    initial_step : dict, optional
        Rough expected uncertainty per parameter, e.g.
        {"H0": 1.0, "ombh2": 5e-4, "omch2": 5e-3, "As": 5e-11,
         "ns": 0.01, "A_ps_TT": 10.0, "A_ps_EE": 1.0}. Sets iminuit's
         initial step size for its numerical gradient/Hessian estimate
         (`m.errors[...]`). Without this, wildly different parameter
         scales (H0 ~ 70 vs As ~ 1e-9) can leave MIGRAD's first gradient
         estimate poorly conditioned, occasionally landing in a
         negative-curvature region that triggers an expensive recovery
         search (a `NegativeG2LineSearch`) -- minutes instead of seconds
         to converge, even though the final answer is the same either
         way. Worth setting whenever a fit is unexpectedly slow.
    compute_minos : bool
        Also run MINOS (asymmetric profile-likelihood errors) for every
        free parameter, on the final (possibly rescued) best fit. Off by
        default: MINOS runs a separate bisection search *per parameter* on
        top of MIGRAD+HESSE, each step another full chi_square (CAMB)
        evaluation -- for a chi_square this expensive (a full Boltzmann
        solve per call), this easily becomes the dominant cost of the
        whole fit, sometimes by an order of magnitude, for marginal
        benefit if all you're reading is the symmetric `errors` dict. Turn
        on only if you specifically want `result["minos"]`'s asymmetric
        error bars.
    n_starts : int
        Number of starting points to try if the fit at `initial_guess`
        looks unreliable (not valid, an inaccurate/non-positive-definite
        covariance, HESSE failure, MIGRAD hitting its call limit, EDM
        above tolerance, or a free parameter pinned at its bound -- see
        `_is_unreliable`). The rescue starts are spread across the *whole*
        bounded parameter space (a manual Latin Hypercube), not jittered
        near `initial_guess` -- a fit stuck at a bound or in a local
        minimum usually hasn't explored the rest of the space, so a local
        retry doesn't reliably help. Only paid when needed: a well-behaved
        fit costs exactly one MIGRAD+HESSE pass, same as before. Each
        rescue candidate re-pays a full SIMPLEX+MIGRAD+HESSE pass -- with
        CAMB's ~1-1.5s per chi^2 call, a rescued patch can take several
        minutes longer; tune this down while iterating, up for a final
        run.
    seed : int
        RNG seed for the rescue starting points, for reproducibility.
    force_multistart : bool
        Always run the full diverse-start search, even when the fit at
        `initial_guess` already looks reliable by every Minuit diagnostic
        (valid, accurate positive-definite covariance, nothing at a
        bound). Off by default: those diagnostics catch every failure mode
        actually seen in this pipeline's output (a non-converged fit, or a
        parameter pinned at its bound with a fake tiny error). What they
        *can't* catch is MIGRAD converging "cleanly" to a local minimum
        that happens to trip none of those flags -- no flag-based check
        can, only actually searching elsewhere can. Set this to True for a
        final, paranoid production run where you want that extra
        assurance and are willing to pay full multi-start cost on every
        fit, not just routine iteration.

    Returns
    -------
    result : dict with keys
        'best_fit'  : dict of best-fit parameter values
        'errors'    : dict of hesse (parabolic) errors
        'minos'     : dict of asymmetric minos errors (None unless
                      compute_minos=True, or if minos was unreliable)
        'chi2'      : chi^2 at the best fit
        'valid'     : whether MIGRAD converged
        'minuit'    : the raw Minuit object, for further inspection/plots
        'status'    : 'ok' (converged first try), 'recovered' (a
                      multi-start rescue found a reliable minimum), or
                      'unresolved' (no candidate, including the rescue
                      starts, looked reliable -- treat best_fit/errors as
                      approximate, and check params_at_limit)
        'n_starts_tried'  : how many starting points were actually run
        'params_at_limit' : free parameters sitting at a bound in the
                      returned best_fit -- their HESSE errors are not
                      meaningful there (HESSE's parabolic assumption fails
                      at a boundary)
        'chi2_spread'     : max - min chi^2 across every starting point
                      tried (0.0 if only one was run); small means every
                      explored region agrees, large means the returned
                      minimum is still worth a manual look even if
                      status == 'recovered'
    """
    def neg_log_like(H0, ombh2, omch2, As, ns, A_ps_TT, A_ps_EE):
        return chi_square(data, H0, ombh2, omch2, As, ns, A_ps_TT, A_ps_EE)

    outcome = _fit_with_recovery(
        neg_log_like, initial_guess, bounds, fixed, initial_step, n_starts, seed,
        force_multistart=force_multistart,
    )
    m = outcome["minuit"]

    minos_errors = None
    if compute_minos:
        free_params = [p for p in initial_guess if not (fixed and p in fixed)]
        try:
            m.minos(*free_params)
            minos_errors = {
                p: (m.merrors[p].lower, m.merrors[p].upper) for p in free_params
            }
        except Exception:
            # minos can fail near parameter boundaries or for a poorly
            # conditioned Hessian -- fall back to hesse errors only, but
            # don't hide that this happened.
            pass

    return {
        "best_fit": {p: m.values[p] for p in initial_guess},
        "errors": {p: m.errors[p] for p in initial_guess},
        "minos": minos_errors,
        "chi2": m.fval,
        "valid": m.valid,
        "minuit": m,
        "status": outcome["status"],
        "n_starts_tried": outcome["n_starts_tried"],
        "params_at_limit": outcome["params_at_limit"],
        "chi2_spread": outcome["chi2_spread"],
    }


# ---------------------------------------------------------------------------
# 4. TT-only chi^2 and fit -- a separate, self-contained path (not the
#    joint TT+TE+EE functions above with TE/EE left empty). No spin0 x
#    spin2 / spin2 x spin2 workspace, no TE/EE terms, no A_ps_EE.
# ---------------------------------------------------------------------------

@dataclass
class FitDataTT:
    """
    Bundles everything the TT-only chi^2 needs so Minuit only sees the
    free parameters.

    `cl_data` and `cov_inv` are the TT data vector and its inverse
    covariance, with the first bin (ell = [2, 31]) already dropped --
    typically built with power_spectrum.compute_gaussian_covariance
    (the single-spectrum, full-bin-to-bin-correlation covariance; see
    REPORT.md for why this -- not the joint function's same-bin-only
    simplification -- is the right choice for a standalone TT fit).
    """

    cl_data: np.ndarray
    cov_inv: np.ndarray
    workspace_tt: object   # nmt.NmtWorkspace, spin0 x spin0
    lmax: int
    n_tt: int               # bandpower count to keep (after dropping bin 0)
    tau: float = 0.0602
    mnu: float = 0.06
    r: float = 0.01


def chi_square_tt(
    data: FitDataTT,
    H0: float,
    ombh2: float,
    omch2: float,
    As: float,
    ns: float,
    A_ps_TT: float,
) -> float:
    """
    chi^2 between the masked-sky TT data and TT-only theory at the given
    parameters. tau, mnu, r are held fixed (see FitDataTT); A_ps_TT is
    the point-source nuisance amplitude fit alongside the 5 LambdaCDM
    parameters. No A_ps_EE -- there's no EE term here for it to enter.
    """
    cl_tt = camb_cl(
        H0=H0, ombh2=ombh2, omch2=omch2, As=As, ns=ns,
        tau=data.tau, mnu=data.mnu, r=data.r, lmax=data.lmax,
    )["TT"]
    cl_tt = cl_tt + point_source_dl_template(data.lmax, A_ps_TT)

    theory_binned = decouple_theory(data.workspace_tt, [cl_tt])[1:1 + data.n_tt]
    residual = data.cl_data - theory_binned
    return float(residual @ data.cov_inv @ residual)


def fit_parameters_tt(
    data: FitDataTT,
    initial_guess: dict,
    bounds: dict | None = None,
    fixed: set | None = None,
    initial_step: dict | None = None,
    compute_minos: bool = False,
    n_starts: int = 6,
    seed: int = 0,
    force_multistart: bool = False,
) -> dict:
    """
    Run the TT-only chi^2 minimization with iminuit. Same machinery and
    caveats as fit_parameters (see its docstring for `initial_step`,
    `compute_minos`, `n_starts`, `seed`, and the extra 'status'/
    'params_at_limit'/'chi2_spread' return keys), specialized to
    chi_square_tt's 6 parameters (no A_ps_EE). TT alone has a weaker, more
    degenerate likelihood than the joint fit (no TE/EE to break the usual
    H0-omch2-ns degeneracies), so it's noticeably more prone to needing the
    multi-start rescue.

    Parameters
    ----------
    data : FitDataTT
    initial_guess : dict
        e.g. {"H0": 67.0, "ombh2": 0.0224, "omch2": 0.12, "As": 2.1e-9,
              "ns": 0.965, "A_ps_TT": 50.0}
    bounds, fixed, initial_step, compute_minos, n_starts, seed,
        force_multistart : see fit_parameters

    Returns
    -------
    result : dict, same shape as fit_parameters' return value.
    """
    def neg_log_like(H0, ombh2, omch2, As, ns, A_ps_TT):
        return chi_square_tt(data, H0, ombh2, omch2, As, ns, A_ps_TT)

    outcome = _fit_with_recovery(
        neg_log_like, initial_guess, bounds, fixed, initial_step, n_starts, seed,
        force_multistart=force_multistart,
    )
    m = outcome["minuit"]

    minos_errors = None
    if compute_minos:
        free_params = [p for p in initial_guess if not (fixed and p in fixed)]
        try:
            m.minos(*free_params)
            minos_errors = {
                p: (m.merrors[p].lower, m.merrors[p].upper) for p in free_params
            }
        except Exception:
            pass

    return {
        "best_fit": {p: m.values[p] for p in initial_guess},
        "errors": {p: m.errors[p] for p in initial_guess},
        "minos": minos_errors,
        "chi2": m.fval,
        "valid": m.valid,
        "minuit": m,
        "status": outcome["status"],
        "n_starts_tried": outcome["n_starts_tried"],
        "params_at_limit": outcome["params_at_limit"],
        "chi2_spread": outcome["chi2_spread"],
    }
