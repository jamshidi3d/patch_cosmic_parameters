"""
fitting.py
----------
Fit cosmological parameters to a masked-sky TT+TE+EE power spectrum:

  params --[CAMB]--> theory Cl --[NaMaster mode coupling]--> binned theory
  chi^2 = (data - binned theory)^T Cov^-1 (data - binned theory)
  params_best = argmin chi^2   (iminuit / MIGRAD, or Levenberg-Marquardt)

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

Both fit paths build the same FitProblem (section 2b) and run one
minimizer pass on it -- `method="iminuit"` (default) or
`method="levenberg_marquardt"` (levenberg_marquardt.py). There are no
restarts: see docs/FIT_CONVERGENCE_FIXES.md for the causes of the earlier
non-convergence / fake-tiny-error fits and how each is removed.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import camb
from iminuit import Minuit

from . import patch as _patch
from . import power_spectrum as _power_spectrum


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
    cached = _camb_cl_cached(
        float(H0), float(ombh2), float(omch2), float(As), float(ns),
        int(lmax), float(tau), float(mnu), float(r),
    )
    return {name: cl.copy() for name, cl in cached.items()}


@functools.lru_cache(maxsize=64)
def _camb_cl_cached(H0, ombh2, omch2, As, ns, lmax, tau, mnu, r) -> dict:
    """
    The actual Boltzmann solve behind camb_cl, memoized on the exact
    parameter values: a fit evaluating the same cosmology again -- e.g. a
    Jacobian step in a point-source amplitude only, or Minuit's gradient
    call right after its function call at the same point -- reuses the
    spectra instead of re-running CAMB.
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
    return {name: total[:, i].copy() for name, i in idx.items()}


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


def bandpower_windows(workspace, n_keep: int) -> np.ndarray:
    """
    The workspace's bandpower window functions for its component 0 (TT, TE
    or EE), for bins 1 .. n_keep (bin 0 dropped, as everywhere here).

    decouple_theory(workspace, cl_theory)[1:1+n_keep] equals
    np.einsum("bkl,kl->b", windows, cl_theory) to machine precision
    (checked: max relative difference ~1e-16 for TT, TE and EE, with
    beams and different T/P masks). The windows are a few MB, whereas the
    workspace holds the full mode-coupling matrix (GBs at lmax ~ 2000), so
    a FitData built from windows can be kept, saved to disk and reloaded
    to refit without recomputing any spectra.

    Returns
    -------
    windows : ndarray, shape (n_keep, ncls, lmax+1)
        ncls = 1 (TT), 2 (TE: [TE, TB]) or 4 (EE: [EE, EB, BE, BB]).
    """
    return np.ascontiguousarray(workspace.get_bandpower_windows()[0, 1:1 + n_keep])


# ---------------------------------------------------------------------------
# 2b. The fit problem shared by both minimizers: whitened residuals, priors,
#     a soft validity wall, and a finite-difference Jacobian with fixed
#     physical steps. See docs/FIT_CONVERGENCE_FIXES.md for why each piece is
#     here -- in short: Minuit's hard `limits` (sin transform, zero slope at
#     the bound) collapsed the covariance of any parameter that touched a
#     bound, the point-source amplitudes are flat directions on a patch,
#     and MIGRAD's own tiny-step numerical derivatives of a CAMB chi^2 were
#     poorly conditioned (parameter scales 1e-9 .. 70).
# ---------------------------------------------------------------------------

# Finite-difference step / scale per parameter, roughly 0.2-1x the expected
# 1-sigma of a full-sky fit (smaller than any patch's sigma, well above
# CAMB's numerical jitter). Doubles as the O(1) rescaling of Minuit's
# coordinates. Override per fit with `initial_step=`.
DEFAULT_STEP = {
    "H0": 0.5,
    "ombh2": 1e-4,
    "omch2": 1e-3,
    "As": 1e-11,
    "ns": 5e-3,
    "A_ps_TT": 5.0,
    "A_ps_EE": 0.5,
}

# chi^2 cost of one step-scale outside a validity bound. Steep enough that
# a fit never settles outside, smooth (continuous first derivative) so the
# minimizer sees no kink -- unlike Minuit's own `limits`.
WALL_STIFFNESS = 1e4

# With run_hesse=True, HESSE and Gauss-Newton errors disagreeing by more
# than this factor flags the iminuit fit.
HESSE_FISHER_TOLERANCE = 1.3


class FitProblem:
    """
    A chi^2 fit as a least-squares problem, shared by the "iminuit" and
    "levenberg_marquardt" methods so both minimize the exact same function.

    The residual vector is [whitened data rows, prior rows, wall rows]:

    - data rows   : L^T (data - theory), with cov_inv = L L^T, so their
                    squared sum is the usual (d-t)^T Cov^-1 (d-t);
    - prior rows  : (p - mean) / sigma for every Gaussian prior on a free
                    parameter (e.g. the point-source amplitudes on a patch);
    - wall rows   : sqrt(WALL_STIFFNESS) * (p - clip(p)) / step -- zero
                    inside `bounds`, a smooth quadratic penalty outside.
                    The theory is always evaluated at the clipped point, so
                    CAMB never sees an out-of-range parameter.

    `bounds` here are *validity* limits (where CAMB is trustworthy), not
    physical priors: a fit landing on one is flagged, never silently
    pinned with a fake error bar.
    """

    def __init__(
        self,
        theory_fn,
        cl_data: np.ndarray,
        cov_inv: np.ndarray,
        initial_guess: dict,
        bounds: dict | None = None,
        fixed: set | None = None,
        priors: dict | None = None,
        initial_step: dict | None = None,
    ):
        fixed = set(fixed or ())
        bounds = bounds or {}
        priors = priors or {}
        step = {**DEFAULT_STEP, **(initial_step or {})}

        unknown = (fixed | set(priors)) - set(initial_guess)
        if unknown:
            raise ValueError(f"fixed/priors name parameters not in initial_guess: {sorted(unknown)}")

        self.theory_fn = theory_fn
        self.cl_data = np.asarray(cl_data, dtype=float)
        self.names = list(initial_guess)
        self.free = [p for p in self.names if p not in fixed]
        self.fixed_values = {p: float(initial_guess[p]) for p in self.names if p in fixed}
        missing = [p for p in self.free if p not in step]
        if missing:
            raise ValueError(f"no step scale for {missing} -- pass initial_step=")

        self.p0 = np.array([float(initial_guess[p]) for p in self.free])
        self.scale = np.array([float(step[p]) for p in self.free])
        self.lo = np.array([bounds.get(p, (-np.inf, np.inf))[0] for p in self.free], dtype=float)
        self.hi = np.array([bounds.get(p, (-np.inf, np.inf))[1] for p in self.free], dtype=float)

        prior_free = [p for p in priors if p in self.free]
        self.prior_names = prior_free
        self.prior_idx = np.array([self.free.index(p) for p in prior_free], dtype=int)
        self.prior_mean = np.array([priors[p][0] for p in prior_free], dtype=float)
        self.prior_sigma = np.array([priors[p][1] for p in prior_free], dtype=float)

        self.chol = np.linalg.cholesky(cov_inv)  # cov_inv = L L^T
        self.n_data = self.cl_data.size
        self.n_prior = len(prior_free)
        self.n_free = len(self.free)

    # -- parameter vectors <-> dicts ---------------------------------------

    def params(self, p: np.ndarray) -> dict:
        """Full parameter dict (free + fixed), in initial_guess order."""
        free = dict(zip(self.free, (float(v) for v in p)))
        return {n: free[n] if n in free else self.fixed_values[n] for n in self.names}

    def to_physical(self, x: np.ndarray) -> np.ndarray:
        """Minuit's O(1) coordinates x -> physical parameters p."""
        return self.p0 + self.scale * np.asarray(x)

    def clip(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, self.lo, self.hi)

    # -- residuals and chi^2 -----------------------------------------------

    def data_rows(self, pc: np.ndarray) -> np.ndarray:
        theory = self.theory_fn(self.params(pc))
        return self.chol.T @ (self.cl_data - theory)

    def residual(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        pc = self.clip(p)
        prior = (p[self.prior_idx] - self.prior_mean) / self.prior_sigma
        wall = np.sqrt(WALL_STIFFNESS) * (p - pc) / self.scale
        return np.concatenate([self.data_rows(pc), prior, wall])

    def chi2_parts(self, p: np.ndarray) -> tuple[float, float, float]:
        """(data chi^2, prior chi^2, wall penalty) at p."""
        r = self.residual(p)
        nd, npr = self.n_data, self.n_prior
        return (
            float(r[:nd] @ r[:nd]),
            float(r[nd:nd + npr] @ r[nd:nd + npr]),
            float(r[nd + npr:] @ r[nd + npr:]),
        )

    def chi2(self, p: np.ndarray) -> float:
        """Total cost. A theory failure (e.g. CAMB error) is a large finite
        value rather than an exception, so a minimizer backs off from it."""
        try:
            r = self.residual(p)
        except Exception:
            return 1e10
        return float(r @ r)

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """
        d(residual)/dp, shape (n_data + n_prior + n_free, n_free).

        Data rows: central differences with the fixed physical steps
        `self.scale` (one-sided where a validity bound is closer than one
        step). Large enough steps to sit far above CAMB's numerical noise;
        a step in a point-source amplitude reuses the memoized CAMB
        spectrum, so it costs no Boltzmann solve. Prior and wall rows are
        analytic.
        """
        p = np.asarray(p, dtype=float)
        pc = self.clip(p)
        jac = np.zeros((self.n_data + self.n_prior + self.n_free, self.n_free))
        for i in range(self.n_free):
            up, dn = pc.copy(), pc.copy()
            up[i] = min(pc[i] + self.scale[i], self.hi[i])
            dn[i] = max(pc[i] - self.scale[i], self.lo[i])
            jac[:self.n_data, i] = (self.data_rows(up) - self.data_rows(dn)) / (up[i] - dn[i])
        for k, i in enumerate(self.prior_idx):
            jac[self.n_data + k, i] = 1.0 / self.prior_sigma[k]
        outside = (p < self.lo) | (p > self.hi)
        for i in np.flatnonzero(outside):
            jac[self.n_data + self.n_prior + i, i] = np.sqrt(WALL_STIFFNESS) / self.scale[i]
        return jac

    def params_at_bound(self, p: np.ndarray) -> list[str]:
        """Free parameters outside, or within 0.1 step of, a validity bound."""
        p = np.asarray(p, dtype=float)
        near = (p - self.lo < 0.1 * self.scale) | (self.hi - p < 0.1 * self.scale)
        return [name for name, flag in zip(self.free, near) if flag]


def camb_solve_count() -> int:
    """Boltzmann solves done so far in this process (memo misses)."""
    return _camb_cl_cached.cache_info().misses


def _gauss_newton_covariance(jac: np.ndarray) -> np.ndarray:
    """(J^T J)^-1 of the whitened residual Jacobian: the Fisher /
    Gauss-Newton parameter covariance (priors included via their rows)."""
    return np.linalg.inv(jac.T @ jac)


def _fit_iminuit(problem: FitProblem, compute_minos: bool, run_hesse: bool) -> dict:
    """
    One MIGRAD pass, in the O(1) coordinates x = (p - p0) / step; errors
    from the Gauss-Newton curvature at the MIGRAD minimum (see _run_fit).

    - No `m.limits`: validity bounds are the problem's smooth wall, so no
      parameter can be pinned by Minuit's sin transform (zero slope at the
      limit -> collapsed external error).
    - `grad=` is supplied from the problem's fixed-step Jacobian
      (g = 2 J^T r), so MIGRAD never takes its own tiny-step numerical
      derivatives of the CAMB chi^2, which has step-like jumps of ~1e-4
      along H0 / omch2.
    - strategy 0: no extra numerical second-derivative passes inside
      MIGRAD -- those are exactly what that noise corrupts. For the same
      reason HESSE is *not* used for the errors: on a patch-sized chi^2 it
      fails (forced-posdef fallback, errors ~1e-7 -- the old "insane low
      error bars") or underestimates them by up to ~2x. It can still be run
      as a diagnostic with run_hesse=True.
    """
    n = problem.n_free

    def fcn(x):
        return problem.chi2(problem.to_physical(x))

    def grad(x):
        p = problem.to_physical(x)
        return 2.0 * (problem.jacobian(p).T @ problem.residual(p)) * problem.scale

    m = Minuit(fcn, np.zeros(n), grad=grad, name=tuple(problem.free))
    m.errordef = Minuit.LEAST_SQUARES
    m.errors = np.ones(n)
    m.strategy = 0
    m.migrad()

    p = problem.to_physical(np.array(m.values))
    flags = []
    fmin = m.fmin
    if not fmin.is_valid:
        flags.append("migrad_invalid")
    if fmin.has_reached_call_limit:
        flags.append("call_limit")
    if fmin.is_above_max_edm:
        flags.append("above_max_edm")
    n_calls = int(fmin.nfcn)

    cov_hesse = None
    if run_hesse:
        m.hesse()
        if m.fmin.hesse_failed or not m.fmin.has_accurate_covar:
            flags.append("hesse_failed")
        else:
            cov_hesse = np.array(m.covariance) * np.outer(problem.scale, problem.scale)

    minos = None
    if compute_minos:
        try:
            m.minos()
            minos = {
                name: (m.merrors[name].lower * s, m.merrors[name].upper * s)
                for name, s in zip(problem.free, problem.scale)
            }
        except Exception:
            flags.append("minos_failed")

    return {"p": p, "cov": None, "cov_hesse": cov_hesse, "flags": flags, "minuit": m,
            "minos": minos, "n_iter": n_calls, "valid": bool(fmin.is_valid)}


def _run_fit(
    theory_fn,
    cl_data: np.ndarray,
    cov_inv: np.ndarray,
    initial_guess: dict,
    bounds: dict | None,
    fixed: set | None,
    priors: dict | None,
    initial_step: dict | None,
    method: str,
    compute_minos: bool,
    run_hesse: bool = False,
) -> dict:
    """
    Build the FitProblem, run the chosen minimizer exactly once, and package
    the result with the same keys for both methods (see fit_parameters).
    """
    problem = FitProblem(
        theory_fn, cl_data, cov_inv, initial_guess,
        bounds=bounds, fixed=fixed, priors=priors, initial_step=initial_step,
    )
    solves_before = camb_solve_count()

    if method == "iminuit":
        out = _fit_iminuit(problem, compute_minos, run_hesse)
    elif method == "levenberg_marquardt":
        from .levenberg_marquardt import fit_levenberg_marquardt
        out = fit_levenberg_marquardt(problem)
        out.update(minuit=None, minos=None, cov_hesse=None)
    else:
        raise ValueError(f"unknown method {method!r} (use 'iminuit' or 'levenberg_marquardt')")

    p = out["p"]
    flags = list(out["flags"])

    # Gauss-Newton covariance at the best fit, (J^T J)^-1 of the whitened
    # residuals (priors included): the parameter errors for both methods.
    # For a Gaussian chi^2 this is the curvature 1/2 d2chi2/dp2 without the
    # residual-times-second-derivative term -- exact for a model linear
    # around the minimum, and computed from fixed physical steps that are
    # insensitive to CAMB's numerical noise (unchanged to 3 digits for
    # steps 0.5x..2x DEFAULT_STEP).
    try:
        if method == "levenberg_marquardt" and out["cov"] is not None:
            cov_fisher = out["cov"]  # already the Gauss-Newton covariance at p
        else:
            cov_fisher = _gauss_newton_covariance(problem.jacobian(p))
    except np.linalg.LinAlgError:
        cov_fisher = np.full((problem.n_free, problem.n_free), np.nan)
        flags.append("fisher_singular")
    cov = out["cov"] if out["cov"] is not None else cov_fisher

    err_free = np.sqrt(np.clip(np.diag(cov), 0, None))
    err_fisher_free = np.sqrt(np.clip(np.diag(cov_fisher), 0, None))
    errors_hesse = None
    if out["cov_hesse"] is not None:
        err_hesse_free = np.sqrt(np.clip(np.diag(out["cov_hesse"]), 0, None))
        errors_hesse = {n: 0.0 for n in problem.names}
        errors_hesse.update(zip(problem.free, (float(e) for e in err_hesse_free)))
        ratio = err_hesse_free / err_fisher_free
        off = [n for n, r in zip(problem.free, ratio)
               if not (1 / HESSE_FISHER_TOLERANCE <= r <= HESSE_FISHER_TOLERANCE)]
        if off:
            flags.append(f"hesse_fisher_mismatch:{','.join(off)}")

    at_bound = problem.params_at_bound(p)
    if at_bound:
        flags.append(f"at_bound:{','.join(at_bound)}")

    chi2_data, chi2_prior, chi2_wall = problem.chi2_parts(p)
    best_fit = problem.params(p)
    errors = {n: 0.0 for n in problem.names}  # fixed parameters: exactly 0
    errors.update(zip(problem.free, (float(e) for e in err_free)))
    errors_fisher = {n: 0.0 for n in problem.names}
    errors_fisher.update(zip(problem.free, (float(e) for e in err_fisher_free)))

    return {
        "best_fit": best_fit,
        "errors": errors,
        "errors_fisher": errors_fisher,
        "errors_hesse": errors_hesse,
        "covariance": cov,
        "free_params": list(problem.free),
        "minos": out["minos"],
        "chi2": chi2_data,
        "chi2_prior": chi2_prior,
        "chi2_wall": chi2_wall,
        "n_data": problem.n_data,
        "valid": out["valid"] and not flags,
        "status": "ok" if not flags else "flagged",
        "flags": flags,
        "params_at_bound": at_bound,
        "method": method,
        "n_iter": out["n_iter"],
        "n_camb_solves": camb_solve_count() - solves_before,
        "minuit": out["minuit"],
    }


_FIT_DOC = """
    Parameters
    ----------
    data
        Data + inverse covariance + workspaces bundle.
    initial_guess : dict
        Starting point, and the value every `fixed` parameter is held at.
        For a patch, the full-sky best fit is the natural choice.
    bounds : dict, optional
        *Validity* limits, e.g. {"H0": (40, 100), "ns": (0.7, 1.3)} --
        where CAMB is trustworthy, not a physical prior. Implemented as a
        smooth quadratic wall (never as Minuit `limits`, whose sin
        transform pins a parameter with a fake tiny error); a fit ending
        on one is flagged ('at_bound:...'). Keep them wide.
    fixed : set, optional
        Names to hold at their initial_guess value.
    priors : dict, optional
        Gaussian priors {name: (mean, sigma)} added to chi^2 -- e.g. the
        point-source amplitudes on a patch, which the patch's own
        bandpowers barely constrain (A_ps * (ell/3000)^2 is <= A/9 at
        ell <= 1000), set from the full-sky fit. Ignored for fixed names.
    initial_step : dict, optional
        Physical step per parameter, overriding DEFAULT_STEP: the
        finite-difference step of the Jacobian and Minuit's O(1) scaling.
        ~0.2-1 x the expected full-sky sigma.
    method : "iminuit" (default) or "levenberg_marquardt"
        One MIGRAD pass, or Levenberg-Marquardt on the whitened residuals.
        Both minimize the identical chi^2 (FitProblem), with no restarts,
        and both report the Gauss-Newton (Fisher) errors at their minimum.
    compute_minos : bool
        iminuit only: also run MINOS (expensive -- one profile scan per
        free parameter, each step a CAMB solve).
    run_hesse : bool
        iminuit only, diagnostic: also run HESSE and report its errors as
        'errors_hesse' (flagged 'hesse_failed' / 'hesse_fisher_mismatch'
        when they are unusable). Off by default: CAMB's chi^2 is too noisy
        at HESSE's step sizes for its errors to be trusted on a patch.

    Returns
    -------
    dict with keys
        'best_fit'      : all parameter values (fixed ones included)
        'errors'        : 1-sigma errors, Gauss-Newton (Fisher) curvature at
                          the minimum (0.0 for fixed parameters)
        'errors_fisher' : same as 'errors' (kept as an explicit name)
        'errors_hesse'  : HESSE errors if run_hesse=True, else None
        'covariance'    : free-parameter covariance, order 'free_params'
        'free_params'   : names of the free parameters
        'minos'         : asymmetric errors, or None
        'chi2'          : data chi^2 at the best fit ('chi2_prior' and
                          'chi2_wall' are the prior and wall terms)
        'n_data'        : number of bandpowers
        'status'        : 'ok', or 'flagged' with the reasons in 'flags'
                          (invalid MIGRAD / EDM above tolerance, a validity
                          bound reached, LM not converged, singular Fisher
                          matrix, and with run_hesse: HESSE failed or
                          disagreeing with Fisher by > 30%)
        'params_at_bound', 'method', 'n_iter' (Minuit nfcn / LM
                          iterations), 'n_camb_solves', 'valid'
        'minuit'        : the Minuit object (iminuit only). Its values and
                          errors are in the O(1) coordinates
                          (p - initial_guess) / step, not physical units.
"""


# ---------------------------------------------------------------------------
# 3. Joint TT+TE+EE chi^2 and fit
# ---------------------------------------------------------------------------

@dataclass
class FitData:
    """
    Bundles everything the joint TT+TE+EE chi^2 needs so the minimizer only
    sees the free parameters.

    `cl_data` and `cov_inv` are the stacked [TT, TE, EE] data vector and
    its inverse covariance, with each spectrum's first bin (ell = [2, 31],
    discarded following arXiv:2504.05597 sec. 2.2 -- the sky fraction
    there is too small/uncertain) already dropped -- see
    power_spectrum.compute_joint_tt_te_ee_covariance.

    For a TT-only fit, use FitDataTT/chi_square_tt/fit_parameters_tt
    instead -- a separate, self-contained code path (no TT+TE+EE and
    TT-only logic mixed into shared functions).

    The theory is binned with the bandpower windows of the three
    workspaces (`bandpower_windows`). Pass the workspaces and the windows
    are extracted on construction; or pass `windows_tt/te/ee` directly
    (e.g. from `load_likelihood`) with the workspaces left as None.
    """

    cl_data: np.ndarray
    cov_inv: np.ndarray
    workspace_tt: object   # nmt.NmtWorkspace, spin0 x spin0 (or None if windows given)
    workspace_te: object   # nmt.NmtWorkspace, spin0 x spin2 (or None)
    workspace_ee: object   # nmt.NmtWorkspace, spin2 x spin2 (or None)
    lmax: int               # shared field lmax (>= the largest of the 3 bin lmaxes)
    n_tt: int               # bandpower count to keep per spectrum (after
    n_te: int                # dropping bin 0) -- workspace_tt/te/ee all
    n_ee: int                # share one NmtBin, so decouple_cell over-produces
    tau: float = 0.0602
    mnu: float = 0.06
    r: float = 0.01
    windows_tt: np.ndarray | None = None   # (n_tt, 1, lmax+1)
    windows_te: np.ndarray | None = None   # (n_te, 2, lmax+1)
    windows_ee: np.ndarray | None = None   # (n_ee, 4, lmax+1)

    def __post_init__(self):
        if self.windows_tt is None:
            self.windows_tt = bandpower_windows(self.workspace_tt, self.n_tt)
        if self.windows_te is None:
            self.windows_te = bandpower_windows(self.workspace_te, self.n_te)
        if self.windows_ee is None:
            self.windows_ee = bandpower_windows(self.workspace_ee, self.n_ee)


def theory_vector(data: FitData, params: dict) -> np.ndarray:
    """
    Stacked [TT, TE, EE] binned theory at `params` (H0, ombh2, omch2, As,
    ns, A_ps_TT, A_ps_EE), binned with the data's own bandpower windows
    (identical to pushing it through the workspaces, see bandpower_windows).
    """
    cl = camb_cl(
        H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"],
        As=params["As"], ns=params["ns"],
        tau=data.tau, mnu=data.mnu, r=data.r, lmax=data.lmax,
    )
    cl_tt = cl["TT"] + point_source_dl_template(data.lmax, params["A_ps_TT"])
    cl_ee = cl["EE"] + point_source_dl_template(data.lmax, params["A_ps_EE"])

    # Only non-zero inputs contribute: TB, EB, BE are zero in LambdaCDM.
    tt_binned = data.windows_tt[:, 0] @ cl_tt
    te_binned = data.windows_te[:, 0] @ cl["TE"]
    ee_binned = data.windows_ee[:, 0] @ cl_ee + data.windows_ee[:, 3] @ cl["BB"]
    return np.concatenate([tt_binned, te_binned, ee_binned])


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
    residual = data.cl_data - theory_vector(data, dict(
        H0=H0, ombh2=ombh2, omch2=omch2, As=As, ns=ns, A_ps_TT=A_ps_TT, A_ps_EE=A_ps_EE,
    ))
    return float(residual @ data.cov_inv @ residual)


def fit_parameters(
    data: FitData,
    initial_guess: dict,
    bounds: dict | None = None,
    fixed: set | None = None,
    priors: dict | None = None,
    initial_step: dict | None = None,
    method: str = "iminuit",
    compute_minos: bool = False,
    run_hesse: bool = False,
) -> dict:
    return _run_fit(
        lambda p: theory_vector(data, p), data.cl_data, data.cov_inv,
        initial_guess, bounds, fixed, priors, initial_step, method, compute_minos,
        run_hesse,
    )


fit_parameters.__doc__ = """
    Joint TT+TE+EE fit of (H0, ombh2, omch2, As, ns, A_ps_TT, A_ps_EE), in
    a single minimizer pass.
""" + _FIT_DOC


# ---------------------------------------------------------------------------
# 4. TT-only chi^2 and fit -- a separate, self-contained path (not the
#    joint TT+TE+EE functions above with TE/EE left empty). No spin0 x
#    spin2 / spin2 x spin2 workspace, no TE/EE terms, no A_ps_EE.
# ---------------------------------------------------------------------------

@dataclass
class FitDataTT:
    """
    Bundles everything the TT-only chi^2 needs so the minimizer only sees
    the free parameters.

    `cl_data` and `cov_inv` are the TT data vector and its inverse
    covariance, with the first bin (ell = [2, 31]) already dropped --
    typically built with power_spectrum.compute_gaussian_covariance
    (the single-spectrum, full-bin-to-bin-correlation covariance; see
    docs/REPORT.md for why this -- not the joint function's same-bin-only
    simplification -- is the right choice for a standalone TT fit).
    """

    cl_data: np.ndarray
    cov_inv: np.ndarray
    workspace_tt: object   # nmt.NmtWorkspace, spin0 x spin0 (or None if windows given)
    lmax: int
    n_tt: int               # bandpower count to keep (after dropping bin 0)
    tau: float = 0.0602
    mnu: float = 0.06
    r: float = 0.01
    windows_tt: np.ndarray | None = None   # (n_tt, 1, lmax+1), see FitData

    def __post_init__(self):
        if self.windows_tt is None:
            self.windows_tt = bandpower_windows(self.workspace_tt, self.n_tt)


def theory_vector_tt(data: FitDataTT, params: dict) -> np.ndarray:
    """Binned TT theory at `params` (H0, ombh2, omch2, As, ns, A_ps_TT)."""
    cl_tt = camb_cl(
        H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"],
        As=params["As"], ns=params["ns"],
        tau=data.tau, mnu=data.mnu, r=data.r, lmax=data.lmax,
    )["TT"]
    cl_tt = cl_tt + point_source_dl_template(data.lmax, params["A_ps_TT"])
    return data.windows_tt[:, 0] @ cl_tt


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
    residual = data.cl_data - theory_vector_tt(data, dict(
        H0=H0, ombh2=ombh2, omch2=omch2, As=As, ns=ns, A_ps_TT=A_ps_TT,
    ))
    return float(residual @ data.cov_inv @ residual)


def fit_parameters_tt(
    data: FitDataTT,
    initial_guess: dict,
    bounds: dict | None = None,
    fixed: set | None = None,
    priors: dict | None = None,
    initial_step: dict | None = None,
    method: str = "iminuit",
    compute_minos: bool = False,
    run_hesse: bool = False,
) -> dict:
    return _run_fit(
        lambda p: theory_vector_tt(data, p), data.cl_data, data.cov_inv,
        initial_guess, bounds, fixed, priors, initial_step, method, compute_minos,
        run_hesse,
    )


fit_parameters_tt.__doc__ = """
    TT-only fit of (H0, ombh2, omch2, As, ns, A_ps_TT), in a single
    minimizer pass. Same machinery as fit_parameters (no A_ps_EE).
""" + _FIT_DOC


# ---------------------------------------------------------------------------
# 5. One-patch end-to-end pipeline: maps + masks -> spectra -> covariance
#    -> joint TT+TE+EE fit
# ---------------------------------------------------------------------------

def n_bins_upto(lmax_cut: int, bandpower_width: int = 30, ell_start: int = 2) -> int:
    """
    Bandpowers kept per spectrum after dropping bin 0: a linear
    `bandpower_width` binning starting at `ell_start` gives
    (lmax_cut - ell_start + 1) // bandpower_width bins covering
    ell=[ell_start, lmax_cut] -- minus 1 for the dropped first bin.
    """
    return (lmax_cut - ell_start + 1) // bandpower_width - 1


def fit_one_patch(
    maps_hm1,
    maps_hm2,
    mask_t,
    mask_pol,
    nside: int,
    lmax_tt: int,
    lmax_te: int,
    lmax_ee: int,
    bandpower_width: int,
    beam_function: np.ndarray,
    pixel_window_t: np.ndarray,
    pixel_window_pol: np.ndarray,
    cl_fiducial: dict,
    initial_guess: dict,
    bounds: dict | None = None,
    fixed: set | None = None,
    max_cov_cond: float = 1e12,
    likelihood_cache: str | None = None,
    likelihood_meta: dict | None = None,
    **fit_kwargs,
) -> dict:
    """
    Full TT+TE+EE pipeline on one sky region: hm1 x hm2 cross-spectra with
    their own mode-coupling workspaces, the joint Gaussian covariance, and
    the chi^2 fit.

    With `likelihood_cache` (a .npz path), the expensive part -- spectra,
    workspaces, covariance -- is computed once and stored as a likelihood
    package (`save_likelihood`); later calls with the same
    `likelihood_meta` (the data-side settings: maps, masks, lmax, binning,
    beam, ...) load it and go straight to the fit, so refits with other
    fixed parameters / priors / minimizers cost only CAMB time.

    Parameters
    ----------
    maps_hm1, maps_hm2 : (T, Q, U) sequences of HEALPix maps at `nside`
        The two half-mission maps (independent noise), uK. The region is
        selected by the masks, not by the maps, so pass full-sky maps.
    mask_t, mask_pol : ndarray
        Apodized region masks for temperature and polarization (e.g. a
        superpixel mask x survey mask).
    nside : int
        Resolution of the maps/masks.
    lmax_tt, lmax_te, lmax_ee : int
        Per-spectrum multipole cuts. The shared field/CAMB lmax is
        max(lmax_tt, lmax_te, lmax_ee); `beam_function` and both pixel
        windows must reach at least that far.
    bandpower_width : int
    beam_function, pixel_window_t, pixel_window_pol : ndarray
    cl_fiducial : dict
        CAMB spectra ("TT", "TE", "EE", "BB") from `camb_cl`, for the
        analytic covariance; must reach the shared lmax.
    initial_guess, bounds, fixed, **fit_kwargs
        Forwarded to `fit_parameters` -- including `priors=` (Gaussian
        priors on the point-source amplitudes, which a patch can't
        constrain by itself), `initial_step=` and `method=`.
    max_cov_cond : float
        A covariance more ill-conditioned than this raises RuntimeError
        instead of being inverted into numerical garbage.
    likelihood_cache, likelihood_meta : see above. The meta dict must
        describe everything the spectra depend on; a mismatch recomputes.

    Returns
    -------
    dict with keys 'f_sky_t', 'f_sky_pol', 'ell', 'cl_data', 'errors'
    (the stacked TT,TE,EE error vector), 'error_slices', 'result' (the
    `fit_parameters` return value) and 'fit_data' (the FitData).
    """
    if likelihood_cache is not None:
        package = load_likelihood(likelihood_cache, likelihood_meta)
        if package is not None:
            fit_data, extra = package
            result = fit_parameters(fit_data, initial_guess, bounds, fixed=fixed, **fit_kwargs)
            return dict(extra, result=result, fit_data=fit_data, likelihood_loaded=True)

    lmax = max(lmax_tt, lmax_te, lmax_ee)
    map_t_hm1, map_q_hm1, map_u_hm1 = maps_hm1
    map_t_hm2, map_q_hm2, map_u_hm2 = maps_hm2

    p_t_hm1 = _patch.extract_patch(map_t_hm1, mask_t)
    p_t_hm2 = _patch.extract_patch(map_t_hm2, mask_t)
    p_pol_hm1 = _patch.extract_patch(np.array([map_q_hm1, map_u_hm1]), mask_pol)
    p_pol_hm2 = _patch.extract_patch(np.array([map_q_hm2, map_u_hm2]), mask_pol)

    def build(p, spin, pixel_window):
        return _power_spectrum.build_field(
            p, spin=spin, beam_function=beam_function,
            pixel_window_function=pixel_window, lmax=lmax,
        )

    field_t_hm1 = build(p_t_hm1, 0, pixel_window_t)
    field_t_hm2 = build(p_t_hm2, 0, pixel_window_t)
    field_pol_hm1 = build(p_pol_hm1, 2, pixel_window_pol)
    field_pol_hm2 = build(p_pol_hm2, 2, pixel_window_pol)

    bins = _power_spectrum.make_bins(nside, bandpower_width=bandpower_width, lmax=lmax)

    spec_tt = _power_spectrum.compute_power_spectrum(field_t_hm1, field_t_hm2, bins)
    spec_te = _power_spectrum.compute_power_spectrum(field_t_hm1, field_pol_hm2, bins)
    spec_ee = _power_spectrum.compute_power_spectrum(field_pol_hm1, field_pol_hm2, bins)
    ws_tt, ws_te, ws_ee = spec_tt["workspace"], spec_te["workspace"], spec_ee["workspace"]

    n_tt = n_bins_upto(lmax_tt, bandpower_width)
    n_te = n_bins_upto(lmax_te, bandpower_width)
    n_ee = n_bins_upto(lmax_ee, bandpower_width)

    ell = {
        "TT": spec_tt["ell"][1:1 + n_tt],
        "TE": spec_te["ell"][1:1 + n_te],
        "EE": spec_ee["ell"][1:1 + n_ee],
    }
    cl = {
        "TT": spec_tt["cl"][1:1 + n_tt],
        "TE": spec_te["cl"][1:1 + n_te],
        "EE": spec_ee["cl"][1:1 + n_ee],
    }
    cl_data = np.concatenate([cl["TT"], cl["TE"], cl["EE"]])

    # Noise level of each half-mission split, for the covariance (see
    # power_spectrum.estimate_noise_cl).
    noise_tt = _power_spectrum.estimate_noise_cl(
        map_t_hm1, map_t_hm2, mask_t, lmax,
        beam_function=beam_function, pixel_window_function=pixel_window_t, spin=0,
    )
    noise_ee = _power_spectrum.estimate_noise_cl(
        [map_q_hm1, map_u_hm1], [map_q_hm2, map_u_hm2], mask_pol, lmax,
        beam_function=beam_function, pixel_window_function=pixel_window_pol, spin=2,
    )

    cov = _power_spectrum.compute_joint_tt_te_ee_covariance(
        ws_tt, ws_te, ws_ee,
        field_t_hm1, field_t_hm2, field_pol_hm1, field_pol_hm2,
        cl_fiducial["TT"], cl_fiducial["TE"], cl_fiducial["EE"], cl_fiducial["BB"],
        noise_tt, noise_ee,
        n_tt, n_te, n_ee,
        transfer_t=beam_function * pixel_window_t,
        transfer_pol=beam_function * pixel_window_pol,
    )
    errors = _power_spectrum.compute_errors(cov)

    # A near-singular covariance (small f_sky, few independent modes) would
    # otherwise silently invert into numerical garbage that poisons the fit
    # without raising -- fail loudly so callers can record the region as failed.
    cond = np.linalg.cond(cov)
    if cond > max_cov_cond:
        raise RuntimeError(f"ill-conditioned covariance (cond={cond:.2e})")
    cov_inv = np.linalg.inv(cov)

    # Keep only the bandpower windows (MBs), not the workspaces (GBs): the
    # FitData stays small enough to hold for all 12 patches and to save.
    fit_data = FitData(
        cl_data=cl_data, cov_inv=cov_inv,
        workspace_tt=None, workspace_te=None, workspace_ee=None,
        lmax=lmax, n_tt=n_tt, n_te=n_te, n_ee=n_ee,
        windows_tt=bandpower_windows(ws_tt, n_tt),
        windows_te=bandpower_windows(ws_te, n_te),
        windows_ee=bandpower_windows(ws_ee, n_ee),
    )
    extra = dict(
        f_sky_t=float(p_t_hm1["f_sky"]), f_sky_pol=float(p_pol_hm1["f_sky"]),
        ell=ell, cl_data=cl, errors=errors,
        error_slices={
            "TT": slice(0, n_tt),
            "TE": slice(n_tt, n_tt + n_te),
            "EE": slice(n_tt + n_te, n_tt + n_te + n_ee),
        },
    )
    del spec_tt, spec_te, spec_ee, ws_tt, ws_te, ws_ee
    del field_t_hm1, field_t_hm2, field_pol_hm1, field_pol_hm2
    if likelihood_cache is not None:
        save_likelihood(likelihood_cache, fit_data, extra, cov, likelihood_meta)

    result = fit_parameters(fit_data, initial_guess, bounds, fixed=fixed, **fit_kwargs)
    return dict(extra, result=result, fit_data=fit_data, likelihood_loaded=False)


# ---------------------------------------------------------------------------
# 5b. Likelihood packages: everything a fit needs, without NaMaster objects
# ---------------------------------------------------------------------------

def save_likelihood(path: str, fit_data: "FitData", extra: dict, covariance: np.ndarray,
                    meta: dict | None) -> None:
    """
    Store a joint TT+TE+EE likelihood as a compressed .npz: data vector,
    covariance (and its inverse), bandpower windows, bin counts, effective
    ells, errors, f_sky, plus `meta` (a JSON-able dict of the data-side
    settings it was computed under, checked again by `load_likelihood`).
    A few MB per patch at lmax = 2011.
    """
    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"  # write-then-rename, as in save_patch_result
    np.savez_compressed(
        tmp,
        meta=json.dumps(meta or {}, sort_keys=True),
        cl_data=fit_data.cl_data, covariance=covariance, cov_inv=fit_data.cov_inv,
        windows_tt=fit_data.windows_tt, windows_te=fit_data.windows_te,
        windows_ee=fit_data.windows_ee,
        lmax=fit_data.lmax, n_tt=fit_data.n_tt, n_te=fit_data.n_te, n_ee=fit_data.n_ee,
        tau=fit_data.tau, mnu=fit_data.mnu, r=fit_data.r,
        ell_tt=extra["ell"]["TT"], ell_te=extra["ell"]["TE"], ell_ee=extra["ell"]["EE"],
        errors=extra["errors"], f_sky_t=extra["f_sky_t"], f_sky_pol=extra["f_sky_pol"],
    )
    os.replace(tmp, path)


def load_likelihood(path: str, meta: dict | None = None):
    """
    Load a package written by `save_likelihood`. Returns (fit_data, extra)
    -- `extra` has the same 'f_sky_t', 'f_sky_pol', 'ell', 'cl_data',
    'errors', 'error_slices' keys `fit_one_patch` returns -- or None if the
    file is missing/unreadable or was computed under a different `meta`.
    """
    import json
    import os

    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        stored_meta = json.loads(str(d["meta"]))
    except (OSError, ValueError, KeyError):
        return None
    if meta is not None and stored_meta != json.loads(json.dumps(meta, sort_keys=True)):
        return None

    n_tt, n_te, n_ee = int(d["n_tt"]), int(d["n_te"]), int(d["n_ee"])
    fit_data = FitData(
        cl_data=d["cl_data"], cov_inv=d["cov_inv"],
        workspace_tt=None, workspace_te=None, workspace_ee=None,
        lmax=int(d["lmax"]), n_tt=n_tt, n_te=n_te, n_ee=n_ee,
        tau=float(d["tau"]), mnu=float(d["mnu"]), r=float(d["r"]),
        windows_tt=d["windows_tt"], windows_te=d["windows_te"], windows_ee=d["windows_ee"],
    )
    cl_data = d["cl_data"]
    extra = dict(
        f_sky_t=float(d["f_sky_t"]), f_sky_pol=float(d["f_sky_pol"]),
        ell={"TT": d["ell_tt"], "TE": d["ell_te"], "EE": d["ell_ee"]},
        cl_data={"TT": cl_data[:n_tt], "TE": cl_data[n_tt:n_tt + n_te],
                 "EE": cl_data[n_tt + n_te:]},
        errors=d["errors"],
        error_slices={"TT": slice(0, n_tt), "TE": slice(n_tt, n_tt + n_te),
                      "EE": slice(n_tt + n_te, n_tt + n_te + n_ee)},
    )
    return fit_data, extra


# ---------------------------------------------------------------------------
# 6. Per-patch result cache -- lets an interrupted multi-patch run resume
# ---------------------------------------------------------------------------

def save_patch_result(path: str, patch_result: dict, config: dict) -> None:
    """
    Write a `fit_one_patch` result to `path` as JSON, together with the
    `config` dict it was computed under (see `load_patch_result`).

    Not saved: the raw Minuit object and the FitData (NaMaster workspaces),
    which can't be serialized to JSON -- a result reloaded from disk has
    'fit_data' = None and no result['minuit'].
    """
    import json

    res = patch_result["result"]
    payload = {
        "config": config,
        "f_sky_t": float(patch_result["f_sky_t"]),
        "f_sky_pol": float(patch_result["f_sky_pol"]),
        "ell": {k: np.asarray(v).tolist() for k, v in patch_result["ell"].items()},
        "cl_data": {k: np.asarray(v).tolist() for k, v in patch_result["cl_data"].items()},
        "errors": np.asarray(patch_result["errors"]).tolist(),
        "error_slices": {k: [s.start, s.stop] for k, s in patch_result["error_slices"].items()},
        "result": {
            "best_fit": {k: float(v) for k, v in res["best_fit"].items()},
            "errors": {k: float(v) for k, v in res["errors"].items()},
            "errors_fisher": {k: float(v) for k, v in res["errors_fisher"].items()},
            "errors_hesse": (None if res.get("errors_hesse") is None
                             else {k: float(v) for k, v in res["errors_hesse"].items()}),
            "covariance": np.asarray(res["covariance"], dtype=float).tolist(),
            "free_params": list(res["free_params"]),
            "chi2": float(res["chi2"]),
            "chi2_prior": float(res["chi2_prior"]),
            "chi2_wall": float(res["chi2_wall"]),
            "n_data": int(res["n_data"]),
            "valid": bool(res["valid"]),
            "status": res["status"],
            "flags": list(res["flags"]),
            "params_at_bound": list(res["params_at_bound"]),
            "method": res["method"],
            "n_iter": int(res["n_iter"]),
            "n_camb_solves": int(res["n_camb_solves"]),
        },
    }
    tmp = path + ".tmp"  # write-then-rename: a crash mid-write can't leave a corrupt cache file
    with open(tmp, "w") as f:
        json.dump(payload, f)
    import os
    os.replace(tmp, path)


def load_patch_result(path: str, config: dict) -> dict | None:
    """
    Load a result saved by `save_patch_result`, or None if `path` doesn't
    exist, is unreadable (e.g. a run killed mid-write), or was computed
    under a different `config` (so a changed lmax / binning / initial guess
    / bounds / priors / method recomputes rather than silently reusing
    stale results).
    """
    import json
    import os

    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if d.get("config") != json.loads(json.dumps(config)):
        return None

    result = dict(d["result"])
    result["covariance"] = np.array(result["covariance"])
    return dict(
        f_sky_t=d["f_sky_t"], f_sky_pol=d["f_sky_pol"],
        ell={k: np.array(v) for k, v in d["ell"].items()},
        cl_data={k: np.array(v) for k, v in d["cl_data"].items()},
        errors=np.array(d["errors"]),
        error_slices={k: slice(*v) for k, v in d["error_slices"].items()},
        result=result,
        fit_data=None,
    )
