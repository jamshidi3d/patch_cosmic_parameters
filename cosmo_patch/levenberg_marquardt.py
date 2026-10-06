"""
levenberg_marquardt.py
----------------------
Levenberg-Marquardt minimizer for the CAMB chi^2 fits -- the alternative to
iminuit/MIGRAD selected with `fit_parameters(..., method="levenberg_marquardt")`.

The chi^2 here is a Gaussian likelihood, i.e. a sum of squared (whitened)
residuals r(p): chi^2 = r^T r. For that structure Gauss-Newton is the
natural algorithm: linearize r(p + d) ~ r(p) + J d and solve the normal
equations (J^T J) d = -J^T r. Marquardt damping,

    (J^T J + lambda * diag(J^T J)) d = -J^T r,

interpolates between Gauss-Newton (lambda -> 0) and scaled steepest descent
(large lambda), so a step that doesn't reduce chi^2 is retried shorter
instead of being accepted. J comes from FitProblem.jacobian (central
differences with fixed physical steps, far above CAMB's numerical noise),
so no tiny-step derivatives of a noisy chi^2 are ever taken. Errors are the
Gauss-Newton (Fisher) covariance (J^T J)^-1 at the minimum.

The minimizer only talks to a `FitProblem` (fitting.py) through
`residual(p)`, `jacobian(p)`, `p0` and `scale` -- it knows nothing about
CAMB, NaMaster or which spectra are fit. One pass, no restarts.
"""

from __future__ import annotations

import numpy as np


def fit_levenberg_marquardt(
    problem,
    max_iter: int = 40,
    lambda_init: float = 1e-3,
    lambda_max: float = 1e8,
    edm_tol: float = 1e-3,
) -> dict:
    """
    Minimize problem.residual(p)^T problem.residual(p), starting at problem.p0.

    Parameters
    ----------
    problem : fitting.FitProblem
    max_iter : int
        Maximum number of Jacobian evaluations (each costs 2 x n_free
        theory evaluations).
    lambda_init, lambda_max : float
        Initial Marquardt damping, and the damping at which a run of
        rejected steps is declared stalled.
    edm_tol : float
        Convergence: the Gauss-Newton estimated distance to the minimum,
        EDM = g^T (J^T J)^-1 g / 2 with g = J^T r (the same quantity
        MIGRAD's EDM measures, in chi^2 units), below this.

    Returns
    -------
    dict with keys 'p' (best fit, free parameters), 'cov' (Gauss-Newton
    covariance), 'flags' (empty if converged), 'valid', 'n_iter',
    'edm', 'chi2'.
    """
    p = problem.p0.copy()
    r = problem.residual(p)
    chi2 = float(r @ r)
    lam = lambda_init
    flags = []
    edm = np.inf
    n_iter = 0
    converged = False

    while n_iter < max_iter:
        jac = problem.jacobian(p)
        n_iter += 1
        a = jac.T @ jac
        g = jac.T @ r
        try:
            edm = 0.5 * float(g @ np.linalg.solve(a, g))
        except np.linalg.LinAlgError:
            edm = np.inf
        if edm < edm_tol:
            converged = True
            break

        diag = np.diag(a).copy()
        diag[diag <= 0] = 1.0
        accepted = False
        while lam <= lambda_max:
            try:
                step = np.linalg.solve(a + lam * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            p_new = p + step
            try:
                r_new = problem.residual(p_new)
            except Exception:  # theory failure (e.g. CAMB) -> treat as uphill
                lam *= 10.0
                continue
            chi2_new = float(r_new @ r_new)
            if chi2_new < chi2:
                p, r, chi2 = p_new, r_new, chi2_new
                lam = max(lam / 10.0, 1e-9)
                accepted = True
                break
            lam *= 10.0

        if not accepted:
            # No downhill step at any damping: we are at the minimum up to
            # the theory's numerical noise, unless EDM says otherwise.
            converged = edm < 10 * edm_tol
            if not converged:
                flags.append(f"lm_stalled(edm={edm:.3g})")
            break
    else:
        flags.append(f"lm_max_iter(edm={edm:.3g})")
        jac = problem.jacobian(p)  # p moved after the last Jacobian
    try:
        cov = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov = None
        flags.append("fisher_singular")
    if cov is not None:
        # Test the correlation matrix, not cov itself: the raw entries span
        # ~1e-21 (As^2) to ~1e1 (H0^2), so eigvalsh(cov) loses the small
        # eigenvalues to rounding and reports a false negative one.
        sigma = np.sqrt(np.abs(np.diag(cov)))
        if np.any(np.diag(cov) <= 0) or not np.all(
            np.linalg.eigvalsh(cov / np.outer(sigma, sigma)) > 0
        ):
            flags.append("covariance_not_posdef")

    return {
        "p": p,
        "cov": cov,
        "flags": flags,
        "valid": converged and not flags,
        "n_iter": n_iter,
        "edm": edm,
        "chi2": chi2,
    }
