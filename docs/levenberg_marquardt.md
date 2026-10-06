# The Levenberg–Marquardt fitting method

`fit_parameters(..., method="levenberg_marquardt")` is implemented in
`cosmo_patch/levenberg_marquardt.py`. This note explains the method and why it
suits this problem. How it sits in the rest of the pipeline is covered in
[theory.md](theory.md).

## 1. The problem is a least-squares problem

The likelihood is Gaussian, so the quantity to minimize is

$$
\chi^2(p) = \big(d - t(p)\big)^{\mathsf T}\, C^{-1}\, \big(d - t(p)\big),
$$

where $d$ are the measured bandpowers, $t(p)$ the binned theory at parameters $p$,
and $C$ the bandpower covariance.

Factor $C^{-1} = L L^{\mathsf T}$ (Cholesky) and define the **whitened residual**
$r(p) = L^{\mathsf T}(d - t(p))$. Then

$$
\chi^2(p) = r(p)^{\mathsf T} r(p) = \sum_i r_i(p)^2 .
$$

Each $r_i$ is "data minus model in units of σ", decorrelated across bandpowers.

Gaussian priors and the validity wall are added as extra rows of the same vector:
- a prior on $p_k$ is the row $(p_k-\mu_k)/\sigma_k$;
- the wall is the row $\sqrt{K}\,(p_k-\mathrm{clip}(p_k))/s_k$, which is zero inside
  the bounds.

So the total cost is still a plain sum of squares (`FitProblem.residual`).

## 2. Gauss–Newton: solve a linear problem at every step

Near the current point, linearize the residual,
$r(p+\delta) \approx r(p) + J\delta$, with Jacobian $J = \partial r/\partial p$.
Minimizing $\|r + J\delta\|^2$ over $\delta$ is a linear least-squares problem, with
the *normal equations*

$$
(J^{\mathsf T}J)\,\delta = -J^{\mathsf T} r .
$$

- If the model were linear in $p$, one such step would land exactly on the minimum.
- The CMB spectra are smooth and only mildly non-linear in the ΛCDM parameters over
  the range a fit explores. A few steps therefore suffice, typically 3–5 iterations
  here.
- The point-source amplitudes enter *exactly* linearly.

$J^{\mathsf T}J$ is the Gauss–Newton approximation to half the Hessian of $\chi^2$.
It drops the term $\sum_i r_i\,\partial^2 r_i/\partial p^2$, which is small when the
fit is good, because the residuals are O(1) and the model is nearly linear.

## 3. Marquardt damping: never accept an uphill step

Far from the minimum, or along a strongly curved degeneracy (H0–Ωch²–ns), a full
Gauss–Newton step can overshoot. Levenberg–Marquardt solves instead

$$
\big(J^{\mathsf T}J + \lambda\, \mathrm{diag}(J^{\mathsf T}J)\big)\,\delta = -J^{\mathsf T} r .
$$

- **$\lambda \to 0$:** pure Gauss–Newton (fast, near the minimum).
- **Large $\lambda$:** a short step along the gradient, scaled per parameter by
  $\mathrm{diag}(J^{\mathsf T}J)$. That scaling (Marquardt's choice) makes the method
  insensitive to parameter units, so As ~ 1e-9 and H0 ~ 70 are handled alike.

The step-acceptance loop:
1. Propose $\delta$ and evaluate $\chi^2(p+\delta)$.
2. If $\chi^2$ decreased, accept and set $\lambda \leftarrow \lambda/10$, trusting the
   linear model more.
3. Otherwise, reject and set $\lambda \leftarrow 10\lambda$, taking a shorter, more
   gradient-like step.

A step that fails inside CAMB counts as rejected. $\lambda$ starts at $10^{-3}$, is
floored at $10^{-9}$, and gives up at $10^{8}$.

## 4. Convergence and error bars

**Stopping rule.** Before each step, the code computes the estimated distance to the
minimum,

$$
\mathrm{EDM} = \tfrac12\, g^{\mathsf T}(J^{\mathsf T}J)^{-1} g, \qquad g = J^{\mathsf T} r,
$$

which is half the $\chi^2$ decrease a full Gauss–Newton step would achieve.
- It stops when EDM < $10^{-3}$, far below the Δχ² = 1 that defines a 1σ error.
- If no step is accepted at any damping, the fit is at the minimum to within the
  model's numerical noise. It is accepted if EDM < $10^{-2}$; otherwise it is flagged
  `lm_stalled`.
- Reaching 40 iterations is flagged `lm_max_iter`.

**Errors.** For $\chi^2 = r^{\mathsf T}r$ with errordef Δχ² = 1, the parameter
covariance is twice the inverse Hessian, which in the Gauss–Newton approximation is

$$
\mathrm{Cov}(p) = (J^{\mathsf T}J)^{-1}.
$$

This is the Fisher matrix of the Gaussian likelihood evaluated at the best fit. Prior
rows enter it automatically, so a prior's uncertainty propagates into every
parameter. Positive-definiteness is checked on the correlation matrix, because the
raw entries span ~$10^{-21}$ to ~$10^{1}$.

## 5. The Jacobian: fixed physical steps

$J$ is computed by central differences in each free parameter, with steps from
`fitting.DEFAULT_STEP` (H0 0.5, Ωbh² 1e-4, Ωch² 1e-3, As 1e-11, ns 5e-3, A_ps^TT 5,
A_ps^EE 0.5). These are about 0.2–1 σ of a full-sky fit.

The steps are deliberately *not* small. CAMB's χ² has step-like numerical jumps of
~$10^{-4}$ (along H0 and Ωch²), and finite differences over tiny steps would measure
those jumps instead of the real slope.
- Checked: second derivatives from steps of 0.001–0.01 in H0 scatter by ±40%.
- With the chosen steps, the Fisher errors are unchanged to three digits between 0.5×
  and 2× the step.

A Jacobian costs $2\,n_{\rm cosmo}$ CAMB solves (10 for 5 ΛCDM parameters).
Columns for the point-source amplitudes are free, because `camb_cl` is memoized and
those amplitudes don't change the CAMB call.

## 6. Compared with MIGRAD

| | Levenberg–Marquardt | iminuit / MIGRAD (as used here) |
|---|---|---|
| Uses | residual vector $r$ and $J$ | scalar χ² and its gradient $2J^{\mathsf T}r$ |
| Curvature model | $J^{\mathsf T}J$, exact for a linear model, from step 1 | built up over iterations (variable-metric) |
| Typical cost per fit | 3–5 Jacobians, 20–35 CAMB solves | 100–180 CAMB solves |
| Errors reported | $(J^{\mathsf T}J)^{-1}$ | the same $(J^{\mathsf T}J)^{-1}$ at the MIGRAD minimum |

- **Agreement.** On real patches (7 and 9) and in synthetic tests the two methods agree
  to < 0.01σ, with identical errors. LM is about 3–6× cheaper in CAMB solves (22–33
  vs 102–209 in those tests) because it exploits the least-squares structure.
- **Limits.** The Gauss–Newton approximation needs residuals of order 1 and a model
  that is nearly linear across about 1σ. A very poor fit (χ²/n ≫ 1) or a strongly
  non-Gaussian likelihood would call for a profile or MCMC error analysis instead.
