"""
cosmo_patch: patch power spectrum + cosmological parameter fitting.

Typical usage
-------------
    from cosmo_patch import patch, power_spectrum, fitting

    mask = patch.make_circular_mask(nside=256, center_lonlat_deg=(30, 40), radius_deg=10)
    mask = patch.apodize_mask(mask, apodize_deg=1.0)
    p = patch.extract_patch(sky_map, mask)

    field = power_spectrum.build_field(p)
    bins = power_spectrum.make_bins(p["nside"], bandpower_width=20)
    spec = power_spectrum.compute_power_spectrum(field, bins)

    cov = power_spectrum.compute_gaussian_covariance(spec["workspace"], field, fiducial_cl)
    errors = power_spectrum.compute_errors(cov)

    data = fitting.FitData(ell=spec["ell"], cl_data=spec["cl"],
                            cov_inv=np.linalg.inv(cov),
                            workspace=spec["workspace"], lmax=3*p["nside"]-1)
    result = fitting.fit_parameters(data, initial_guess={...}, bounds={...})
"""

from . import patch, power_spectrum, fitting

__all__ = ["patch", "power_spectrum", "fitting"]
