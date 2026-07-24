"""
patch.py
--------
Utilities for defining a sky patch on a HEALPix map: building a mask,
apodizing it, and packaging (map, mask) ready for power-spectrum estimation.

Conventions
-----------
- All maps are HEALPix, RING ordering, float64.
- Masks are weight maps in [0, 1] (1 = fully unmasked). A binary mask is a
  special case of this.
"""

from __future__ import annotations

import numpy as np
import healpy as hp


def make_circular_mask(
    nside: int,
    center_lonlat_deg: tuple[float, float],
    radius_deg: float,
) -> np.ndarray:
    """
    Build a binary circular mask on a HEALPix grid.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    center_lonlat_deg : (lon, lat) in degrees
        Patch center in Galactic (or whatever frame your map uses) lon/lat.
    radius_deg : float
        Angular radius of the patch, in degrees.

    Returns
    -------
    mask : ndarray, shape (12*nside**2,)
        Binary mask (1 inside the patch, 0 outside).
    """
    npix = hp.nside2npix(nside)
    vec_center = hp.ang2vec(center_lonlat_deg[0], center_lonlat_deg[1], lonlat=True)
    pix_in_disc = hp.query_disc(nside, vec_center, np.radians(radius_deg))
    mask = np.zeros(npix)
    mask[pix_in_disc] = 1.0
    return mask


def make_superpixel_mask(
    nside: int,
    low_nside: int,
    low_pix: int,
) -> np.ndarray:
    """
    Build a binary mask selecting the high-resolution pixels nested inside a
    single "superpixel" of a lower-resolution HEALPix grid.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter of the output mask (must be >= low_nside).
    low_nside : int
        HEALPix resolution parameter of the coarse grid defining the patch.
    low_pix : int
        RING-ordered pixel index in the low_nside grid to select.

    Returns
    -------
    mask : ndarray, shape (12*nside**2,)
        Binary mask (1 inside the superpixel, 0 outside).
    """
    if nside < low_nside:
        raise ValueError(f"nside ({nside}) must be >= low_nside ({low_nside})")
    if not hp.isnsideok(low_nside):
        raise ValueError(f"low_nside ({low_nside}) is not a valid HEALPix nside")

    ratio = nside // low_nside
    if nside % low_nside != 0 or ratio & (ratio - 1) != 0:
        raise ValueError(
            f"nside ({nside}) must be low_nside ({low_nside}) times a power of two"
        )

    npix = hp.nside2npix(nside)
    factor = ratio * ratio

    low_pix_nest = hp.ring2nest(low_nside, low_pix)
    high_pix_nest = np.arange(low_pix_nest * factor, (low_pix_nest + 1) * factor)
    high_pix_ring = hp.nest2ring(nside, high_pix_nest)

    mask = np.zeros(npix)
    mask[high_pix_ring] = 1.0
    return mask


def apodize_mask(mask: np.ndarray, apodize_deg: float, method: str = "C1") -> np.ndarray:
    """
    Apodize a binary mask so its edges taper smoothly to zero.

    This matters for NaMaster: a hard-edged mask injects power at high ell
    through mode coupling, and an un-apodized patch will bias the pseudo-Cl
    estimate. Always apodize before feeding a mask into NmtField.

    Parameters
    ----------
    mask : ndarray
        Binary (or already-weighted) mask.
    apodize_deg : float
        Apodization scale in degrees.
    method : str
        NaMaster apodization type: "C1", "C2", or "Smooth".

    Returns
    -------
    apodized_mask : ndarray
    """
    import pymaster as nmt

    if apodize_deg <= 0:
        return mask
    return nmt.mask_apodization(mask, apodize_deg, apotype=method)


def extract_patch(
    sky_map: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Package a full-sky map + mask into a validated patch, ready for NaMaster.

    Parameters
    ----------
    sky_map : ndarray, shape (npix,) or (2, npix)
        Full-sky HEALPix map (e.g. a temperature or convergence map), or a
        [Q, U] pair for a spin-2 (polarization) field.
    mask : ndarray, shape (npix,)
        Weight/apodized mask in [0, 1], same nside as sky_map.

    Returns
    -------
    patch : dict with keys
        'map'   : the input map (unchanged; NaMaster wants the unmasked map
                  and the weight mask separately, it does the multiplication)
        'mask'  : the weight mask
        'nside' : HEALPix nside
        'f_sky' : effective sky fraction, mean(mask**2) as the standard estimator
    """
    if sky_map.shape[-1] != mask.shape[-1]:
        raise ValueError(
            f"map and mask pixel-count mismatch: {sky_map.shape} vs {mask.shape}"
        )

    nside_map = hp.npix2nside(sky_map.shape[-1])
    nside_mask = hp.npix2nside(mask.size)
    if nside_map != nside_mask:
        raise ValueError(f"nside mismatch: map={nside_map}, mask={nside_mask}")

    f_sky = np.mean(mask**2)
    if f_sky <= 0:
        raise ValueError("mask is empty (f_sky = 0) -- check patch definition")
    if f_sky >= 0.999:
        raise ValueError(
            "mask covers essentially the full sky (f_sky ~ 1) -- "
            "this looks like a full-sky map, not a patch"
        )

    return {
        "map": sky_map,
        "mask": mask,
        "nside": nside_map,
        "f_sky": f_sky,
    }
