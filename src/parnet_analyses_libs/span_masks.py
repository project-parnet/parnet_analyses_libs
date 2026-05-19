"""Functions related to binary span masks and probability density profiles.

Note:
    The span-mask utilities have been moved to ``pylbsr.bio.span_masks``.
    They are still importable from this module for backwards compatibility,
    but will raise a ``DeprecationWarning``. Update your imports, e.g.::

        # before
        from parnet_analyses_libs.span_masks import intervals_to_span_masks
        # after
        from pylbsr.bio.span_masks import intervals_to_span_masks

    Moved names: coordinates_from_binary_mask, coordinates_from_binary_masks,
    intervals_to_span_masks, combine_span_masks_on_identifiers, scatter_span_masks,
    RelativeCoordinatesIntervals, relative_coordinates_to_scattered_span_masks.
"""

import warnings

import numpy as np
import pylbsr.bio.span_masks as _pylbsr_span_masks
from jaxtyping import Array, Bool, Float

# ---------------------------------------------------------------------------
# Deprecation shim — names moved to pylbsr.bio.span_masks
# ---------------------------------------------------------------------------

_MOVED_TO_PYLBSR: set[str] = {
    "coordinates_from_binary_mask",
    "coordinates_from_binary_masks",
    "intervals_to_span_masks",
    "combine_span_masks_on_identifiers",
    "scatter_span_masks",
    "RelativeCoordinatesIntervals",
    "relative_coordinates_to_scattered_span_masks",
}


def __getattr__(name: str) -> object:
    """Re-export moved names with a DeprecationWarning."""
    if name in _MOVED_TO_PYLBSR:
        warnings.warn(
            f"{name!r} has moved to 'pylbsr.bio.span_masks'. "
            f"Update your import:\n"
            f"    from pylbsr.bio.span_masks import {name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_pylbsr_span_masks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# HPD utilities — kept here (different conceptual domain from span masks)
# ---------------------------------------------------------------------------


def make_hpd_masks_from_probability_vector(
    p: Float[Array, " N"],
    alpha: float = 0.05,
    atol: float = 1e-8,
) -> Bool[Array, " N"]:
    """Highest-Posterior-Density (HPD) mask for a discrete PDF.

    Parameters
    ----------
    p : 1‑D array‑like
        Probabilities (Σ p = 1).
    alpha : float, default 0.05
        Mass allowed outside the HPD set (e.g. 0.05 ⇒ 95 % HPD).
    atol : float, default 1e-8
        Absolute tolerance for checking if p sums to 1.

    Returns:
    -------
    mask : np.ndarray
        Boolean vector (True for HPD positions).
    """
    if p.ndim != 1:
        raise ValueError("p must be 1‑D")

    if not np.isclose(p.sum(), 1.0, atol=atol):
        raise ValueError("p must sum to 1")

    idx_sorted = np.argsort(p)[::-1]
    cumsum = np.cumsum(p[idx_sorted])
    k = np.searchsorted(cumsum, 1 - alpha) + 1
    hpd_idx = idx_sorted[:k]

    mask = np.zeros_like(p, dtype=bool)
    mask[hpd_idx] = True
    return mask


def make_hpd_segments(
    p: Float[Array, " N"],
    alpha: float = 0.05,
    atol: float = 1e-8,
) -> list[tuple[int, int]]:
    """Segments of the highest-posterior-density (HPD) set.

    Parameters
    ----------
    p : 1‑D array‑like
        Probabilities (Σ p = 1).
    alpha : float, default 0.05
        Mass allowed outside the HPD set (e.g. 0.05 ⇒ 95 % HPD).
    atol : float, default 1e-8
        Absolute tolerance for checking if p sums to 1.

    Returns:
    -------
    segments : list of tuples
        List of (start, end) tuples for each segment in the HPD set.
    """
    mask = make_hpd_masks_from_probability_vector(p, alpha=alpha, atol=atol)
    coords = _pylbsr_span_masks.coordinates_from_binary_mask(mask.astype(np.int64))
    return [(int(s), int(e)) for s, e in coords]
