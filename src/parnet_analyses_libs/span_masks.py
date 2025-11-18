"""Functions to create and manipulate binary span masks from intervals.

Mostly aimed at working with genomic intervals and converting from
relative coordinates to binary masks of length L (e.g. length of a sequence).

A span mask is a binary array of length L with 1s at positions covered by
the intervals and 0s elsewhere.
"""

from collections.abc import Hashable, Sequence

import numpy as np
import pandera.pandas as pa
from jaxtyping import Array, Bool, Float, Int64, Integer
from pandera.typing.pandas import DataFrame


def coordinates_from_binary_mask(
    binary_mask: Integer[Array, " length"],
) -> Int64[Array, "n_regions 2"]:
    """Return an array of [start, end) coordinates of contiguous regions with 1 in the array."""
    return np.where(np.diff(binary_mask, prepend=0, append=0) != 0)[0].reshape(-1, 2).astype(np.int64)


def coordinates_from_binary_masks(
    binary_masks: Integer[Array, "#batch_size n_masks length"],
) -> Int64[Array, "#batch_size n_masks k 2"]:
    """Return a ragged array with for each n_masks an array of k>=0 [start, end) coordinates."""
    result = np.vectorize(
        coordinates_from_binary_mask,
        signature="(n)->()",
        otypes=[object],
    )(binary_masks)
    return result.astype(np.int64)


def intervals_to_span_masks(
    starts: Integer[Array, " intervals"],
    ends: Integer[Array, " intervals"],
    length: int,
) -> Int64[Array, "intervals length"]:
    """Produce binary arrays of length `length` with `1` at positions from each (start, end) interval.

    Example:
        >>> starts = np.array([0, 4, 7])
        >>> ends = np.array([4, 8, 10])
        >>> length = 11
        >>> span_masking_intervals(starts, ends, length)
        array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]], dtype=int64)

    """
    positions: Int64[Array, " length"] = np.arange(length)
    positions: Int64[Array, "length singleton"] = positions[:, np.newaxis]
    span_masks: Int64[Array, "intervals length"] = (
        (positions >= starts) & (positions < ends)
    ).T.astype(np.int64)
    return span_masks


def combine_span_masks_on_identifiers(
    span_masks: Integer[Array, "intervals length"],
    identifiers: Sequence[Hashable],
) -> Int64[Array, "unique_identifiers length"]:
    """Combine span masks from the same identifiers into a single mask.

    Each binary mask in `span_masks` has a corresponding identifier in `identifiers`.
    The function combines the masks for each unique identifier by summing the masks
    and converting the result back into a binary mask.

    This results in N ≤ len(identifiers) = len(set(identifiers) binary arrays.

    Example:
        > span_masks = np.array(
            [[1 1 1 1 0 0 0 0 0 0 0]
             [0 0 0 0 1 1 1 1 0 0 0]
             [0 0 0 0 0 0 0 1 1 1 0]])
        > identifiers = np.array([0, 1, 1])
        > combine_span_masks(span_masks, identifiers)
        array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]])

    """
    if not len(identifiers) == span_masks.shape[0]:
        raise ValueError(
            f"Expected identifiers to have the same number of elements as span_masks rows, "
            f"but got {len(identifiers)} and {span_masks.shape[0]}."
        )

    unique_identifiers: Integer[Array, " unique_identifiers"]
    idx_identifiers: Integer[Array, " unique_identifiers"]

    unique_identifiers, idx_identifiers = np.unique(identifiers, return_inverse=True)

    # Combine the span masks for each unique identifier.
    combined_span_masks: Int64[Array, "unique_identifiers length"] = np.zeros(
        (len(unique_identifiers), span_masks.shape[1]), dtype=int
    )
    np.add.at(combined_span_masks, idx_identifiers, span_masks)

    # Convert to binary mask.
    combined_span_masks = (combined_span_masks > 0).astype(np.int64)
    return combined_span_masks


def scatter_span_masks(
    span_masks: Integer[Array, "n_unique_identifiers length"],
    span_masks_identifiers: Sequence[Hashable],
    all_identifiers: Sequence[Hashable],
) -> Int64[Array, "all_identifiers length"]:
    """Scatter the span_masks into a binary matrix of N=len(all_identifiers) span masks.

    Complete an array of binary masks with assigned identifiers (where we likely have
    `span_masks.sum(axis=1)>0).all() == True`) with 0-only arrays for identifiers
    from `all_identifiers` that are missing from `span_masks_identifiers`.

    Arguments:
        span_masks: A binary matrix of shape (M, length) where M is the number of unique identifiers
            in `span_masks_identifiers`.
        span_masks_identifiers: A sequence of *unique* identifiers corresponding to the rows
            in `span_masks`.
        all_identifiers: A sequence of all possible *unique* identifiers.

    Returns:
        Int64[Array, "all_identifiers length"]: A binary matrix of shape (N, length) where
            N is the number of unique identifiers in `all_identifiers`.
    """
    if not span_masks.shape[0] == len(span_masks_identifiers):
        raise ValueError(
            f"Expected span_masks to have the same number of rows as identifiers, "
            f"but got {span_masks.shape[0]} and {len(span_masks_identifiers)}."
        )

    # Verify that the two sequences are composed of unique identifiers
    if not len(span_masks_identifiers) == len(set(span_masks_identifiers)):
        raise ValueError("span_masks_identifiers must be unique.")

    if not len(all_identifiers) == len(set(all_identifiers)):
        raise ValueError("all_identifiers must be unique.")

    # Verify that all span mask identifiers are in the all_identifiers list
    if not np.isin(span_masks_identifiers, all_identifiers).all():
        raise ValueError("All span_masks_identifiers must be in all_identifiers.")

    # Scatter the span masks to the full identifier space.
    scattered_span_masks: Int64[Array, "all_identifiers length"] = np.zeros(
        (len(all_identifiers), span_masks.shape[1]), dtype=int
    )

    scattered_span_masks[np.isin(all_identifiers, span_masks_identifiers), :] = span_masks

    return scattered_span_masks


class RelativeCoordinatesIntervals(pa.DataFrameModel):
    """Schema for named intervals with start-end positions relative to a pre-defined length."""

    name: pa.typing.String
    start: pa.typing.Int64
    end: pa.typing.Int64


def relative_coordinates_to_scattered_span_masks(
    relative_coordinates: DataFrame[RelativeCoordinatesIntervals],
    all_identifiers: Sequence[str],
    length: int,
    allow_empty: bool,
) -> Int64[Array, "all_identifiers length"]:
    """Process a dataframe of (start,end,identifier) coordinates into a binary matrix of (N, L) masks.

    Example of application: after intersecting small intervals with a query sequence interval,
    we want to produce a mask indicating the positions of these intervals within the query.

    Steps:
    - make the span_masks for all of the intervals in `relative_coordinates`
    - combine the span_masks for intervals with the same identifier
    - scatter the combined span_masks into a binary matrix of shape (N, L) where
      N is the number of unique identifiers and L is the length of the sequence.

    Arguments:
        relative_coordinates: A dataframe containing the relative coordinates information.
            The [start,end) coordinates should be relative to the `length`.
        all_identifiers: A list of all possible identifiers, which will determine
            the `L` dimension. This is used to create empty binary masks for identifiers that
            are not present in the dataframe.
        length: The length of the sequence over which the relative coordinates are defined.
        allow_empty: Whether to include empty binary masks for identifiers not present
            in the dataframe.

    Returns:
        A binary matrix of shape (N, L) where N is the number of identifiers and L is the
            length (`length`) of the sequence.
    """
    if relative_coordinates.empty:
        if not allow_empty:
            raise ValueError("relative_coordinates is empty, but allow_empty is set to False.")

        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)
        identifiers = np.array([], dtype=np.str_)

    else:
        starts = relative_coordinates["start"].values
        ends = relative_coordinates["end"].values
        identifiers = relative_coordinates["name"].values

    unexpected_coordinates_identifiers = set(identifiers) - set(all_identifiers)
    if len(unexpected_coordinates_identifiers) > 0:
        raise ValueError(
            f"Unexpected coordinates identifiers found: {unexpected_coordinates_identifiers}"
        )

    span_masks = intervals_to_span_masks(
        starts=starts,
        ends=ends,
        length=length,
    )

    combined_span_masks = combine_span_masks_on_identifiers(
        span_masks=span_masks,
        identifiers=identifiers,
    )
    scattered_span_masks = scatter_span_masks(
        span_masks=combined_span_masks,
        span_masks_identifiers=np.unique(identifiers),
        all_identifiers=all_identifiers,
    )

    return scattered_span_masks



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
        Mass allowed outside the HPD set (e.g. 0.05 ⇒ 95 % HPD).
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

    # 1. sort indices by descending probability
    idx_sorted = np.argsort(p)[::-1]
    cumsum = np.cumsum(p[idx_sorted])

    # 2. minimal set carrying ≥ 1‑alpha mass
    k = np.searchsorted(cumsum, 1 - alpha) + 1
    hpd_idx = idx_sorted[:k]

    # 3. boolean mask in original order
    mask = np.zeros_like(p, dtype=bool)
    mask[hpd_idx] = True
    return mask


# TODO: the operation of finding segments is not optimal, as it iterates over the mask.
# Also : can be externalized to a more general utility function.
# => Span masks has a function.
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
        Mass allowed outside the HPD set (e.g. 0.05 ⇒ 95 % HPD).
    atol : float, default 1e-8
        Absolute tolerance for checking if p sums to 1.

    Returns:
    -------
    segments : list of tuples
        List of (start, end) tuples for each segment in the HPD set.
    """
    mask = make_hpd_masks_from_probability_vector(p, alpha=alpha)
    segments = []
    in_run = False
    for i, flag in enumerate(mask):
        if flag and not in_run:
            start = i
            in_run = True

        elif not flag and in_run:  # run ends
            segments.append((start, i))
            in_run = False
    if in_run:
        segments.append((start, len(p)))

    return segments


