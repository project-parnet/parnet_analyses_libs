"""Scoring functions for performance evaluation.

NOTE: while scorers are defined with arguments x and y,
some scorers are not symmetric, i.e. the order of x and y matters.

By convention we consider that x is the "ground truth" and y the "predictions".
"""


import functools
import inspect
from collections.abc import Callable
from enum import Enum
from typing import ClassVar, Protocol

import more_itertools
import torch
import torchmetrics
from jaxtyping import Array, Float, Integer


class ScorerFunction3D(Protocol):
    """Protocol for scorer functions that take two 3D tensors and return a 2D tensor."""
    def __call__(
        self,
        preds: Float[Array, "batch tracks length"],
        target: Float[Array, "batch tracks length"] | Integer[Array, "batch tracks length"],
        **kwargs,  # noqa: ANN003
    ) -> Float[Array, "batch tracks"] | Integer[Array, "batch tracks"]:
        """Compute a score between two 3D tensors x and y, considering additional parameters."""
        ...

# Parametrized scorer: only takes as input the two tensors to compare.
ParametrizedScorerFunction3D = Callable[
    [
        Float[Array, "batch tracks length"],
        Float[Array, "batch tracks length"] | Integer[Array, "batch tracks length"],
    ], Float[Array, "batch tracks"] | Integer[Array, "batch tracks"]]


def standardize_args(target_mapping: dict[str, str]) -> Callable:
    """Decorator to standardize argument names for factory registration.

    Args:
        target_mapping: Dict mapping standard names to original names
                        e.g., {"preds": "x", "target": "y"}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            # If called with positional args, use as-is
            if args:
                return func(*args, **kwargs)

            # Map standardized kwargs to original names
            mapped_kwargs = {}
            for std_name, orig_name in target_mapping.items():
                if std_name in kwargs:
                    mapped_kwargs[orig_name] = kwargs.pop(std_name)

            # Keep any remaining kwargs
            mapped_kwargs.update(kwargs)
            return func(**mapped_kwargs)

        return wrapper
    return decorator



class Scorer3DFactory:
    """Factory class for 3D scorer functions.

    This enables configuration-based parametrization of scorer functions.
    Using this, the user can specify a scorer by its name and parameters in a config file.
    The factory will return a callable that only takes the two tensors to compare as input.
    """

    registry: ClassVar[dict[str, ScorerFunction3D]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register new mask functions."""

        def decorator(fn: Callable) -> Callable:
            cls.registry[name] = fn
            return fn

        return decorator

    @classmethod
    def register_with_standardized_arg_names(cls, name: str, arg_mapping: dict[str, str]) -> Callable:
        """Register a scorer function with standardized arguments names."""
        if "preds" not in arg_mapping or "target" not in arg_mapping:
            raise ValueError("arg_mapping must contain 'preds' and 'target' keys.")

        def decorator(fn: Callable) -> Callable:
            # Create standardized version for factory
            standardized_fn = standardize_args(arg_mapping)(fn)
            cls.registry[name] = standardized_fn

            # Return the unmodified original function to allow direct usage with original names.
            return fn
        return decorator



    @classmethod
    def list_scorers(cls) -> list[str]:
        """Return a list of registered scorer names."""
        return list(cls.registry.keys())

    @classmethod
    def get_scorer(cls, name: str, **kwargs: dict) -> ParametrizedScorerFunction3D:
        """Get a registered parametrized scorer by name, or None if not found."""
        scorer = cls.registry[name]
        return functools.partial(scorer, **kwargs)


def batched_metrics_3d(
    preds: Float[Array, "batch tracks length"],
    target: Float[Array, "batch tracks length"],
    fun_metrics_3d: ParametrizedScorerFunction3D,
    device: torch.device,
    dtype: torch.dtype,
    batch_device: torch.device,
    batch_size: int,
) -> Float[Array, "batch tracks"]:
    """Compute metrics over preds and target 3D tensors, split in batches on first dimension.

    Args:
        preds: Predictions Tensor of shape (Batch_size, Tasks, Length)
        target: Ground truths Tensor of shape (Batch_size, Tasks, Length)
        fun_metrics_3d: Function to compute metrics on a batch of preds and target.
                        It should take two 3D tensors and return a 2D tensor.
        device: Device to store the final metrics tensor.
        dtype: Data type to store the final metrics tensor.
        batch_device: Device to use for computation of each batch.
        batch_size: Size of each batch to process at a time.

    Returns:
        Tensor of shape (Batch_size, Tasks) containing the computed metrics.
    """
    assert preds.shape == target.shape
    assert preds.ndim == 3

    metrics = torch.empty((preds.shape[0], preds.shape[1]), device=device, dtype=dtype)
    batched_indices = more_itertools.chunked(range(preds.shape[0]), batch_size)

    for batch_indices in batched_indices:
        preds_batch = preds[batch_indices, :, :].to(device=batch_device)
        target_batch = target[batch_indices, :, :].to(device=batch_device)
        metric_sub = fun_metrics_3d(
            preds=preds_batch,
            target=target_batch
        ).to(dtype=dtype, device=device)
        metrics[batch_indices, :] = metric_sub

    return metrics


@Scorer3DFactory.register_with_standardized_arg_names("pearson_correlation", {
    "preds": "x",
    "target": "y"
})
def pearson_corr_3d(
    x: Float[Array, "batch tracks length"],
    y: Float[Array, "batch tracks length"],
) -> Float[Array, "batch tracks"]:
    """Pearson Correlation Coefficient over dim=-1 length of each element in x and y."""
    # x, y: (B, T, L)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)

    num = (x * y).sum(dim=-1)
    denom = torch.sqrt((x.pow(2).sum(dim=-1)) * (y.pow(2).sum(dim=-1)))
    return num / denom  # shape (B, T)


@Scorer3DFactory.register_with_standardized_arg_names("spearman_correlation", {
    "preds": "x",
    "target": "y"
})
def spearman_corr_3d(
    x: Float[Array, "batch tracks length"],
    y: Float[Array, "batch tracks length"],
) -> Float[Array, "batch tracks"]:
    """Spearman Rank Correlation Coefficient over dim=-1 length of each element in x and y."""
    dtype = torch.float32
    # x, y: (B, T, L)
    # 1) rank along last dimension
    x_rank = x.argsort(dim=-1).argsort(dim=-1).to(dtype)
    y_rank = y.argsort(dim=-1).argsort(dim=-1).to(dtype)

    # 2) compute pearson correlation along last axis
    x_rank = x_rank - x_rank.mean(dim=-1, keepdim=True)
    y_rank = y_rank - y_rank.mean(dim=-1, keepdim=True)

    num = (x_rank * y_rank).sum(dim=-1)
    denom = torch.sqrt((x_rank.pow(2).sum(dim=-1)) * (y_rank.pow(2).sum(dim=-1)))
    return num / denom  # shape (B, T)


def cosine_sim_3d(
    x: Float[Array, "batch tracks length"],
    y: Float[Array, "batch tracks length"],
) -> Float[Array, "batch tracks"]:
    """Cosine Similarity over dim=-1 length of each element in x and y."""
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)


#TODO: should the normalize operation happen outside?
# Also : unsure if necessary.
def kldiv_3d(
    p: Integer[Array, "batch tracks length"],
    q: Float[Array, "batch tracks length"],
    eps: float = 1e-8,
) -> Float[Array, "batch tracks"]:
    """Kullback-Leibler Divergence over dim=-1 length of each element in p and q."""
    # Normalize counts -> empirical distribution
    #TODO: would thi rather be done outside?
    p = p / (p.sum(dim=-1, keepdim=True) + eps)

    # Clamp to avoid log(0).
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    kld = (p * (p.log() - q.log())).sum(dim=-1)
    return kld


#TODO: verify
def crossentropy_3d(
    p: Integer[Array, "batch tracks length"],
    q: Float[Array, "batch tracks length"],
    eps: float = 1e-8,
) -> Float[Array, "batch tracks"]:
    """Cross Entropy Loss over dim=-1 length of each element in x and y."""
    # Normalize counts -> empirical probabilities
    p = p / (p.sum(dim=-1, keepdim=True) + eps)

    # Clamp to avoid log(0).
    q = q.clamp(min=eps)

    ce = -(p * q.log()).sum(dim=-1)
    return ce


#TODO: verify
def multinomial_log_likelihood_3d(
    x: Float[Array, "batch tracks length"],
    y: Float[Array, "batch tracks length"],
    eps: float = 1e-8,
) -> Float[Array, "batch tracks"]:
    """Multinomial Log Likelihood over dim=-1 length of each element in x and y."""
    # clamp to avoid log(0)
    q = y.clamp(min=eps)
    loglik = (x * q.log()).sum(dim=-1)
    return loglik



@Scorer3DFactory.register("binary_auroc")
def binary_auroc_3d(
    preds: Float[Array, "batch tracks length"],
    target: Integer[Array, "batch tracks length"],
    eps: float = 1e-8,
) -> Float[Array, "batch tracks"]:
    """Manual implementation of AUROC computation using vectorized operations.

    The AUROC is computed over the last dimension of target and preds, i.e. pairing the 1D
    (preds, target) values each of shape = (length,).

    Args:
        preds: predictions Tensor of shape (Batch_size, Tasks, Length)
        target: ground truths Tensor of shape (Batch_size, Tasks, Length) with binary values
        eps: Small epsilon to handle numerical stability

    Returns:
        Tensor of shape (batch, tasks) containing AUROC values
    """
    # Verify that the target is binary
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("Target tensor must be binary (0 or 1).")

    # target, preds: (B, T, L)
    batch_size, tasks, length = target.shape
    device = target.device

    # Sort preds and corresponding ground truths
    sorted_indices = torch.argsort(preds, dim=2, descending=True)
    sorted_gt = torch.gather(target, dim=2, index=sorted_indices)

    # Count positive and negative samples for each (batch, task) pair
    pos_counts = target.sum(dim=2)  # (Batch_size, Tasks)
    neg_counts = length - pos_counts       # (Batch_size, Tasks)

    # Handle edge cases where all samples are of one class
    valid_mask = (pos_counts > 0) & (neg_counts > 0)

    # Compute cumulative sum of true positives
    tp_cumsum = torch.cumsum(sorted_gt, dim=2)

    # Compute cumulative sum of false positives
    fp_cumsum = (
        torch.arange(1, length + 1, device=device)
        .expand(batch_size, tasks, -1)
        - tp_cumsum
    )

    # Compute TPR and FPR
    tpr = tp_cumsum.float() / (pos_counts.unsqueeze(2) + eps)
    fpr = fp_cumsum.float() / (neg_counts.unsqueeze(2) + eps)

    # Add (0,0) point at the beginning
    tpr_padded = torch.cat(
        [
            torch.zeros(batch_size, tasks, 1, device=device),
            tpr
        ],
        dim=2
    )
    fpr_padded = torch.cat(
        [
            torch.zeros(batch_size, tasks, 1, device=device),
            fpr
        ],
        dim=2
    )

    # Compute AUC using trapezoidal rule
    fpr_diff = fpr_padded[:, :, 1:] - fpr_padded[:, :, :-1]
    tpr_avg = (tpr_padded[:, :, 1:] + tpr_padded[:, :, :-1]) / 2
    auc = torch.sum(fpr_diff * tpr_avg, dim=2)

    # Set AUC to 0.5 for invalid cases (all same class)
    auc = torch.where(valid_mask, auc, torch.tensor(0.5, device=device))
    return auc


class AuPrcEstimator(str, Enum):
    """Enum for AUPRC estimation methods.

    ROC curve is smooth and monotonic ⇒ trapezoidal rule works fine.
    PR curve is jagged, with vertical jumps ⇒ trapezoidal integration overestimates performance.

    Hence the Average Precision (AP) metric is favored over trapezoidal AUC for PR curves.

    Scikit-learn's `auc function uses the trapezoidal rule.

    But in torch:
    - from torchmetrics.functional import average_precision
    - from torcheval.metrics.functional import binary_auprc

    Both implement the "average precision" (AP) metric, not trapezoidal AUC.
    """
    TRAPEZOIDAL = "trapezoidal"
    AVERAGE_PRECISION = "average_precision"


@Scorer3DFactory.register("binary_auprc")
def binary_auprc_3d(
    preds: Float[Array, "batch tracks length"],
    target: Integer[Array, "batch tracks length"],
    auprc_estimator: AuPrcEstimator = AuPrcEstimator.AVERAGE_PRECISION,
    eps: float = 1e-8,
) -> Float[Array, "batch tracks"]:
    """Manual implementation of AUPRC (Area Under Precision-Recall Curve).

    NOTE: the implementation is matching the `torchmetrics.functional.average_precision`
    at a precision of 1e-3.

    Args:
        preds: Tensor of shape (Batch_size, Tasks, Length)
        target: Tensor of shape (Batch_size, Tasks, Length) with binary values
        auprc_estimator: Method to estimate AUPRC, either TRAPEZOIDAL or AVERAGE_PRECISION.
                        Consider using AVERAGE_PRECISION for a more conservative estimate.
        eps: Small epsilon for numerical stability

    Returns:
        Tensor of shape (Batch_size, Tasks) containing AUPRC values
    """
    batch_size, tasks, length = preds.shape
    device = preds.device

    # Verify that the target is binary
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("Target tensor must be binary (0 or 1).")

    # Sort preds in descending order and get corresponding ground truths
    sorted_indices = torch.argsort(preds, dim=2, descending=True)
    sorted_gt = torch.gather(target, dim=2, index=sorted_indices)

    # Count total positive samples for each (batch, task) pair
    pos_counts = target.sum(dim=2)  # (Batch_size, Tasks)

    # Handle edge cases where there are no positive samples
    valid_mask = pos_counts > 0

    # Compute cumulative true positives and false positives
    tp_cumsum = torch.cumsum(sorted_gt, dim=2)  # (Batch_size, Tasks, Length)

    # Compute precision and recall
    # Precision = TP / (TP + FP) = TP / total_preds_so_far
    total_preds = torch.arange(1, length + 1, device=device).expand(batch_size, tasks, -1)
    precision = tp_cumsum.float() / (total_preds + eps)

    # Recall = TP / total_positives
    recall = tp_cumsum.float() / (pos_counts.unsqueeze(2) + eps)

    ## Add starting point (0, 1) for precision-recall curve
    ## At threshold=inf, we predict nothing: precision=1 (undefined, but we use 1), recall=0
    precision_padded = torch.cat([torch.ones(batch_size, tasks, 1, device=device), precision], dim=2)
    recall_padded = torch.cat([torch.zeros(batch_size, tasks, 1, device=device), recall], dim=2)

    # Compute AUPRC using trapezoidal rule
    # Note: We integrate over recall (x-axis), so we need recall differences
    recall_diff = recall_padded[:, :, 1:] - recall_padded[:, :, :-1]

    if auprc_estimator == AuPrcEstimator.TRAPEZOIDAL:
        # The trapezoidal rule: sum of (width * average_height)
        precision_avg = (precision_padded[:, :, 1:] + precision_padded[:, :, :-1]) / 2
        auprc = torch.sum(recall_diff * precision_avg, dim=2)

    elif auprc_estimator == AuPrcEstimator.AVERAGE_PRECISION:
        # Average Precision (step-wise):
        # NOTE: start from precision at recall > 0, hence precision_padded[:, :, 1:]
        # Less conservative than using the start of each recall interval,
        # i.e. with precision_padded[:, :, :-1]
        # But: this matches better the values from `torchmetrics.functional.average_precision`
        precision_current = precision_padded[:, :, 1:]
        auprc = torch.sum(recall_diff * precision_current, dim=2)

    else:
        raise ValueError(f"Unknown AUPRC estimator: {auprc_estimator}")

    # For cases with no positive samples, return 0
    auprc = torch.where(valid_mask, auprc, torch.tensor(0.0, device=device))

    return auprc
