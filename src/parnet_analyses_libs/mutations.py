"""Module for comparing sequence embeddings or profiles using slicing and scoring methods."""

from typing import Literal

import torch
from pydantic import BaseModel, model_validator
from pylbsr.bio.coordinates import SliceConfig, SliceCoordinateSystem
from torchmetrics.functional.regression import jensen_shannon_divergence
from typing_extensions import Self


class SequenceEmbeddingComparatorParameters(BaseModel):
    """Parameters for SequenceEmbeddingComparator."""

    scoring_method: Literal["cosine_similarity", "cosine_distance"]
    aggregate_method: Literal["mean", "max", "min"] | None
    scoring_sequence_context: Literal["window_based", "full_sequence"]
    slice_config: SliceConfig

    @model_validator(mode="after")
    def context_versus_slice_config(self) -> Self:
        """Validate that scoring_sequence_context and slice_config are compatible."""
        if (
            self.scoring_sequence_context == "full_sequence"
            and self.slice_config.mode == SliceCoordinateSystem.ABSOLUTE
        ):
            raise ValueError(
                "For full_sequence scoring_sequence_context, slice_config.mode must be ABSOLUTE"
            )
        return self


class SequenceEmbeddingComparator:
    """Class to compare two sequence embeddings using slicing and scoring methods."""

    def __init__(
        self,
        slice_config: SliceConfig,
        pool_slice_method: Literal["mean", "max", "min"] | None,
        scoring_method: Literal["cosine_similarity", "cosine_distance"],
        aggregate_scores_method: Literal["mean", "max", "min"] | None,
    ) -> Self:
        """Initialize the comparator with slicing and scoring configurations."""
        self.slice_config = slice_config
        self.pool_slice_method = pool_slice_method
        self.scoring_method = scoring_method
        self.aggregate_scores_method = aggregate_scores_method

    def _score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.scoring_method == "cosine_similarity":
            return torch.cosine_similarity(x, y, dim=0)

        elif self.scoring_method == "cosine_distance":
            return 1.0 - torch.cosine_similarity(x, y, dim=0)

        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")

    def _slice(self, tensor: torch.Tensor) -> torch.Tensor:
        L = tensor.shape[-1]
        slc = self.slice_config.to_slice(L)
        tensor = tensor[:, slc]
        return tensor

    def _aggregate(
        self,
        tensor: torch.Tensor,
        aggregate_method: Literal["mean", "max", "min"] | None = None,
    ) -> torch.Tensor:
        if aggregate_method is None:
            return tensor
        elif aggregate_method == "mean":
            tensor = torch.mean(tensor, dim=-1, keepdim=True)
        elif aggregate_method == "max":
            tensor = torch.max(tensor, dim=-1, keepdim=True).values
        elif aggregate_method == "min":
            tensor = torch.min(tensor, dim=-1, keepdim=True).values
        else:
            raise ValueError(f"Unknown aggregation function: {aggregate_method}")
        return tensor

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compare two sequence embeddings and return the computed scores."""
        assert x.shape == y.shape, (x.shape, y.shape)
        x = self._slice(x)
        y = self._slice(y)

        x = self._aggregate(x, self.pool_slice_method)
        y = self._aggregate(y, self.pool_slice_method)

        scores = self._score(x, y)
        scores = self._aggregate(scores, self.aggregate_scores_method)
        return scores


class SequenceProfilesComparator:
    """Class to compare two groups of profiles using slicing and scoring methods."""

    def __init__(
        self,
        slice_config: SliceConfig,
        scoring_method: Literal["jensen_shannon_divergence", "abs_delta_p", "delta_p"],
        aggregate_across_length_method: Literal["mean", "max", "min"] | None,
        aggregate_across_tasks_method: Literal["mean", "max", "min"] | None,
    ):
        self.slice_config = slice_config
        self.scoring_method = scoring_method
        self.aggregate_across_length_method = aggregate_across_length_method
        self.aggregate_across_tasks_method = aggregate_across_tasks_method

    def _score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.scoring_method == "jensen_shannon_divergence":
            assert x.shape == y.shape, (x.shape, y.shape)
            assert x.ndim == 2, x.shape  # (Tasks, Length)

            scores = jensen_shannon_divergence(x, y, reduction=None)

        elif self.scoring_method == "abs_delta_p":
            assert x.shape == y.shape, (x.shape, y.shape)
            assert x.ndim == 2, x.shape  # (Tasks, Length)

            scores = torch.abs(x - y)

        elif self.scoring_method == "delta_p":
            assert x.shape == y.shape, (x.shape, y.shape)
            assert x.ndim == 2, x.shape  # (Tasks, Length)

            scores = x - y

        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")

        return scores

    def _slice(self, tensor: torch.Tensor) -> torch.Tensor:
        L = tensor.shape[-1]
        slc = self.slice_config.to_slice(L)
        tensor = tensor[:, slc]
        return tensor

    def _aggregate(
        self,
        tensor: torch.Tensor,
        keep_dim: bool,
        aggregate_method: Literal["mean", "max", "min"] | None = None,
    ) -> torch.Tensor:
        #
        # NOTE: IN THIS SPECIFIC SITUATION (PARNET multi-task profile scoring)
        # it might make sense to return TWO values:
        # - the retrieved value
        # - and the index (of the task) corresponding to the value.
        #
        # Since this could also be done downstream, it might make more sense
        # to NOT PERFORM this aggregation here.

        if aggregate_method is None:
            return tensor

        elif aggregate_method == "mean":
            tensor = torch.mean(tensor, dim=-1, keepdim=keep_dim)
        elif aggregate_method == "max":
            tensor = torch.max(tensor, dim=-1, keepdim=keep_dim).values
        elif aggregate_method == "min":
            tensor = torch.min(tensor, dim=-1, keepdim=keep_dim).values
        else:
            raise ValueError(f"Unknown aggregation function: {aggregate_method}")
        return tensor

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, (x.shape, y.shape)
        x = self._slice(x)
        y = self._slice(y)

        scores = self._score(x=x, y=y)
        if self.aggregate_across_length_method is not None:
            # Here: verify that we still have a "length" dimension to aggregate over.
            assert scores.ndim == 2, scores.shape  # (Tasks, Length)
            scores = self._aggregate(
                tensor=scores,
                aggregate_method=self.aggregate_across_length_method,
                keep_dim=False,
            )

        scores = self._aggregate(
            tensor=scores,
            aggregate_method=self.aggregate_across_tasks_method,
            keep_dim=True,
        )
        return scores
