"""Masks for tasks: provided a vector (Batch tasks Length), create boolean masks to drop specific tasks.

Example of application:

- when comparing the predictions of a model against the ground truth, some
    ground truth tasks may not have any signal (e.g. 0 read counts across the entire length.)
- while it is simpler to calculate a metric across all tasks and elements in the batch,
    we want to ignore the metric results for such cases.
- This module provides a way to create boolean masks to ignore such tasks.
- Then downstream, when calculating the aggregate of the metric across tasks or elements,
    one can simply use the mask to filter out the unwanted tasks/elements.

"""

import operator
from collections.abc import Callable
from functools import reduce
from typing import ClassVar, TypeAlias

import torch
from jaxtyping import Array, Bool, Float

OPS = {
    "and": lambda args: reduce(operator.and_, args),
    "or": lambda args: reduce(operator.or_, args),
    "not": lambda args: operator.not_(args[0]),
}

LogicTree: TypeAlias = str | dict[str, "LogicTree"]


def eval_logic(
    tree: LogicTree,
    masks: dict[str, Bool[Array, "batch tasks"]],
) -> Bool[Array, "batch tasks"]:
    """Recursively evaluate the logical expression tree against the provided masks.

    The function receives a tree with internal nodes as strings representing boolean
    operations and leaf nodes representing mask names, provided in the `masks` argument.

    Leaf nodes are stored as a list of strings.

    Example of structure:

        '''
        # Example of a simple tree structure:
        # {
        #     "and": [
        #         "mask1",
        #         {
        #             "or": ["mask2", "mask3"]
        #         }
        #     ]
        # }
        #
        # Which can be represented in a YAML file as:
        # ---
        # and:
        #   - mask1
        #   - or:
        #       - mask2
        #       - mask3
        '''

    Arguments:
    ----------
    tree: The logical expression tree to evaluate.
    masks: A dictionary mapping mask names to their boolean values.

    Return:
    -------
    A boolean array representing the result of the logical expression evaluation.

    """
    if isinstance(tree, str):
        # leaf = mask name
        if tree not in masks:
            raise ValueError(f"Unknown mask: {tree}")

        return masks[tree]

    if isinstance(tree, dict):
        for op, subtrees in tree.items():
            if op not in OPS:
                raise ValueError(f"Unknown operator: {op}")

            if not isinstance(subtrees, list):
                subtrees = [subtrees]

            # Recursively evaluate each subtree
            vals = [eval_logic(t, masks) for t in subtrees]
            return OPS[op](vals)

    raise ValueError(f"Unexpected tree node: {tree}")


class MaskFactory:
    """MaskFactory: A factory for creating masks."""

    registry: ClassVar[dict[str, Callable]] = {}
    composed_registry: ClassVar[dict[str, Callable]] = {}

    @classmethod
    def register(cls, name: str, composed: bool = False) -> Callable:
        """Decorator to register new mask functions.

        The decorator should be applied to any function that is to be used to create a mask.

        Arguments:
            name: str
                The name of the mask function to register.

            composed: bool, default False
                If True, the function is registered as a composed mask function.
                Composed mask functions take as input a dictionary of existing masks and
                produce a new mask based on them.
                So such a mask function can exploit the existing masks to create more complex
                masking behavior.

        Returns:
            Callable: The registered mask function.
        """

        def decorator(fn: Callable) -> Callable:
            if composed:
                cls.composed_registry[name] = fn
            else:
                cls.registry[name] = fn
            return fn

        return decorator

    def __init__(
        self,
        config: dict,
        precomputed_masks: dict[str, Bool[Array, "batch tasks"]] | None = None,
    ):
        self.config = config
        self.precomputed_masks = precomputed_masks if precomputed_masks is not None else {}

    def generate_masks(
        self, tensor: Float[Array, "batch tasks length"]
    ) -> dict[str, Float[Array, "batch tasks"]]:
        """Apply the self.config to the input tensor: (Batch, Tasks, Length).

        Returns: dict[str, mask] with mask: (Batch, Tasks).
        """
        assert tensor.ndim == 3, "Input tensor must be 3D (Batch, Tasks, Length)"

        # Map identifiers of masks to boolean tensors.
        masks: dict[str, Bool[Array, "batch tasks"]] = {}
        # Update with precomputed masks.
        masks.update(self.precomputed_masks)

        # Iterate over config and apply masks.
        for mask_identifier, mask_config in self.config.items():
            if mask_config["mask_operation"] in self.registry:
                masks[mask_identifier] = self.registry[mask_config["mask_operation"]](
                    tensor, **mask_config["params"]
                )
            elif mask_config["mask_operation"] in self.composed_registry:
                masks[mask_identifier] = self.composed_registry[mask_config["mask_operation"]](
                    masks=masks, **mask_config["params"]
                )
            else:
                raise ValueError(
                    f"Unknown mask operation {mask_config['mask_operation']} for mask: {mask_identifier}"
                )

        return masks

    def mask(
        self, tensor: Float[Array, "batch tasks length"]
    ) -> Bool[Array, "batch tasks"]:
        batch, tasks, _ = tensor.shape
        final_mask = torch.ones((batch, tasks), dtype=torch.bool)

        masks = self.generate_masks(tensor)
        # Combine all masks
        for mask in masks.values():
            final_mask &= mask

        return final_mask


@MaskFactory.register("mask_minimum_coverage_of_window", composed=False)
def mask_minimum_coverage_of_window(
    signal: Float[Array, "batch tasks length"],
    min_proportion_covered: float,
    min_value_at_position: int | float,
) -> Bool[Array, "batch tasks"]:
    """Mask: is the proportion of non-zero positions across length of the window above threshold."""
    assert 0 <= min_proportion_covered <= 1, (
        "min_proportion_nonzero_in_window must be between 0 and 1"
    )
    mask_coverage = (signal > min_value_at_position).sum(dim=-1) / signal.shape[
        -1
    ] >= min_proportion_covered
    return mask_coverage


@MaskFactory.register("mask_minimum_total_sum_across_window", composed=False)
def mask_minimum_total_sum_across_window(
    signal: Float[Array, "batch tasks length"],
    min_total_in_window: int | float,
) -> Bool[Array, "batch tasks"]:
    """Mask: is total value across length of the window above threshold."""
    mask_sum_gt_threshold = signal.sum(dim=-1) >= min_total_in_window
    return mask_sum_gt_threshold


@MaskFactory.register("mask_minimum_value_in_window_at_any_position", composed=False)
def mask_minimum_value_in_window_at_any_position(
    signal: Float[Array, "batch tasks length"],
    min_value_at_position: int | float,
) -> Bool[Array, "batch tasks"]:
    """Mask: is there any position in the window above threshold."""
    mask_any_pos_gt_threshold = (signal >= min_value_at_position).any(dim=-1)
    return mask_any_pos_gt_threshold


### Composed masks


@MaskFactory.register("mask_minimum_count_of_passing_tasks", composed=True)
def mask_minimum_count_of_passing_tasks(
    masks: dict[str, Float[Array, "batch tasks"]],
    name_mask_to_use: str,
    min_count_passing_tasks: int,
) -> Bool[Array, "batch tasks"]:
    """ """
    if name_mask_to_use not in masks:
        raise ValueError(f"Mask {name_mask_to_use} not found in provided masks.")

    mask = masks[name_mask_to_use]
    # Here: from the (B, T) mask we make a (B, T) mask that is True if ANY TASK passes the threshold.
    # (B, T) -> (B,)
    mask_any_task_pass_sum_threshold = mask.sum(dim=-1) >= min_count_passing_tasks
    # Expand back to (B, T)
    mask_any_task_pass_sum_threshold = mask_any_task_pass_sum_threshold.unsqueeze(-1).expand_as(
        mask
    )
    # Reapply the selected mask to retain only tasks that pass the original mask.
    mask_any_task_pass_sum_threshold = mask_any_task_pass_sum_threshold & mask
    return mask_any_task_pass_sum_threshold


@MaskFactory.register("mask_from_precomputed_mask", composed=True)
def mask_from_precomputed_mask(
    masks: dict[str, Float[Array, "batch tasks"]],
    name_mask_to_use: str,
) -> Bool[Array, "batch tasks"]:
    """Hack function to apply a pre-defined masked **that was provided externally**.

    "Primary" masks are ran on-the-fly on the data, and "secondary" masks are applied
    on the resulting masks. But the MaskFactory can also be provided with precomputed
    masks.
    This function allows to apply these precomputed masks.
    """
    if name_mask_to_use not in masks:
        raise ValueError(f"Mask {name_mask_to_use} not found in provided masks.")

    mask = masks[name_mask_to_use]
    return mask
