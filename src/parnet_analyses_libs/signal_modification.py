"""Functions to modify signal representations, including masking and interpolation.

Ideas:

- Crop a vector to only consider central positions.
- Mask positions basing on a filter or a pre-computed mask (e.g. positions lower than a threshold.)


Main entities:

- SignalModifierFactory:
    - This has to be instantiated provided a configuration dictionary and precomputed masks.
    - The factory has a registry of available modification functions.
    - New modification functions can be easily registered with the appropriate decorators.
    - A signal modifier function should return both the (modified) tensor and a mask.
    - The mask will be used downstream e.g. by a scoring function to only consider relevant positions.
    - Call the `.apply(x)` method to apply the modification to a tensor x.
    - This method is expected to return a TensorAndMask object.

Todo:
- Do I also need a logic handler? Or can we simply stack `mod3(mod2(mod1(x)))`?
- Is it OK to keep together the tensor and the mask, or should I simplify?

"""

import collections
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Array, Bool, Float, Integer


@dataclass
class TensorAndMask:
    tensor: Float[Array, " length"]
    mask: Bool[Array, " length"]



# Define a protocol for the signal modifier functions.
class SignalModifierFunction(Protocol):
    """Protocol for signal modifier functions."""

    def __call__(self, x: Float[Array, " length"], **kwargs: object) -> TensorAndMask: ...


# Create a decorator function that will be used to decorate functions that merely
# modify the content of a tensor without producing any mask, by wrapping its returned
# tensor into a TensorAndMask object with a mask of all True values.
def complete_signal_modifier_function_with_mask(
    fn: Callable[[Float[Array, " length"]], Float[Array, " length"]]
) -> SignalModifierFunction:
    """Convert a function that modifies a tensor into one that returns a TensorAndMask."""

    def wrapper(x: Float[Array, " length"], **kwargs: object) -> TensorAndMask:
        modified_tensor = fn(x)
        mask = torch.ones_like(modified_tensor, dtype=torch.bool)
        return TensorAndMask(tensor=modified_tensor, mask=mask)

    return wrapper



class SignalModifierFactory:


    registry: ClassVar[dict[str, Callable]] = {}

    @classmethod
    def register(cls, name):
        """Decorator to register new mask functions."""

        def decorator(fn):
            cls.registry[name] = fn
            return fn

        return decorator

    @classmethod
    def list_modifiers(cls) -> list[str]:
        return list(cls.registry.keys())

    @classmethod
    def get_modifier(cls, name: str):
        return cls.registry.get(name)
