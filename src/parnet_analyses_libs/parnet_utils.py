import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
from parnet.models import RBPNet

# Make a class which should be a dict that has a single key "sequences"
# This way I want to enforce that the input data always contains the "sequences" key

class ParnetInputDict(dict):
    """ """

    def __init__(self, sequences):
        super().__init__({"sequences": sequences})

    def __setitem__(self, key, value):
        if key != "sequences":
            raise KeyError("Only 'sequences' key is allowed")
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        if any(k != "sequences" for k in dict(*args, **kwargs).keys()):
            raise KeyError("Only 'sequences' key is allowed")
        return super().update(*args, **kwargs)

    def __delitem__(self, key):
        raise KeyError("'sequences' key cannot be deleted")

def parnet_predict(
    model: RBPNet,
    model_metadata: dict[str, Any],
    input_data: ParnetInputDict
) -> dict[str, torch.Tensor]:
    if model_metadata.version == "0.5.0":
        return model(input_data['sequences'])

    elif model_metadata.version != "0.5.0":
        return model(input_data)
    else:
        raise ValueError(f"Unsupported model version: {model_metadata.version}")


PARNET_MODEL_NAMES = Literal["parnet_new.11m-5.0", "parnet.21m-none", "parnet.21m-0.0", "parnet.21m-5.0", "parnet.7m-0.0", "parnet.7m-2.5", "parnet.7m-10.0", "parnet.7m-20.0", "parnet.7m-80.0"]

# TODO: generalize through the metadata model config file content.
def load_parnet_model(
    parnet_model_name: PARNET_MODEL_NAMES,
    filepath: os.PathLike,
    dtype: torch.dtype,
    device: torch.device,
) -> RBPNet:
    """Load a ParNet model from a given filepath."""
    if parnet_model_name == "parnet_new.11m-5.0":
        model = torch.load(filepath, map_location=device).to(dtype)
        model.head.use_maximum_target_control_logprob = True

    elif parnet_model_name in ["parnet.21m-none", "parnet.21m-0.0", "parnet.21m-5.0"]:
        model = torch.load(filepath, map_location=device).to(dtype)
        model.projection = lambda x: x
        model.head.use_maximum_target_control_logprob = False
        model.head.control_nograd = False

    elif parnet_model_name in [
        "parnet.7m-0.0",
        "parnet.7m-2.5",
        "parnet.7m-10.0",
        "parnet.7m-20.0",
        "parnet.7m-80.0",
    ]:
        model = torch.load(filepath, map_location=device).to(dtype)
        model.projection = lambda x: x
        model.head.use_maximum_target_control_logprob = True

    else:
        raise ValueError(f"Unknown model name: {parnet_model_name}")

    return model
