import logging
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def is_provided_parameter_a_valid_batch_id_path(
    parameter_value: str,
    first_batch_id: str,
) -> bool:
    """Check if a provided parameter value is a valid path with {BATCH_ID} field."""
    if "/" in parameter_value:
        # Check that BATCH_ID is an available field to format in the string.
        if "{BATCH_ID}" not in parameter_value:
            raise ValueError(
                "When using a path for params_evaluation_target_source, "
                f"it should contain a {{BATCH_ID}} field to be formatted. "
                f"Got {parameter_value}"
            )

        # Test if first batch exists.
        fp = parameter_value.format(BATCH_ID=first_batch_id)
        if not Path(fp).exists():
            raise FileNotFoundError(
                f"Provided path for parameter is not valid. "
                f"Tested with first batch id {first_batch_id}, got {fp} which does not exist."
            )

        return True

    return False
