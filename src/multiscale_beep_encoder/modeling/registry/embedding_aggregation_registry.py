from typing import Callable, Dict

import torch

# This is the registry of the base aggregation functions from torch
_AGGREGATIONS: Dict[str, Callable[..., torch.Tensor]] = {
    "mean": torch.mean,
    "nop": lambda input_tensor: input_tensor,
    "median": torch.median,
}


def get_registry(dim: int) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Returns a registry of aggregation functions with application on the dim dimension.

    Each function in the returned registry will take a single torch.Tensor as input,
    perform its aggregation along the specified 'dim', and return a torch.Tensor
    with the specified 'dtype'.

    Args:
        dim: The dimension along which to perform the aggregation.

    Returns:
        A dictionary where keys are aggregation names (e.g., "mean", "median")
        and values are the curried aggregation functions.
    """
    curried_registry: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = _AGGREGATIONS.copy()

    if "mean" in _AGGREGATIONS:
        curried_registry["mean"] = lambda tensor_input: torch.mean(tensor_input, dim=dim)
    if "median" in _AGGREGATIONS:
        curried_registry["median"] = lambda tensor_input: torch.median(
            input=tensor_input, dim=dim
        ).values

    return curried_registry
