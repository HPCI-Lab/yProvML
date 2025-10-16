from fvcore.nn import FlopCountAnalysis
from typing import Any

def _init_flops_counters() -> None:
    """Initializes the global FLOPs counters."""
    global FLOPS_PER_BATCH_COUNTER
    global FLOPS_PER_EPOCH_COUNTER
    FLOPS_PER_BATCH_COUNTER = 0
    FLOPS_PER_EPOCH_COUNTER = 0

def get_flops_per_epoch(model: Any, dataset: Any) -> int:
    """
    Calculates and returns the total FLOPs per epoch for the given model and dataset.

    Args:
        model (Any): The model for which FLOPs per epoch are to be calculated.
        dataset (Any): The dataset used for training the model.

    Returns:
        int: The total FLOPs per epoch.
    """
    global FLOPS_PER_EPOCH_COUNTER

    x, _ = dataset[0]
    flops = FlopCountAnalysis(model, x)
    total_flops = flops.total() * len(dataset)
    FLOPS_PER_EPOCH_COUNTER += total_flops
    return FLOPS_PER_EPOCH_COUNTER

def get_flops_per_batch(model: Any, batch: Any) -> int:
    """
    Calculates and returns the total FLOPs per batch for the given model and batch of data.

    Args:
        model (Any): The model for which FLOPs per batch are to be calculated.
        batch (Any): A batch of data used for inference with the model.

    Returns:
        int: The total FLOPs per batch.
    """
    global FLOPS_PER_BATCH_COUNTER
    x, _ = batch
    flops = FlopCountAnalysis(model, x)
    FLOPS_PER_BATCH_COUNTER += flops.total()
    return FLOPS_PER_BATCH_COUNTER