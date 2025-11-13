
from typing import Any, Callable, Tuple
from codecarbon import EmissionsTracker

def carbon_tracked_function(f: Callable, *args, **kwargs) -> Tuple[Any, Any]:
    """
    Tracks carbon emissions for a given function call.
    
    Args:
        f (Callable): The function to be executed and carbon emissions tracked.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.
    
    Returns:
        Tuple[Any, Any]: A tuple containing the result of the function call and the total emissions tracked.
    """
    TRACKER.start()
    result = f(*args, **kwargs)
    _ = TRACKER.stop()
    total_emissions = TRACKER._prepare_emissions_data()
    return result, total_emissions

def _carbon_init() -> None:
    """Initializes the carbon emissions tracker."""
    global TRACKER
    TRACKER = EmissionsTracker(
        save_to_file=False,
        save_to_api=False,
        save_to_logger=False, 
        log_level="error",
    ) #carbon emission tracker, don't save anywhere, just get the emissions value to log with prov4ml
    TRACKER.start()

def stop_carbon_tracked_block() -> Any:
    """
    Stops the tracking of carbon emissions for a code block and returns the total emissions tracked.
    
    Returns:
        Any: The total emissions tracked.
    """
    #_ = TRACKER.stop()
    total_emissions = TRACKER._prepare_emissions_data()
    return total_emissions
