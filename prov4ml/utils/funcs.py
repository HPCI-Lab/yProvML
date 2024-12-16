
import os
from typing import Optional
import time

def prov4ml_experiment_matches(
        experiment_name : str,
        exp_folder : str
    ) -> bool:
    """
    Check if the experiment name matches the experiment name in the provenance data.

    Parameters:
    -----------
    experiment_name : str
        The name of the experiment.
    exp_folder : str
        The name of the experiment folder.

    Returns:
    --------
    bool
        True if the experiment name matches the experiment name in the provenance data, False otherwise
    """
    exp_folder = "_".join(exp_folder.split("_")[:-1])
    return experiment_name == exp_folder

def get_current_time_millis() -> int:
    """
    Get the current time in milliseconds.

    Returns:
    --------
    int
        The current time in milliseconds.
    """
    return int(round(time.time() * 1000))

def get_global_rank() -> Optional[int]:
    """
    Retrieves the global rank of the current process in a distributed computing environment.

    This function checks if a distributed computing environment is available and initialized, and
    returns the global rank of the current process accordingly. The function supports two types of
    distributed environments:

    2. **SLURM**: If the environment variable `SLURM_PROCID` is present, it assumes that the process is
       managed by SLURM and retrieves the local rank from this environment variable.

    If neither distributed environment is detected, the function returns `None`.

    Returns:
    --------
    Optional[int]
        The global rank of the process if a distributed environment is detected, otherwise `None`.

    Examples:
    ---------
    >>> get_global_rank()
    0
    >>> get_global_rank()
    None
    """    
    # if on slurm, return the local rank
    if "SLURM_PROCID" in os.environ:
        return int(os.getenv("SLURM_PROCID", None))
    
    return 0


def get_runtime_type(): 
    """
    Get the runtime type.

    Returns:
    --------
    str
        The runtime type.

    Examples:
    ---------
    >>> get_runtime_type()
    "single_core"
    """
    return "single_core"