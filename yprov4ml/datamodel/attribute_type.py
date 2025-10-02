
import sys
from enum import Enum

class LoggingItemKind(Enum): 
    """
    An enumeration representing different kinds of logging items used in provenance data collection.

    Attributes:
    -----------
    METRIC : str
        Represents a custom metric.
    FLOPS_PER_BATCH : str
        Represents the FLOPS (floating-point operations per second) calculated per batch.
    FLOPS_PER_EPOCH : str
        Represents the FLOPS calculated per epoch.
    SYSTEM_METRIC : str
        Represents system metrics related to hardware and system performance.
    CARBON_METRIC : str
        Represents carbon emission metrics.
    EXECUTION_TIME : str
        Represents the execution time of code segments.
    MODEL_VERSION : str
        Represents the version of the machine learning model.
    FINAL_MODEL_VERSION : str
        Represents the final version of the machine learning model.
    PARAMETER : str
        Represents parameters used in the experiment.

    Notes:
    ------
    - This enumeration helps categorize different types of logging items for better organization and management
      of provenance data.
    - Each item corresponds to a specific aspect of logging or metrics that might be tracked in an experiment.
    """
    METRIC = 'metric'
    FLOPS_PER_BATCH = 'flops_pb'
    FLOPS_PER_EPOCH = 'flops_pe'
    SYSTEM_METRIC = 'system'
    CARBON_METRIC = 'carbon'
    EXECUTION_TIME = 'execution_time'
    MODEL_VERSION = 'model_version'
    FINAL_MODEL_VERSION = 'model_version_final'
    PARAMETER = 'param'

def get_source_from_kind(kind: LoggingItemKind) -> str:
    """
    Returns the source string based on the logging item kind.

    Parameters:
    -----------
    kind : LoggingItemKind
        The type of logging item which determines the source.

    Returns:
    --------
    str
        The source string associated with the provided logging item kind.
    """
    if kind == LoggingItemKind.METRIC or kind == None:
        return 'custom_metric'
    elif kind == LoggingItemKind.FLOPS_PER_BATCH or kind == LoggingItemKind.FLOPS_PER_EPOCH:
        return 'fvcore.nn.FlopCountAnalysis'
    elif kind == LoggingItemKind.SYSTEM_METRIC:
        if sys.platform != 'darwin':
            return 'pyamdgpuinfo'
        else: 
            return "apple_gpu"            
    elif kind == LoggingItemKind.CARBON_METRIC:
        return 'codecarbon'
    elif kind == LoggingItemKind.EXECUTION_TIME:
        return 'std.time'
    elif kind == LoggingItemKind.MODEL_VERSION or kind == LoggingItemKind.FINAL_MODEL_VERSION:
        return 'torch'
    else:
        return ""

