
from enum import Enum

class MetricsType(Enum):
    """Enumeration for different types of metrics storage formats.

    Attributes:
        TXT (str): Represents text file format.
        ZARR (str): Represents Zarr file format.
        NETCDF (str): Represents NetCDF file format.
    """
    CSV = "csv"
    ZARR = 'zarr'
    NETCDF = 'netcdf'

def get_file_type(metrics_file_type): 
    if metrics_file_type == MetricsType.CSV:
        return ".csv"
    elif metrics_file_type == MetricsType.NETCDF: 
        return ".nc"
    elif metrics_file_type == MetricsType.ZARR: 
        return ".zarr"
    else: 
        raise NotImplementedError() 