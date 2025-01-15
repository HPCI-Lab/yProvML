from enum import Enum

class MetricsType(Enum):
    """Enumeration for different types of metrics storage formats.

    Attributes:
        TXT (str): Represents text file format.
        ZARR (str): Represents Zarr file format.
    """
    TXT = 'txt'
    ZARR = 'zarr'
