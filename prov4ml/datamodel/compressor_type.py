
import zarr
from enum import Enum

class CompressorType(Enum):
    NONE = None
    ZIP = "zip"
    BLOSC_ZSTD = "blosc_zstd"

COMPRESSORS_FOR_ZARR = [CompressorType.BLOSC_ZSTD]

def compressor_to_type(comp): 
    if comp == CompressorType.NONE: 
        return None
    elif comp == CompressorType.ZIP: 
        return "zip"
    elif comp == CompressorType.BLOSC_ZSTD: 
        return zarr.codecs.BloscCodec(cname='zstd')
    else: 
        raise AttributeError(f">compressor_to_type(): compressor {comp} not found")