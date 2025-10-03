
import zarr
import numcodecs
from enum import Enum

class CompressorType(Enum):
    NONE = None
    ZIP = "zip"
    BLOSC_ZSTD = "blosc_zstd"
    LZ4 = "l4"

COMPRESSORS_FOR_ZARR = [CompressorType.BLOSC_ZSTD, CompressorType.LZ4]

def compressor_to_type(comp): 
    if comp == CompressorType.NONE: 
        return None
    elif comp == CompressorType.ZIP: 
        return "zip"
    elif comp == CompressorType.BLOSC_ZSTD: 
        return zarr.codecs.BloscCodec(cname='zstd')
    elif comp == CompressorType.LZ4: 
        return numcodecs.lz4.LZ4()
    else: 
        raise AttributeError(f">compressor_to_type(): compressor {comp} not found")