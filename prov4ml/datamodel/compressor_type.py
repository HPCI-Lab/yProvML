

from enum import Enum

def compressor_to_type(comp): 
    if comp == CompressorType.NONE: 
        return None

class CompressorType(Enum):
    NONE = None
