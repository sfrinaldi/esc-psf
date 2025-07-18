import importlib.metadata
import os
from pathlib import Path
from .math_module import  np_backend, scipy_backend
from .esc_fraunhofer import single, parallel 
from .utils import *

__version__ = importlib.metadata.version(__package__ or "esc_psf")
__all__ = [ "__version__"]

path = Path(os.path.dirname(__file__))


