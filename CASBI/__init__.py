
from importlib import metadata
from .utils.utils import *
from .generator.nf.nf_model import *
from .generator.fff.fff_model import *

__version__ = metadata.version(__package__)
del metadata


