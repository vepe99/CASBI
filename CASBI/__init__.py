
from importlib import metadata
from .utils import *
from .generator.nf.generator import *

__version__ = metadata.version(__package__)
del metadata


