from .autotranscript import *
from .transcriber import *
from .audio import *
from .transcript_exporter import *
from .diarisation import *

from .misc import *

from .cli import *
 
 # set __version__ attribute
 
import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)
