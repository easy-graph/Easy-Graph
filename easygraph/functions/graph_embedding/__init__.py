from easygraph._optional import HAS_TORCH as _HAS_TORCH
from easygraph._optional import optional_torch_modules as _torch_imports

from .deepwalk import *
from .NOBE import *
from .node2vec import *


if _HAS_TORCH:
    with _torch_imports():
        from .line import *
        from .sdne import *
