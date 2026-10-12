from easygraph._optional import HAS_TORCH as _HAS_TORCH
from easygraph._optional import optional_torch_modules as _torch_imports


if _HAS_TORCH:
    with _torch_imports():
        from .base import BaseTask
        from .hypergraphs import HypergraphVertexClassificationTask
