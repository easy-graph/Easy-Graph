from easygraph._optional import HAS_TORCH as _HAS_TORCH
from easygraph._optional import optional_torch_modules as _torch_imports

from .directed_graph import DiGraph
from .directed_graph import DiGraphC
from .directed_multigraph import MultiDiGraph
from .graph import Graph
from .graph import GraphC
from .graphviews import *
from .multigraph import MultiGraph
from .operation import *


if _HAS_TORCH:
    with _torch_imports():
        from .base import BaseHypergraph
        from .base import load_structure
        from .hypergraph import Hypergraph
