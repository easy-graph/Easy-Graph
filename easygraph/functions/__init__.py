from easygraph._optional import HAS_TORCH as _HAS_TORCH
from easygraph._optional import optional_torch_modules as _torch_imports
from easygraph.functions.basic import *
from easygraph.functions.centrality import *
from easygraph.functions.community import *
from easygraph.functions.components import *
from easygraph.functions.core import *
from easygraph.functions.drawing import *
from easygraph.functions.graph_embedding import *
from easygraph.functions.graph_generator import *
from easygraph.functions.isolate import *
from easygraph.functions.path import *
from easygraph.functions.structural_holes import *


if _HAS_TORCH:
    with _torch_imports():
        from easygraph.functions.hypergraph import *
