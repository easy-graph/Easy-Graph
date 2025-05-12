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


try:
    from easygraph.functions.hypergraph import *
except:
    print(
        "Warning raise in module:model.Please install "
        "Pytorch before you use functions"
        " related to Hypergraph"
    )
