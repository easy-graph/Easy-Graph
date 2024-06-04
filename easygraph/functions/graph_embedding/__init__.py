from .deepwalk import *
from .NOBE import *
from .node2vec import *


try:
    from .line import *
    from .sdne import *
except:
    print(
        "Warning raise in module:graph_embedding. Please install packages Pytorch"
        " before you use functions related to graph_embedding"
    )
