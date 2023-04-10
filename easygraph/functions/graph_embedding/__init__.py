from .deepwalk import *
from .NOBE import *
from .node2vec import *


try:
    from .torch_line import *
    from .torch_sdne import *
except:
    print(
        "Warning raise in module:graph_embedding. Please install packages Pytorch and"
        " cogdl before you use functions related to graph_embedding"
    )


# from .sdne import SDNE
