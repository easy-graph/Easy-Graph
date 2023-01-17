try:
    from .deepwalk import *
    from .line import LINE
    from .NOBE import *
    from .node2vec import *
    from .sdne import SDNE
except:
    print(
        "Warning raise in module:graph_embedding. Please install Tensorflow before you"
        " use functions related to graph_embedding"
    )
