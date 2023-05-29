<<<<<<< HEAD
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
=======
from .deepwalk import *
from .NOBE import *
from .node2vec import *


try:
    from .line import *
    from .sdne import *
except:
    print(
        "Warning raise in module:graph_embedding. Please install packages pytorch"
        "before you use functions related to graph_embedding"
    )


# from .sdne import SDNE
>>>>>>> 622d76c2ce75db856dfd2eb6540dea6c9a7fe225
