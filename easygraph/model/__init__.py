try:
    from .hypergraphs import *
except:
    print(
        "Error raise in module:model.Please install Pytorch before you use functions"
        " related to Hypergraph"
    )
