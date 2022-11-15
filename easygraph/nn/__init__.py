try:
    from .convs import *
except:
    print(
        "Error raise in module:nn. Please install Pytorch before you use functions"
        " related to Hypergraph"
    )
