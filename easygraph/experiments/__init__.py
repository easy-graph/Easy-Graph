try:
    from .base import BaseTask
    from .hypergraphs import HypergraphVertexClassificationTask


except:
    print(
        "Warning raise in module: experiments. Please install Pytorch before you use"
        " functions related to nueral network"
    )
