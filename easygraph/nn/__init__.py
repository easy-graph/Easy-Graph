try:
    from .convs.common import MultiHeadWrapper
    from .convs.hypergraphs import HGNNConv
    from .convs.hypergraphs import HGNNPConv
    from .convs.hypergraphs import HNHNConv
    from .convs.hypergraphs import HyperGCNConv
    from .convs.hypergraphs import JHConv
    from .convs.hypergraphs import UniGATConv
    from .convs.hypergraphs import UniGCNConv
    from .convs.hypergraphs import UniGINConv
    from .convs.hypergraphs import UniSAGEConv
    from .loss import BPRLoss
    from .regularization import EmbeddingRegularization
except:
    print(
        "Warning raise in module:nn. Please install Pytorch before you use functions"
        " related to Hypergraph"
    )
