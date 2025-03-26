try:
    from .hypergraphs import DHCF
    from .hypergraphs import HGNN
    from .hypergraphs import HGNNP
    from .hypergraphs import HNHN
    from .hypergraphs import HWNN
    from .hypergraphs import HyperGCN
    from .hypergraphs import SetGNN
    from .hypergraphs import UniGAT
    from .hypergraphs import UniGCN
    from .hypergraphs import UniGIN
    from .hypergraphs import UniSAGE

except:
    print(
        "Warning raise in module:model.Please install "
        "Pytorch before you use hypergraph neural networks"
        " related to Hypergraph"
    )
