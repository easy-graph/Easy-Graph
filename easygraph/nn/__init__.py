from easygraph._optional import HAS_TORCH as _HAS_TORCH
from easygraph._optional import optional_torch_modules as _torch_imports


if _HAS_TORCH:
    with _torch_imports():
        from .convs.common import MLP
        from .convs.common import MultiHeadWrapper
        from .convs.hypergraphs import HGNNConv
        from .convs.hypergraphs import HGNNPConv
        from .convs.hypergraphs import HNHNConv
        from .convs.hypergraphs import HWNNConv
        from .convs.hypergraphs import HyperGCNConv
        from .convs.hypergraphs import JHConv
        from .convs.hypergraphs import UniGATConv
        from .convs.hypergraphs import UniGCNConv
        from .convs.hypergraphs import UniGINConv
        from .convs.hypergraphs import UniSAGEConv
        from .convs.hypergraphs.halfnlh_conv import HalfNLHconv
        from .convs.pma import PMA
        from .loss import BPRLoss
        from .regularization import EmbeddingRegularization
