from easygraph._optional import HAS_TORCH as _HAS_TORCH
from easygraph._optional import optional_torch_modules as _torch_imports


if _HAS_TORCH:
    with _torch_imports():
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
