from easygraph.utils import *


__all__ = [
    "k_core",
]


@hybrid("cpp_k_core")
def k_core(G):
    raise EasyGraphError("Please input GraphC or DiGraphC.")
