"""Check the installed Python package, native extension and SciPy conversion."""

import math
import os
import tempfile
import warnings

from importlib.metadata import version
from importlib.util import find_spec


# Run outside the source directory so an uninstalled checkout cannot pass.
os.chdir(tempfile.gettempdir())

import cpp_easygraph


has_torch = find_spec("torch") is not None
with warnings.catch_warnings(record=True) as import_warnings:
    warnings.simplefilter("always")
    import easygraph
    import easygraph as eg

torch_warnings = [
    item
    for item in import_warnings
    if "EasyGraph: PyTorch is not installed" in str(item.message)
]
assert len(torch_warnings) == (0 if has_torch else 1)
if torch_warnings:
    print(str(torch_warnings[0].message))

from scipy.sparse import csr_matrix


for graph_type in (eg.Graph, eg.DiGraph):
    graph = graph_type()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    native = graph.cpp()
    assert isinstance(native, (cpp_easygraph.Graph, cpp_easygraph.DiGraph))
    assert native.number_of_nodes() == 3
    assert native.number_of_edges() == 3
    native.add_edge(2, 3, weight=2.0)
    assert native.number_of_nodes() == 4
    assert native.number_of_edges() == 4
    restored = native.py()
    assert restored.is_directed() == graph.is_directed()
    assert restored[2][3]["weight"] == 2.0
    native.remove_node(3)
    assert native.number_of_nodes() == 3
    assert native.number_of_edges() == 3

graph = eg.Graph()
graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
native = graph.cpp()
python_pagerank = eg.pagerank(graph)
assert set(python_pagerank) == {0, 1, 2}
assert all(math.isclose(value, 1.0 / 3.0) for value in python_pagerank.values())
centrality = cpp_easygraph.cpp_degree_centrality(native)
assert set(centrality) == {0, 1, 2}
assert all(math.isclose(value, 1.0) for value in centrality.values())
pagerank = cpp_easygraph.cpp_pagerank(native)
assert len(pagerank) == 3
assert all(math.isclose(value, 1.0 / 3.0) for value in pagerank)

matrix = csr_matrix([[0.0, 2.0], [2.0, 0.0]])
converted = eg.from_scipy_sparse_matrix(matrix)
assert converted.number_of_nodes() == 2
assert converted.number_of_edges() == 1
assert converted[0][1]["weight"] == 2.0

if has_torch:
    import numpy as np
    import torch

    array = np.array([1.0, 2.0], dtype=np.float32)
    assert np.array_equal(torch.from_numpy(array).numpy(), array)
    hypergraph = eg.Hypergraph(num_v=3, e_list=[[0, 1], [1, 2]])
    features = torch.tensor([[1.0], [2.0], [3.0]])
    assert torch.allclose(hypergraph.v2e(features), torch.tensor([[1.5], [2.5]]))
    assert torch.allclose(hypergraph.v2v(features), torch.tensor([[1.5], [2.0], [2.5]]))
    print("PyTorch {}: hypergraph message passing passed".format(torch.__version__))
else:
    for operation in (lambda: eg.Hypergraph(num_v=3), lambda: graph.A):
        try:
            operation()
        except ImportError as error:
            assert "https://pytorch.org/get-started/locally/" in str(error)
        else:
            raise AssertionError(
                "A PyTorch-dependent feature did not report its missing dependency"
            )

print(
    "EasyGraph {}: Python, C++ and SciPy smoke checks passed".format(
        version("Python-EasyGraph")
    )
)
