
import os

from .context import Context


class DistributedGraph:

    cflag = 0
    _easygraph_distributed = True

    def __init__(self, native_graph, context):
        self._native = native_graph
        self.context = context

    def is_directed(self):
        return False

    def is_multigraph(self):
        return False

    def number_of_nodes(self):
        self.context._ensure_open()
        return self._native.global_num_vertices

    def number_of_edges(self):
        self.context._ensure_open()
        return self._native.global_num_edges

    @property
    def local_nodes(self):
        self.context._ensure_open()
        return tuple(self._native.vertices)

    @property
    def local_num_arcs(self):
        self.context._ensure_open()
        return self._native.local_num_arcs

    def _unsupported(self):
        raise NotImplementedError(
            "This operation requires a distributed implementation. "
            "Distributed graphs do not expose a global node/adjacency view."
        )

    def __iter__(self):
        return self._unsupported()

    def __len__(self):
        return self.number_of_nodes()

    @property
    def nodes(self):
        return self._unsupported()

    @property
    def adj(self):
        return self._unsupported()

    @property
    def edges(self):
        return self._unsupported()


class DistributedDiGraph(DistributedGraph):
    def is_directed(self):
        return True


def read_edgelist(path, *, context, directed=False):
    if not isinstance(context, Context):
        raise TypeError("context must be an easygraph.distributed.Context")
    context._ensure_open()
    graph = context._backend.read_edgelist(
        os.fsdecode(os.fspath(path)), context._comm, bool(directed)
    )
    graph_type = DistributedDiGraph if directed else DistributedGraph
    return graph_type(graph, context)
