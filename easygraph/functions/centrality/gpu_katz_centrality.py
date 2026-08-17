try:
    import cpp_easygraph
except ImportError:
    cpp_easygraph = None

from easygraph.utils import *
from easygraph.utils.decorators import *

__all__ = [
    "gpu_katz_centrality",
    "PreparedKatzContext",
    "prepare_gpu_katz",
    "GpuKatzTsvDataset",
    "load_gpu_katz_tsv_dataset",
]


@not_implemented_for("multigraph")
@hybrid("cpp_gpu_katz_centrality")
def gpu_katz_centrality(
    G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-6, normalized=True
):
    """Compute deterministic incoming Katz centrality with EGGPU.

    The input must be a ``DiGraphC`` created by an EasyGraph build with
    ``EASYGRAPH_ENABLE_GPU=TRUE``. Scores satisfy ``x = alpha * A_in * x +
    beta`` and are returned as a dictionary keyed by the graph's public node
    labels. The function raises instead of returning partial scores when the
    iteration limit is reached.
    """
    raise EasyGraphError(
        "gpu_katz_centrality requires DiGraphC and EASYGRAPH_ENABLE_GPU=TRUE"
    )


class PreparedKatzContext:
    """Explicit immutable graph snapshot for repeated EGGPU Katz queries.

    The context owns an incoming-CSR snapshot, the device buffers needed by
    Katz, and the mapping back to the graph's public node labels. Later changes
    to the source graph do not change this context. Call :meth:`run` for each
    parameter set, then :meth:`close` when the context is no longer needed.
    """

    def __init__(self, G):
        if cpp_easygraph is None:
            raise EasyGraphError(
                "prepare_gpu_katz requires DiGraphC and EASYGRAPH_ENABLE_GPU=TRUE"
            )
        try:
            self._handle = cpp_easygraph.cpp_prepare_gpu_katz(G)
        except AttributeError as error:
            raise EasyGraphError(
                "prepare_gpu_katz requires DiGraphC and EASYGRAPH_ENABLE_GPU=TRUE"
            ) from error

    @classmethod
    def _from_handle(cls, handle):
        context = cls.__new__(cls)
        context._handle = handle
        return context

    def run(self, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-6, normalized=True):
        """Run exact incoming Katz against the prepared graph snapshot."""
        if self._handle is None:
            raise EasyGraphError("PreparedKatzContext is closed")
        return cpp_easygraph.cpp_run_prepared_gpu_katz(
            self._handle, alpha, beta, max_iter, tol, normalized
        )

    def close(self):
        """Release the prepared GPU context and its device allocations."""
        self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@not_implemented_for("multigraph")
def prepare_gpu_katz(G):
    """Prepare an explicit EGGPU Katz context for repeated queries on ``G``."""
    return PreparedKatzContext(G)


class GpuKatzTsvDataset:
    """Reusable two-column TSV input for optional EGGPU Katz workflows.

    ``arcs.tsv`` contains contiguous ``source<TAB>target`` IDs. ``node_map.tsv``
    begins with ``new_id<TAB>original_id`` and preserves the public labels,
    including isolates. Loading uses mmap on POSIX systems. This object does not
    change the default :func:`gpu_katz_centrality` behavior.
    """

    def __init__(self, arcs_path, node_map_path):
        if cpp_easygraph is None:
            raise EasyGraphError(
                "load_gpu_katz_tsv_dataset requires EASYGRAPH_ENABLE_GPU=TRUE"
            )
        try:
            self._handle = cpp_easygraph.cpp_load_gpu_katz_tsv_dataset(
                str(arcs_path), str(node_map_path)
            )
            self._node_count, self._edge_count, self._max_in_degree = (
                cpp_easygraph.cpp_gpu_katz_tsv_dataset_metadata(self._handle)
            )
        except AttributeError as error:
            raise EasyGraphError(
                "load_gpu_katz_tsv_dataset requires EASYGRAPH_ENABLE_GPU=TRUE"
            ) from error

    def _require_open(self):
        if self._handle is None:
            raise EasyGraphError("GpuKatzTsvDataset is closed")

    @property
    def node_count(self):
        return self._node_count

    @property
    def edge_count(self):
        return self._edge_count

    @property
    def max_in_degree(self):
        return self._max_in_degree

    def to_digraphc(self):
        """Build a ``DiGraphC`` with original labels for the unchanged public API."""
        self._require_open()
        return cpp_easygraph.cpp_gpu_katz_tsv_dataset_to_digraph(self._handle)

    def prepare(self):
        """Create an explicit reusable EGGPU Katz context from the loaded CSR."""
        self._require_open()
        return PreparedKatzContext._from_handle(
            cpp_easygraph.cpp_prepare_gpu_katz_tsv_dataset(self._handle)
        )

    def close(self):
        """Release the host-side TSV data owned by this dataset object."""
        self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def load_gpu_katz_tsv_dataset(arcs_path, node_map_path):
    """Load a standardized integer TSV graph for optional EGGPU Katz reuse.

    Use :meth:`GpuKatzTsvDataset.to_digraphc` followed by the unchanged
    :func:`gpu_katz_centrality` API for graph compatibility. Use
    :meth:`GpuKatzTsvDataset.prepare` only when an explicit reusable GPU Katz
    context is desired.
    """
    return GpuKatzTsvDataset(arcs_path, node_map_path)
