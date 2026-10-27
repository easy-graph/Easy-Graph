
import atexit
import importlib


def _load_backend():
    try:
        return importlib.import_module("cpp_easygraph_distributed")
    except ImportError as exc:
        raise ImportError(
            "The distributed extension is unavailable. Install MPI and rebuild with "
            "EASYGRAPH_ENABLE_DISTRIBUTED=ON; see docs/distributed.md."
        ) from exc


class Context:

    def __init__(self):
        self._backend = _load_backend()
        atexit.register(self._backend.finalize_mpi)
        self._comm = self._backend.Communicator()

    def _ensure_open(self):
        if self._comm.closed:
            raise RuntimeError("The distributed Context is closed.")

    @property
    def rank(self):
        self._ensure_open()
        return self._comm.rank

    @property
    def size(self):
        self._ensure_open()
        return self._comm.size

    def barrier(self):
        self._ensure_open()
        self._comm.barrier()

    def close(self):
        self._comm.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
