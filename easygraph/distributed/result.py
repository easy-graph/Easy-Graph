
import json
import time
from pathlib import Path


class DistributedResult:
    def __init__(self, context, local, *, iterations=None):
        self.context = context
        self._local = local
        self.iterations = iterations

    def local(self):
        return dict(self._local)

    def gather(self, root=0):
        ctx = self.context
        ctx._ensure_open()
        ctx._comm.agree(str(root))
        ctx._comm.check_error(
            "root must be an integer rank in this Context."
            if not isinstance(root, int) or not 0 <= root < ctx.size else ""
        )
        result = ctx._comm.gather_result(
            list(self._local), list(self._local.values()), root
        )
        return dict(zip(*result)) if ctx.rank == root else None

    def write(self, directory):
        ctx = self.context
        ctx._ensure_open()
        directory = Path(directory)
        ctx._comm.agree(str(directory))
        error = ""
        if ctx.rank == 0:
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                error = str(exc)
        ctx._comm.check_error(error)
        ctx.barrier()
        error = ""
        try:
            deadline = time.monotonic() + 35
            while True:
                try:
                    stream = (directory / f"part-{ctx.rank:05d}.tsv").open("x")
                    break
                except FileNotFoundError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            with stream:
                for vertex, value in sorted(self._local.items()):
                    stream.write(f"{vertex}\t{value:.17g}\n")
        except OSError as exc:
            error = str(exc)
        ctx._comm.check_error(error)
        ctx.barrier()
        error = ""
        if ctx.rank == 0:
            try:
                with (directory / "manifest.json").open("x") as stream:
                    json.dump({
                        "format": "vertex-value-tsv", "parts": ctx.size,
                        "iterations": self.iterations,
                    }, stream, indent=2)
            except OSError as exc:
                error = str(exc)
        ctx._comm.check_error(error)
