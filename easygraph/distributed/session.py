import atexit
import http.client
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid

DEFAULT_IMAGE = "ghcr.io/easy-graph/easygraph-distributed:1.6.4"


class SessionError(RuntimeError):
    pass


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Session:

    def __init__(self, *, cluster_type="k8s", num_workers=4, processes_per_worker=1,
                 threads_per_worker=1,
                 namespace="default", image=DEFAULT_IMAGE, pvc="easygraph-data",
                 worker_memory="2Gi", node_selector=None, worker_nodes=None,
                 launcher_node=None, launcher_tolerations=None, endpoint=None, token=None,
                 startup_timeout=180, request_timeout=300, idle_timeout=900,
                 heartbeat_interval=30, mpi_command="mpirun", python_executable=None,
                 data_root=None, kubectl="kubectl", kubeconfig=None):
        if type(num_workers) is not int or num_workers < 1:
            raise ValueError("num_workers must be a positive integer")
        if type(processes_per_worker) is not int or processes_per_worker < 1:
            raise ValueError("processes_per_worker must be a positive integer")
        if type(threads_per_worker) is not int or threads_per_worker < 1:
            raise ValueError("threads_per_worker must be a positive integer")
        if worker_nodes is not None:
            if (not isinstance(worker_nodes, (list, tuple)) or not worker_nodes or
                any(not isinstance(node, str) or not node.strip() for node in worker_nodes)):
                raise ValueError("worker_nodes must be a nonempty list or tuple of node names")
            if len(set(worker_nodes)) != len(worker_nodes) or len(worker_nodes) < num_workers:
                raise ValueError("worker_nodes must contain at least num_workers distinct node names")
            worker_nodes = list(worker_nodes)
        if launcher_node is not None and (not isinstance(launcher_node, str) or not launcher_node.strip()):
            raise ValueError("launcher_node must be a nonempty node name")
        if launcher_tolerations is not None and (not isinstance(launcher_tolerations, list) or
                any(not isinstance(item, dict) for item in launcher_tolerations)):
            raise ValueError("launcher_tolerations must be a list of Kubernetes tolerations")
        if cluster_type != "k8s" and (worker_nodes is not None or launcher_node is not None):
            raise ValueError("worker_nodes and launcher_node require cluster_type='k8s'")
        if min(startup_timeout, request_timeout, idle_timeout) <= 0 or heartbeat_interval < 0:
            raise ValueError("Timeouts must be positive, heartbeat_interval must be nonnegative")
        if heartbeat_interval and heartbeat_interval >= idle_timeout:
            raise ValueError("heartbeat_interval must be shorter than idle_timeout")
        if token is not None and (not isinstance(token, str) or len(token) < 32):
            raise ValueError("token must be a string of at least 32 characters")
        self.name = "eg-session-" + uuid.uuid4().hex[:16]
        self.namespace = namespace
        self._closed = self._faulted = False
        self._owned = endpoint is None
        self._resources_created = False
        self._kind = cluster_type
        self._token = token or secrets.token_urlsafe(32)
        self._timeout = request_timeout
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._heartbeat = self._process = self._forward = None
        self._temp = tempfile.TemporaryDirectory(prefix="easygraph-session-")
        self._log = open(Path(self._temp.name) / "session.log", "w+b")
        self._kubectl = [kubectl] + (["--kubeconfig", os.fspath(kubeconfig)] if kubeconfig else [])
        try:
            if endpoint is not None:
                if token is None:
                    raise ValueError("Connecting requires an explicit token")
                self.endpoint = endpoint
            elif cluster_type == "hosts":
                port = _free_port()
                self.endpoint = f"http://127.0.0.1:{port}"
                command = [mpi_command, "-np", str(num_workers * processes_per_worker),
                           python_executable or sys.executable, "-m", "easygraph.distributed.server",
                           "--session-id", self.name, "--port", str(port),
                           "--idle-timeout", str(idle_timeout)]
                if data_root:
                    command += ["--data-root", os.path.abspath(data_root)]
                env = dict(os.environ, EASYGRAPH_SESSION_TOKEN=self._token,
                           OMP_NUM_THREADS=str(threads_per_worker), OMP_DYNAMIC="FALSE",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                self._process = subprocess.Popen(command, env=env, stdout=self._log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            elif cluster_type == "k8s":
                from ._kubernetes import resources
                if shutil.which(kubectl) is None:
                    raise SessionError("kubectl is required to start a Kubernetes Session")
                secret, job = resources(self.name, self._token, image=image, namespace=namespace,
                    workers=num_workers, slots=processes_per_worker, threads=threads_per_worker,
                    memory=worker_memory, pvc=pvc, worker_nodes=worker_nodes,
                    launcher_node=launcher_node, launcher_tolerations=launcher_tolerations,
                    idle_timeout=idle_timeout, node_selector=node_selector if node_selector is not None
                    else {})
                manifest = json.dumps({"apiVersion": "v1", "kind": "List", "items": [secret, job]})
                self._kube("apply", "--dry-run=server", "-f", "-", input=manifest,
                           timeout=startup_timeout)
                self._resources_created = True
                self._kube("apply", "-f", "-", input=manifest, timeout=startup_timeout)
                created = json.loads(self._kube("get", "mpijob", self.name, "-o", "json"))
                owner = {"apiVersion": "kubeflow.org/v2beta1", "kind": "MPIJob",
                         "name": self.name, "uid": created["metadata"]["uid"]}
                self._kube("patch", "secret", self.name + "-auth", "--type=merge", "-p",
                    json.dumps({"metadata": {"ownerReferences": [owner]}}))
                deadline = time.monotonic() + startup_timeout
                while True:
                    try:
                        pod = json.loads(self._kube("get", "pod", self.name + "-worker-0", "-o", "json"))
                    except SessionError:
                        pod = None
                    if pod and pod["status"].get("phase") == "Running":
                        break
                    if pod and pod["status"].get("phase") in ("Failed", "Succeeded"):
                        raise SessionError("Session worker exited before startup. " + self._diagnostics())
                    if time.monotonic() >= deadline:
                        raise SessionError("Worker startup timed out. " + self._diagnostics())
                    self._stop.wait(0.25)
                port = _free_port()
                self.endpoint = f"http://127.0.0.1:{port}"
                self._forward_args = self._kubectl + ["-n", namespace,
                    "port-forward", "--address=127.0.0.1",
                    f"--pod-running-timeout={int(startup_timeout)}s",
                    "pod/" + self.name + "-worker-0", f"{port}:8787"]
                self._start_forward()
            else:
                raise ValueError("cluster_type must be 'k8s' or 'hosts'")
            parsed = urlsplit(self.endpoint)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/"):
                raise ValueError("endpoint must be an http(s) origin")
            self._address = parsed
            deadline = time.monotonic() + startup_timeout
            while True:
                try:
                    hello = self._request("ping", _timeout=min(2, startup_timeout), _startup=True)
                    self.session_id, self.size = hello["session_id"], hello["size"]
                    self.name = self.session_id
                    break
                except (OSError, http.client.HTTPException, SessionError) as exc:
                    if self._process is not None and self._process.poll() is not None:
                        raise SessionError("Session process exited during startup. " + self._log_tail()) from exc
                    if time.monotonic() >= deadline:
                        raise SessionError("Session startup timed out. " + self._diagnostics()) from exc
                    if self._forward is not None and self._forward.poll() is not None:
                        self._start_forward()
                    self._stop.wait(0.25)
            if self._owned and self.size != num_workers * processes_per_worker:
                raise SessionError("MPI rank count differs from the requested configuration")
            if self._owned and threads_per_worker > 1:
                ranks = self.info()["ranks"]
                if len(ranks) != self.size or any(
                    not rank.get("parallelism", {}).get("openmp_enabled") or
                    rank["parallelism"].get("max_threads") != threads_per_worker or
                    rank["parallelism"].get("team_size") != threads_per_worker
                    for rank in ranks
                ):
                    details = [{"rank": rank.get("rank"), "node": rank.get("node"),
                                "parallelism": rank.get("parallelism")} for rank in ranks]
                    raise SessionError(f"Workers do not provide {threads_per_worker} OpenMP threads: "
                                       + json.dumps(details))
            if heartbeat_interval:
                self._heartbeat = threading.Thread(target=self._keepalive,
                    args=(heartbeat_interval,), name="easygraph-session-heartbeat", daemon=True)
                self._heartbeat.start()
            atexit.register(self.close)
        except BaseException:
            self._cleanup()
            self._closed = True
            raise

    def _kube(self, *args, input=None, timeout=30):
        completed = subprocess.run(self._kubectl + ["-n", self.namespace, *args], input=input,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if completed.returncode:
            raise SessionError(completed.stderr.strip() or "kubectl failed")
        return completed.stdout

    def _start_forward(self):
        self._forward = subprocess.Popen(self._forward_args, stdout=self._log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)

    def _log_tail(self):
        self._log.flush()
        with open(self._log.name, "rb") as stream:
            stream.seek(max(0, os.path.getsize(self._log.name) - 4000))
            return stream.read().decode(errors="replace")

    def _diagnostics(self):
        output = self._log_tail()
        if self._kind == "k8s" and self._resources_created:
            try:
                output += self._kube("get", "pods", "-l", "easygraph-session=" + self.name, "-o", "wide")
                output += self._kube("logs", "job/" + self.name + "-launcher", "--tail=20")
            except Exception:
                pass
        return output

    def _request(self, op, _timeout=None, _startup=False, **parameters):
        with self._lock:
            if self._closed or (self._faulted and not _startup):
                raise SessionError("Session is closed or its transport failed; close it and create a new session")
            request_id = uuid.uuid4().hex
            data = json.dumps(dict(parameters, op=op, request_id=request_id), allow_nan=False).encode()
            kind = http.client.HTTPSConnection if self._address.scheme == "https" else http.client.HTTPConnection
            connection = kind(self._address.hostname, self._address.port,
                              timeout=_timeout or self._timeout)
            try:
                connection.request("POST", "/rpc", body=data, headers={
                    "Content-Type": "application/json", "Authorization": "Bearer " + self._token})
                response = connection.getresponse()
                raw = response.read(32 * 1024 * 1024 + 1)
                if response.status != 200:
                    raise SessionError(f"Session HTTP error {response.status}")
                if len(raw) > 32 * 1024 * 1024:
                    raise SessionError("Response exceeds 32 MiB; use sharded write()")
                reply = json.loads(raw)
                if reply.get("request_id") != request_id:
                    raise SessionError("Session response does not match the request")
            except (OSError, http.client.HTTPException, ValueError, SessionError) as exc:
                if not _startup:
                    self._faulted = True
                raise SessionError("Session transport failed; request was not retried because its outcome may be unknown: " + str(exc)) from exc
            finally:
                connection.close()
            if not reply.get("ok"):
                raise SessionError(reply.get("error", "Remote operation failed"))
            return reply["value"]

    def _keepalive(self, interval):
        while not self._stop.wait(interval):
            if self._lock.acquire(blocking=False):
                try:
                    self._request("ping", _timeout=min(5, self._timeout))
                except SessionError:
                    return
                finally:
                    self._lock.release()

    def info(self):
        return self._request("info")

    def read_edgelist(self, path, *, directed=False):
        value = self._request("load", path=os.fspath(path), directed=directed)
        return RemoteGraph(self, value)

    def graph(self, graph_id):
        return RemoteGraph(self, self._request("graph_info", graph_id=graph_id))

    def connection_info(self):
        return {"endpoint": self.endpoint, "token": self._token}

    def pagerank(self, graph, alpha=0.85, weight=None, *, max_iter=500, tol=1e-6):
        if not isinstance(graph, RemoteGraph) or graph.session is not self:
            raise ValueError("Graph must belong to this Session")
        graph._ensure_open()
        value = self._request("pagerank", graph_id=graph.id, alpha=alpha,
                              weight=weight, max_iter=max_iter, tol=tol)
        return RemoteResult(self, value)

    @staticmethod
    def _terminate(process):
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)

    def _cleanup(self):
        self._stop.set()
        self._terminate(self._forward)
        self._terminate(self._process)
        cleanup_error = None
        if self._owned and self._resources_created:
            errors = []
            for kind, name in (("mpijob", self.name), ("secret", self.name + "-auth")):
                try:
                    self._kube("delete", kind, name, "--ignore-not-found", "--wait=false")
                except Exception as exc:
                    errors.append(str(exc))
            if not errors:
                self._resources_created = False
            else:
                cleanup_error = "; ".join(errors)
        self._log.close()
        self._temp.cleanup()
        return cleanup_error

    def close(self):
        if self._closed:
            return
        atexit.unregister(self.close)
        self._stop.set()
        if self._heartbeat is not None and self._heartbeat is not threading.current_thread():
            self._heartbeat.join(timeout=6)
        error = None
        with self._lock:
            try:
                if self._owned and not self._faulted:
                    self._request("shutdown", _timeout=10)
                    if self._process is not None:
                        self._process.wait(timeout=5)
            except (SessionError, subprocess.TimeoutExpired) as exc:
                error = exc
            finally:
                self._closed = True
                cleanup_error = self._cleanup()
        if cleanup_error:
            raise SessionError(f"Failed to clean up Session {self.name}: {cleanup_error}")
        if error and not self._owned:
            raise error

    def shutdown(self):
        try:
            self._request("shutdown", _timeout=10)
        finally:
            self._faulted = True
            self.close()

    def __enter__(self):
        if self._closed:
            raise SessionError("Session is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class RemoteGraph:
    _easygraph_distributed = True
    _easygraph_remote = True
    cflag = 0

    def __init__(self, session, metadata):
        self.session, self.id = session, metadata["id"]
        self._metadata, self._closed = metadata, False

    def _ensure_open(self):
        if self._closed or self.session._closed:
            raise SessionError("Remote graph or Session is closed")

    def info(self):
        self._ensure_open()
        return self.session._request("graph_info", graph_id=self.id)

    def number_of_nodes(self):
        self._ensure_open()
        return self._metadata["num_nodes"]

    def number_of_edges(self):
        self._ensure_open()
        return self._metadata["num_edges"]

    def is_directed(self):
        return self._metadata["directed"]

    def is_multigraph(self):
        return False

    def close(self):
        if not self._closed and not self.session._closed:
            self.session._request("drop_graph", graph_id=self.id)
        self._closed = True


class RemoteResult:
    def __init__(self, session, metadata):
        self.session, self.id = session, metadata["id"]
        self.iterations, self.count = metadata["iterations"], metadata["count"]
        self._closed = False

    def _ensure_open(self):
        if self._closed or self.session._closed:
            raise SessionError("Remote result or Session is closed")

    def to_dict(self, *, max_vertices=100000):
        self._ensure_open()
        pairs = self.session._request("gather", result_id=self.id, max_vertices=max_vertices)
        return dict(pairs)

    def gather(self, *, max_vertices=100000):
        return self.to_dict(max_vertices=max_vertices)

    def to_dataframe(self, *, max_vertices=100000):
        import pandas as pd
        return pd.DataFrame(self.to_dict(max_vertices=max_vertices).items(), columns=["id", "value"])

    def write(self, path):
        self._ensure_open()
        return self.session._request("write", result_id=self.id, path=os.fspath(path))["path"]

    def close(self):
        if not self._closed and not self.session._closed:
            self.session._request("drop_result", result_id=self.id)
        self._closed = True


session = Session
