import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import socket
import time
import uuid

from .context import Context
from .graph import read_edgelist
from .pagerank import pagerank

MAX_REQUEST = 1024 * 1024


class Runtime:
    def __init__(self, context, session_id, data_root=None):
        self.ctx = context
        self.session_id = session_id
        self.data_root = Path(data_root).resolve() if data_root else None
        self.graphs, self.results = {}, {}
        self.load_count = self.query_count = 0
        node_file = Path("/tmp/easygraph-node")
        self.identity = {"rank": context.rank, "pid": os.getpid(),
                         "hostname": socket.gethostname(),
                         "node": node_file.read_text().strip() if node_file.exists()
                         else socket.gethostname()}
        if hasattr(context._backend, "parallel_info"):
            self.identity["parallelism"] = context._backend.parallel_info()

    def path(self, value):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("A nonempty filesystem path is required")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("Use an absolute server-side file path")
        resolved = path.resolve()
        if self.data_root is not None and resolved != self.data_root and self.data_root not in resolved.parents:
            raise ValueError("Path is outside the configured data root")
        return str(resolved)

    def records(self, local):
        value = json.dumps(local, allow_nan=False)
        return [json.loads(self.ctx._comm.broadcast_string(value, rank))
                for rank in range(self.ctx.size)]

    def prepare(self, command):
        op = command["op"]
        if op in ("pagerank", "graph_info", "drop_graph"):
            if command["graph_id"] not in self.graphs:
                raise ValueError("Graph has been released or does not belong to this session")
        if op in ("gather", "write", "drop_result"):
            if command["result_id"] not in self.results:
                raise ValueError("Result has been released or does not belong to this session")
        if op == "gather" and self.results[command["result_id"]]["count"] > command["max_vertices"]:
            raise ValueError("Result exceeds max_vertices; increase the explicit limit or use write()")

    def execute(self, command):
        error = ""
        try:
            self.prepare(command)
        except Exception as exc:
            error = str(exc)
        self.ctx._comm.check_error(error)
        op, value = command["op"], None
        graph = result = None
        try:
            if op == "load":
                graph = read_edgelist(command["path"], context=self.ctx,
                                      directed=command["directed"])
            elif op == "pagerank":
                graph = self.graphs[command["graph_id"]]["graph"]
                result = pagerank(graph, alpha=command["alpha"], weight=command["weight"],
                                  max_iter=command["max_iter"], tol=command["tol"])
            elif op == "gather":
                values = self.results[command["result_id"]]["result"].gather(root=0)
                if self.ctx.rank == 0:
                    value = [[int(k), float(v)] for k, v in values.items()]
            elif op == "write":
                self.results[command["result_id"]]["result"].write(command["path"])
                value = {"path": command["path"]}
        except Exception as exc:
            error = str(exc)
        self.ctx._comm.check_error(error)
        if op == "load":
            self.load_count += 1
            self.graphs[command["id"]] = {"graph": graph, "path": command["path"],
                "load_number": self.load_count, "native_identity": id(graph._native)}
            value = self.graph_info(command["id"])
        elif op == "pagerank":
            self.query_count += 1
            value = {"id": command["id"], "graph_id": command["graph_id"],
                     "iterations": result.iterations, "count": graph.number_of_nodes()}
            self.results[command["id"]] = dict(value, result=result)
        elif op == "graph_info":
            value = self.graph_info(command["graph_id"])
        elif op == "drop_graph":
            del self.graphs[command["graph_id"]]
            value = {"released": True}
        elif op == "drop_result":
            del self.results[command["result_id"]]
            value = {"released": True}
        elif op == "info":
            value = {"session_id": self.session_id, "size": self.ctx.size,
                     "load_count": self.load_count, "query_count": self.query_count,
                     "graphs": list(self.graphs), "results": list(self.results),
                     "ranks": self.records(dict(self.identity,
                         load_count=self.load_count, query_count=self.query_count))}
        elif op == "shutdown":
            self.results.clear()
            self.graphs.clear()
            value = {"closed": True}
        elif op == "ping":
            value = {"session_id": self.session_id, "size": self.ctx.size}
        return value

    def graph_info(self, graph_id):
        item = self.graphs[graph_id]
        graph = item["graph"]
        return {"id": graph_id, "path": item["path"], "directed": graph.is_directed(),
                "num_nodes": graph.number_of_nodes(), "num_edges": graph.number_of_edges(),
                "load_number": item["load_number"],
                "partitions": self.records(dict(self.identity,
                    native_identity=item["native_identity"],
                    local_vertices=len(graph.local_nodes), local_arcs=graph.local_num_arcs))}


def validate(body):
    if not isinstance(body, dict) or not isinstance(body.get("request_id"), str):
        raise ValueError("A JSON object with request_id is required")
    op = body.get("op")
    required = {"ping": (), "info": (), "shutdown": (),
                "load": ("path", "directed"), "graph_info": ("graph_id",),
                "drop_graph": ("graph_id",), "drop_result": ("result_id",),
                "pagerank": ("graph_id", "alpha", "weight", "max_iter", "tol"),
                "gather": ("result_id", "max_vertices"), "write": ("result_id", "path")}
    if op not in required:
        raise ValueError("Unsupported operation")
    if set(body) != {"op", "request_id", *required[op]}:
        raise ValueError("Missing or unexpected request fields")
    for key in ("graph_id", "result_id"):
        if key in body and (not isinstance(body[key], str) or len(body[key]) != 32):
            raise ValueError("Invalid object handle")
    if op == "load" and type(body["directed"]) is not bool:
        raise ValueError("directed must be boolean")
    for key in ("max_iter", "max_vertices"):
        if key in body and (type(body[key]) is not int or body[key] < 1):
            raise ValueError(key + " must be a positive integer")
    if op == "pagerank":
        for key in ("alpha", "tol"):
            v = body[key]
            if type(v) not in (float, int) or not math.isfinite(v):
                raise ValueError(key + " must be finite")
        if not 0 < body["alpha"] < 1 or body["tol"] <= 0:
            raise ValueError("Require 0 < alpha < 1 and tol > 0")
        if body["weight"] not in (None, "weight"):
            raise ValueError("Only weight=None or 'weight' is supported")
    command = dict(body)
    if op in ("load", "pagerank"):
        command["id"] = uuid.uuid4().hex
    return command


def serve(ctx, args, token):
    runtime = Runtime(ctx, args.session_id, args.data_root)
    if ctx.rank:
        while True:
            command = json.loads(ctx._comm.broadcast_string("", 0))
            try:
                runtime.execute(command)
            except Exception:
                pass
            if command["op"] == "shutdown":
                return
    else:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *values):
                pass

            def do_POST(self):
                self.connection.settimeout(10)
                if self.path != "/rpc":
                    self.send_error(404)
                    return
                auth = self.headers.get("Authorization", "")
                if not hmac.compare_digest(auth, "Bearer " + token):
                    self.send_error(401)
                    return
                request_id = None
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_REQUEST:
                        raise ValueError("Request body must be between 1 byte and 1 MiB")
                    body = json.loads(self.rfile.read(length))
                    request_id = body.get("request_id") if isinstance(body, dict) else None
                    command = validate(body)
                    if command["op"] in ("load", "write"):
                        command["path"] = runtime.path(command["path"])
                    server.last_activity = time.monotonic()
                    ctx._comm.broadcast_string(json.dumps(command, allow_nan=False), 0)
                    value = runtime.execute(command)
                    response = {"ok": True, "request_id": request_id, "value": value}
                    if command["op"] == "shutdown":
                        server.stopping = True
                except Exception as exc:
                    response = {"ok": False, "request_id": request_id,
                                "error": str(exc), "error_type": type(exc).__name__}
                try:
                    data = json.dumps(response, allow_nan=False).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                server.last_activity = time.monotonic()

        error = ""
        server = None
        try:
            server = HTTPServer((args.bind, args.port), Handler)
            server.timeout = min(1.0, args.idle_timeout)
            server.stopping, server.last_activity = False, time.monotonic()
        except Exception as exc:
            error = str(exc)
        startup = {"op": "ping", "request_id": "startup"}
        ctx._comm.broadcast_string(json.dumps(startup), 0)
        runtime.execute(startup)
        if error:
            command = {"op": "shutdown", "request_id": "startup-failed"}
            ctx._comm.broadcast_string(json.dumps(command), 0)
            runtime.execute(command)
            raise RuntimeError(error)
        print(json.dumps({"event": "session_ready", "session_id": args.session_id,
                          "ranks": ctx.size, "port": args.port}), flush=True)
        try:
            while not server.stopping:
                if time.monotonic() - server.last_activity >= args.idle_timeout:
                    command = {"op": "shutdown", "request_id": "idle-timeout"}
                    ctx._comm.broadcast_string(json.dumps(command), 0)
                    runtime.execute(command)
                    server.stopping = True
                    break
                server.handle_request()
        finally:
            if not server.stopping:
                command = {"op": "shutdown", "request_id": "server-stopped"}
                ctx._comm.broadcast_string(json.dumps(command), 0)
                runtime.execute(command)
            server.server_close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--session-id", required=True)
    p.add_argument("--idle-timeout", type=float, default=900)
    p.add_argument("--data-root")
    args = p.parse_args()
    token = os.environ.get("EASYGRAPH_SESSION_TOKEN", "")
    if len(token) < 32 or args.idle_timeout <= 0:
        p.error("A token of at least 32 characters and positive idle-timeout are required")
    with Context() as ctx:
        serve(ctx, args, token)


if __name__ == "__main__":
    main()
