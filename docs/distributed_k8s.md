# Distributed EasyGraph on Kubernetes

The Python client is part of `Python-EasyGraph`. It launches an MPIJob and
communicates with the Worker engine; the client machine does not need MPI or a
container builder. Worker Pods use the matching public image
`ghcr.io/easy-graph/easygraph-distributed:1.6.4`.

## Cluster prerequisites

- A Kubernetes cluster with at least two schedulable Worker nodes and a
  kubeconfig usable by the Python client.
- [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) installed.
- A ReadWriteMany PVC named `easygraph-data` in the `default` namespace, or
  pass another namespace and PVC name to `eg.session`.
- Nodes able to pull the public Worker image from GHCR. The package must be
  marked public in GitHub Packages before users start a Session.
- Permission for the client to create, read, and delete MPIJobs and Secrets,
  and to read Pods and open a Pod port forward in the selected namespace.

The shared PVC is mounted at `/shared` in every Worker Pod. Put the edge list
there before starting a Session. A client-side file path is not uploaded.

## Client installation and use

```bash
python3 -m pip install --upgrade Python-EasyGraph
```

```python
import easygraph as eg

with eg.session(num_workers=2) as sess:
    graph = sess.read_edgelist("/shared/graph.tsv", directed=True)
    result = eg.pagerank(graph, alpha=0.85)
    print(result.iterations)
    result.write("/shared/pagerank-result")
```

The `Session` holds the loaded graph for repeated algorithm calls. It removes
its MPIJob and Secret when closed. The default Worker image version is pinned
to the matching Python release, and can be overridden with `image=`.

For a cluster with dedicated compute nodes and a tainted control node, specify
placement and resources explicitly:

```python
with eg.session(
    namespace="benchmark",
    pvc="benchmark-data-rwx",
    num_workers=2,
    worker_nodes=["eg-b", "eg-c"],
    launcher_node="eg-a",
    launcher_tolerations=[{
        "key": "node-role.kubernetes.io/control-plane",
        "operator": "Exists",
        "effect": "NoSchedule",
    }],
    threads_per_worker=16,
    worker_memory="60Gi",
) as sess:
    graph = sess.read_edgelist("/shared/graph.tsv", directed=True)
    result = sess.pagerank(graph)
```

This release supports distributed PageRank. Large graph loading still needs
memory for parsed records and CSR construction on each Worker. Use `write()`
for large results because `to_dict()` gathers scores to the client.

## Release order for maintainers

Publish and make the matching GHCR image public first. Then release the
`Python-EasyGraph` wheel and source distribution. The image workflow is
`.github/workflows/distributed_image_publish.yml`; the existing wheel workflow
can publish from a version tag after the image is available. Verify the public
image digest and a fresh `pip install Python-EasyGraph==1.6.4` before announcing
the release.
