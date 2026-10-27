

def resources(name, token, *, image, namespace, workers, slots, memory,
              pvc, idle_timeout, node_selector, threads=1, worker_nodes=None,
              launcher_node=None, launcher_tolerations=None):
    import copy

    command = ["mpirun", "--hostfile", "/etc/mpi/hostfile", "-np", str(workers * slots),
               "--bind-to", "none", "--map-by", "slot", "--mca", "pml", "ob1",
               "--mca", "btl", "self,tcp", "--mca", "btl_tcp_if_exclude", "lo",
               "-x", "EASYGRAPH_SESSION_TOKEN",
               "-x", "OMP_NUM_THREADS", "-x", "OMP_DYNAMIC",
               "-x", "OPENBLAS_NUM_THREADS", "-x", "MKL_NUM_THREADS",
               "/usr/local/bin/python", "-m", "easygraph.distributed.server",
               "--bind", "0.0.0.0", "--port", "8787", "--session-id", name,
               "--data-root", "/shared", "--idle-timeout", str(idle_timeout)]
    container = {"name": "mpi", "image": image, "imagePullPolicy": "IfNotPresent",
        "command": ["bash", "/usr/local/bin/start-easygraph-worker"],
        "env": [{"name": "OMP_NUM_THREADS", "value": str(threads)},
                {"name": "OMP_DYNAMIC", "value": "FALSE"}] + [
                {"name": key, "value": "1"} for key in
                ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")] + [
            {"name": "EG_NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
            {"name": "EASYGRAPH_SESSION_TOKEN", "valueFrom": {"secretKeyRef": {
                "name": name + "-auth", "key": "token"}}}],
        "volumeMounts": [{"name": "data", "mountPath": "/shared"}],
        "resources": {"requests": {"cpu": str(slots * threads), "memory": memory},
                      "limits": {"cpu": str(slots * threads), "memory": memory}},
        "readinessProbe": {"tcpSocket": {"port": 2222}, "periodSeconds": 3},
    }
    pod = {"securityContext": {"runAsUser": 20001, "runAsGroup": 4322, "runAsNonRoot": True},
           "nodeSelector": node_selector,
           "containers": [container], "volumes": [{"name": "data",
               "persistentVolumeClaim": {"claimName": pvc}}]}
    worker = copy.deepcopy(pod)
    worker["affinity"] = {"podAntiAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": [{
        "labelSelector": {"matchLabels": {"easygraph-session": name, "easygraph-role": "worker"}},
        "topologyKey": "kubernetes.io/hostname"}]}}
    if worker_nodes is not None:
        worker["affinity"]["nodeAffinity"] = {"requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [{"matchExpressions": [{"key": "kubernetes.io/hostname",
                "operator": "In", "values": list(worker_nodes)}]}]}}
    launcher = copy.deepcopy(pod)
    if launcher_node is not None:
        launcher["nodeSelector"] = {"kubernetes.io/hostname": launcher_node}
    if launcher_tolerations:
        launcher["tolerations"] = copy.deepcopy(launcher_tolerations)
    c = launcher["containers"][0]
    c["command"] = command
    c.pop("readinessProbe")
    c["resources"] = {"requests": {"cpu": "250m", "memory": "256Mi"},
                      "limits": {"cpu": "1", "memory": memory}}
    job = {"apiVersion": "kubeflow.org/v2beta1", "kind": "MPIJob",
        "metadata": {"name": name, "namespace": namespace,
                     "labels": {"app.kubernetes.io/name": "easygraph-session"}},
        "spec": {"slotsPerWorker": slots, "mpiImplementation": "OpenMPI",
                 "sshAuthMountPath": "/home/mpiuser/.ssh",
                 "launcherCreationPolicy": "WaitForWorkersReady", "runLauncherAsWorker": False,
                 "runPolicy": {"cleanPodPolicy": "Running", "backoffLimit": 0,
                               "ttlSecondsAfterFinished": 300},
                 "mpiReplicaSpecs": {
                     "Launcher": {"replicas": 1, "restartPolicy": "Never", "template": {
                         "metadata": {"labels": {"easygraph-session": name, "easygraph-role": "launcher"}},
                         "spec": launcher}},
                     "Worker": {"replicas": workers, "restartPolicy": "Never", "template": {
                         "metadata": {"labels": {"easygraph-session": name, "easygraph-role": "worker"}},
                         "spec": worker}}}}}
    secret = {"apiVersion": "v1", "kind": "Secret",
              "metadata": {"name": name + "-auth", "namespace": namespace},
              "type": "Opaque", "stringData": {"token": token}}
    return secret, job
