from easygraph.distributed._kubernetes import resources


def test_session_worker_and_control_plane_launcher_contract():
    tolerations = [{"key": "node-role.kubernetes.io/control-plane",
                    "operator": "Exists", "effect": "NoSchedule"}]
    _, job = resources(
        "eg-session-test", "not-a-real-token", image="example/worker:1",
        namespace="benchmark", workers=2, slots=1, threads=16,
        memory="60Gi", pvc="benchmark-data-rwx", idle_timeout=900,
        node_selector={"benchmark-node": "true"},
        worker_nodes=["eg-b", "eg-c"], launcher_node="eg-a",
        launcher_tolerations=tolerations,
    )
    replicas = job["spec"]["mpiReplicaSpecs"]
    launcher = replicas["Launcher"]["template"]["spec"]
    worker = replicas["Worker"]["template"]["spec"]
    assert launcher["nodeSelector"] == {"kubernetes.io/hostname": "eg-a"}
    assert launcher["tolerations"] == tolerations
    assert "tolerations" not in worker
    assert worker["containers"][0]["resources"]["limits"] == {
        "cpu": "16", "memory": "60Gi"}
    assert worker["containers"][0]["image"] == "example/worker:1"
