import pytest

import easygraph as eg


def _require_gpu_katz():
    cpp_easygraph = pytest.importorskip("cpp_easygraph")
    if not hasattr(eg, "DiGraphC") or not hasattr(cpp_easygraph, "cpp_gpu_katz_centrality"):
        pytest.skip("EasyGraph was not built with EGGPU Katz support")


def _graph():
    graph = eg.DiGraphC()
    graph.add_nodes([10, 20, 30, 99])
    graph.add_edges([(10, 20), (20, 30)])
    return graph


def test_gpu_katz_uses_incoming_edges_and_preserves_labels():
    _require_gpu_katz()
    scores = eg.gpu_katz_centrality(
        _graph(), alpha=0.1, beta=1.0, max_iter=10000, tol=1e-10, normalized=False
    )

    assert set(scores) == {10, 20, 30, 99}
    assert scores[30] > scores[20] > scores[10]
    assert scores[10] == pytest.approx(scores[99], abs=1e-12)


def test_gpu_katz_rejects_the_max_in_degree_bound():
    _require_gpu_katz()
    with pytest.raises(ValueError, match="alpha"):
        eg.gpu_katz_centrality(_graph(), alpha=1.0, normalized=False)


def test_prepared_gpu_katz_matches_the_one_shot_snapshot():
    _require_gpu_katz()
    graph = _graph()
    expected = eg.gpu_katz_centrality(
        graph, alpha=0.1, beta=1.0, max_iter=10000, tol=1e-10, normalized=False
    )
    context = eg.prepare_gpu_katz(graph)

    graph.add_edges([(10, 30)])
    observed = context.run(
        alpha=0.1, beta=1.0, max_iter=10000, tol=1e-10, normalized=False
    )
    assert set(observed) == set(expected)
    for node in expected:
        assert observed[node] == pytest.approx(expected[node], abs=1e-12)

    context.close()
    with pytest.raises(eg.EasyGraphError, match="closed"):
        context.run()


def test_gpu_katz_tsv_dataset_preserves_labels_and_reuses_prepared_csr(tmp_path):
    _require_gpu_katz()
    arcs_path = tmp_path / "arcs.tsv"
    node_map_path = tmp_path / "node_map.tsv"
    arcs_path.write_text("0\t1\n1\t2\n", encoding="ascii")
    node_map_path.write_text(
        "new_id\toriginal_id\n0\t10\n1\t20\n2\t30\n3\t99\n", encoding="ascii"
    )

    dataset = eg.load_gpu_katz_tsv_dataset(arcs_path, node_map_path)
    assert (dataset.node_count, dataset.edge_count, dataset.max_in_degree) == (4, 2, 1)

    graph = dataset.to_digraphc()
    assert graph.cflag == 1
    expected = eg.gpu_katz_centrality(
        graph,
        alpha=0.1,
        beta=1.0,
        max_iter=10000,
        tol=1e-10,
        normalized=False,
    )
    context = dataset.prepare()
    observed = context.run(
        alpha=0.1, beta=1.0, max_iter=10000, tol=1e-10, normalized=False
    )
    assert set(expected) == {10, 20, 30, 99}
    assert set(observed) == set(expected)
    for node in expected:
        assert observed[node] == pytest.approx(expected[node], abs=1e-12)

    dataset.close()
    repeated = context.run(
        alpha=0.1, beta=1.0, max_iter=10000, tol=1e-10, normalized=False
    )
    assert repeated == pytest.approx(observed, abs=1e-12)


def test_gpu_katz_tsv_dataset_rejects_invalid_node_map_header(tmp_path):
    _require_gpu_katz()
    arcs_path = tmp_path / "arcs.tsv"
    node_map_path = tmp_path / "node_map.tsv"
    arcs_path.write_text("", encoding="ascii")
    node_map_path.write_text("node\toriginal_id\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="new_id"):
        eg.load_gpu_katz_tsv_dataset(arcs_path, node_map_path)
