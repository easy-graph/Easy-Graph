"""基准测试 – RoadNetCA 数据集 | graph_tool

运行示例:
  python test_gt.py                        # 默认 (5 进程 × 3 次采样，含 1 次热身)
  python test_gt.py --fast                 # 快速模式 (2 进程 × 3 次)
  python test_gt.py -o results_gt.json     # 保存结果供 compare 使用
  python test_gt.py -v                     # 详细输出每次原始耗时
  pyperf compare_to results_eg.json results_ig.json results_gt.json
说明:
  - 图构建在计时之外（模块级），每个 worker 进程独立初始化，保证进程隔离。
  - graph_tool 使用连续顶点 ID；自动去重自环与重边（将有向边视为无向边）。
  - pyperf 自动处理热身、多次重复与统计（中位数 ± 标准差）。
  - 若需固定线程数以公平对比多库，建议: OMP_NUM_THREADS=1 python test_gt.py
"""
import sys

try:
    import graph_tool.all as gt
except ImportError:
    raise ImportError(
        "graph_tool 未安装。请参考 https://graph-tool.skewed.de/ 安装后再运行。"
    )

import pyperf
from easygraph.datasets import AmazonCoBuyComputerDataset

# ── 数据集 & 图构建（在计时外，每个 worker 进程独立初始化）────────────────────
data_eg = AmazonCoBuyComputerDataset()
edges_raw = [(u, v) for u, v, *_ in data_eg[0].edges]

# graph_tool 使用连续顶点 ID；去重自环与重边（无向图）
G_gt = gt.Graph(directed=False)
_vertex_map = {}
_seen_edges = set()
for _u, _v in edges_raw:
    if _u == _v:
        continue
    _key = (min(_u, _v), max(_u, _v))
    if _key in _seen_edges:
        continue
    _seen_edges.add(_key)
    if _u not in _vertex_map:
        _vertex_map[_u] = G_gt.add_vertex()
    if _v not in _vertex_map:
        _vertex_map[_v] = G_gt.add_vertex()
    G_gt.add_edge(_vertex_map[_u], _vertex_map[_v])
del _vertex_map, _seen_edges, edges_raw, _u, _v, _key


# ── 基准函数（纯计算，无 I/O 或初始化）──────────────────────────────────────
def bench_local_clustering():
    # 仅计算 PropertyMap（不求均值），衡量纯计算耗时
    prop = G_gt.new_vertex_property("double")
    gt.local_clustering(G_gt, prop=prop)
    return prop


# ── 线程信息（仅主进程打印，worker 子进程跳过）──────────────────────────────
def _print_thread_info():
    import os
    import multiprocessing
    print("\n[Thread Info]")
    print(f"  OMP_NUM_THREADS       = {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"  CPU cores             = {multiprocessing.cpu_count()}")
    print(f"  graph_tool OMP threads = {gt.openmp_get_num_threads()}")
    print()


if __name__ == "__main__":
    # '--worker' 由 pyperf 在子进程中注入，主进程不含该标志
    if '--worker' not in sys.argv:
        print(f"RoadNetCA: {G_gt.num_vertices()} nodes, {G_gt.num_edges()} edges")
        _print_thread_info()

    runner = pyperf.Runner(processes=5)
    runner.bench_func('graph_tool: local_clustering', bench_local_clustering)
