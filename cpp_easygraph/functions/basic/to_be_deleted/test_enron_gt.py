"""基准测试 – web-NotreDame 数据集 | graph_tool

运行示例:
  python test_enron_gt.py                        # 默认 (5 进程 × 3 次采样，含 1 次热身)
  python test_enron_gt.py --fast                 # 快速模式
  python test_enron_gt.py -o results_enron_gt.json
  pyperf compare_to results_enron_eg.json results_enron_ig.json results_enron_gt.json

说明:
  - 图构建在计时之外（模块级），每个 worker 进程独立初始化，保证进程隔离。
  - graph_tool 使用连续顶点 ID，自动去重自环与重边。
  - pyperf 自动处理热身、多次重复与统计（中位数 ± 标准差）。
  - 若需固定线程数以公平对比多库，建议: OMP_NUM_THREADS=1 python test_enron_gt.py
"""
import os
import sys
import pyperf

try:
    import graph_tool.all as gt
except ImportError:
    raise ImportError(
        "graph_tool 未安装。请参考 https://graph-tool.skewed.de/ 安装后再运行。"
    )

# ── 数据集 & 图构建（在计时外，每个 worker 进程独立初始化）────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Email-Enron.txt')

def _load_edges(path):
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            u, v = line.split()
            edges.append((int(u), int(v)))
    return edges

edges = _load_edges(DATA_PATH)

# graph_tool 使用连续顶点 ID；去重自环与重边（将有向边视为无向边）
G_gt = gt.Graph(directed=False)
vertex_map = {}
seen_edges = set()
for u, v in edges:
    if u == v:
        continue
    key = (min(u, v), max(u, v))
    if key in seen_edges:
        continue
    seen_edges.add(key)
    if u not in vertex_map:
        vertex_map[u] = G_gt.add_vertex()
    if v not in vertex_map:
        vertex_map[v] = G_gt.add_vertex()
    G_gt.add_edge(vertex_map[u], vertex_map[v])
del seen_edges, vertex_map, edges


# ── 基准函数 ────────────────────────────────────────────────────────────────
def bench_local_clustering():
    # 仅计算 PropertyMap（不求均值），衡量纯计算耗时
    prop = G_gt.new_vertex_property("double")
    gt.local_clustering(G_gt, prop=prop)
    return prop


# ── 线程信息（仅主进程打印）──────────────────────────────────────────────────
def _print_thread_info():
    import multiprocessing
    print("\n[Thread Info]")
    print(f"  OMP_NUM_THREADS       = {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"  CPU cores             = {multiprocessing.cpu_count()}")
    print(f"  graph_tool OMP threads = {gt.openmp_get_num_threads()}")
    print()


if __name__ == "__main__":
    if '--worker' not in sys.argv:
        print(f"web-NotreDame: {G_gt.num_vertices()} nodes, {G_gt.num_edges()} edges")
        _print_thread_info()

    runner = pyperf.Runner(processes=5)
    runner.bench_func('graph_tool: local_clustering', bench_local_clustering)
