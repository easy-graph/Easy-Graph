"""基准测试 – RoadNetCA 数据集 | cpp_easygraph / Python-EasyGraph / NetworkX

运行示例:
  python test_eg.py                        # 默认 (5 进程 × 3 次采样，含 1 次热身)
  python test_eg.py --fast                 # 快速模式（2 进程 × 3 次，热身 1 次）
  python test_eg.py -o results_eg.json     # 保存结果供 compare 使用
  python test_eg.py -v                     # 详细输出每次原始耗时
  pyperf compare_to results_eg.json results_ig.json results_gt.json

说明:
  - 图构建在计时之外（模块级），每个 worker 进程独立初始化，保证进程隔离。
  - pyperf 默认在独立子进程中采样，自动处理热身、多次重复与统计（中位数 ± 标准差）。
  - 若需固定线程数以公平对比多库，建议: OMP_NUM_THREADS=1 python test_eg.py
"""
import sys
import pyperf
import cpp_easygraph
import easygraph as eg
import networkx as nx

from easygraph.datasets import RoadNetCADataset
# from easygraph.datasets import WikiTopCatsDataset  # directed

# ── 数据集 & 图构建（在计时外，每个 worker 进程独立初始化）────────────────────
data_eg = RoadNetCADataset()
G_py = data_eg[0]                          # Python-EasyGraph 图对象

G_cpp = cpp_easygraph.Graph()              # CPP-EasyGraph 图对象
G_cpp.add_edges_from(G_py.edges)

G_nx = nx.Graph()                          # NetworkX 图对象
G_nx.add_edges_from(G_py.edges)

# G_ig = igraph.Graph(edges=[(u,v) for u,v,*_ in G_py.edges])  # igraph
# 构建 graph_tool 图对象见 test_gt.py


# ── 基准函数（纯计算，无任何 I/O 或初始化）─────────────────────────────────────
def bench_cpp_clustering():
    return cpp_easygraph.cpp_clustering_array(G_cpp)


# def bench_py_eg():
#     return eg.average_clustering(G_py)


# def bench_nx():
#     return nx.average_clustering(G_nx)


# ── 线程信息（仅主进程打印，worker 子进程跳过）──────────────────────────────────
def _print_thread_info():
    import os
    import multiprocessing
    import ctypes
    print("\n[Thread Info]")
    print(f"  OMP_NUM_THREADS  = {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"  CPU cores        = {multiprocessing.cpu_count()}")
    for libname in ("libgomp.so.1", "libgomp.so", "libiomp5.so", "libiomp5.so.5"):
        try:
            lib = ctypes.CDLL(libname)
            if hasattr(lib, 'omp_get_max_threads'):
                lib.omp_get_max_threads.restype = ctypes.c_int
                print(f"  {libname} omp_get_max_threads() = {lib.omp_get_max_threads()}")
                break
        except Exception:
            continue
    print()


if __name__ == "__main__":
    # '--worker' 由 pyperf 在子进程中注入，主进程不含该标志
    if '--worker' not in sys.argv:
        print(f"RoadNetCA: {G_cpp.number_of_nodes()} nodes, {G_cpp.number_of_edges()} edges")
        _print_thread_info()

    runner = pyperf.Runner(processes=5)
    runner.bench_func('CPP-EasyGraph: cpp_clustering', bench_cpp_clustering)
    # runner.bench_func('Python-EasyGraph: average_clustering', bench_py_eg)
    # runner.bench_func('NetworkX: average_clustering', bench_nx)