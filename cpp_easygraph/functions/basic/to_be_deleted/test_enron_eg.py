"""基准测试 – web-NotreDame 数据集 | cpp_easygraph

运行示例:
  python test_enron_eg.py                        # 默认 (5 进程 × 3 次采样，含 1 次热身)
  python test_enron_eg.py --fast                 # 快速模式
  python test_enron_eg.py -o results_enron_eg.json
  pyperf compare_to results_enron_eg.json results_enron_ig.json results_enron_gt.json

说明:
  - 图构建在计时之外（模块级），每个 worker 进程独立初始化，保证进程隔离。
  - pyperf 自动处理热身、多次重复与统计（中位数 ± 标准差）。
  - 若需固定线程数以公平对比多库，建议: OMP_NUM_THREADS=1 python test_enron_eg.py
"""
import os
import sys
import pyperf
import cpp_easygraph

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

G_cpp = cpp_easygraph.Graph()
G_cpp.add_edges_from(edges)


# ── 基准函数 ────────────────────────────────────────────────────────────────
def bench_cpp_clustering():
    return cpp_easygraph.cpp_clustering(G_cpp)


# ── 线程信息（仅主进程打印）──────────────────────────────────────────────────
def _print_thread_info():
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
    if '--worker' not in sys.argv:
        print(f"web-NotreDame: {G_cpp.number_of_nodes()} nodes, {G_cpp.number_of_edges()} edges")
        _print_thread_info()

    runner = pyperf.Runner(processes=5)
    runner.bench_func('CPP-EasyGraph: cpp_clustering', bench_cpp_clustering)
