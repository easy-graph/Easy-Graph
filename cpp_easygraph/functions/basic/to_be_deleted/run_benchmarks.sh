#!/usr/bin/env bash
# run_benchmarks.sh – 基准测试：16 线程，打印工具/数据集/速度信息
#
# 用法：
#   cd cpp_easygraph/functions/basic/test
#   bash run_benchmarks.sh
#   bash run_benchmarks.sh --fast          # 快速模式（透传给 pyperf）
#   bash run_benchmarks.sh --skip-gt       # 跳过 graph_tool

set -euo pipefail

THREADS=16
SKIP_GT=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-gt) SKIP_GT=1; shift ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── 线程环境变量（统一设为 16）────────────────────────────────────────────────
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export TBB_NUM_THREADS="${THREADS}"

# ── CPU 绑定（0-15，不足时降级）──────────────────────────────────────────────
AVAIL=$(nproc --all)
CPU_END=$(( THREADS < AVAIL ? THREADS - 1 : AVAIL - 1 ))
TASKSET="taskset -c 0-${CPU_END}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULT_DIR}"

echo "================================================================"
echo " Threads : ${THREADS}  (OMP/MKL/OPENBLAS/TBB)"
echo " CPUs    : 0-${CPU_END}"
echo "================================================================"

# ── 运行单个基准 ──────────────────────────────────────────────────────────────
run_bench() {
    local dataset="$1" tool="$2" script="$3" out="$4"
    echo ""
    echo "  Dataset : ${dataset}"
    echo "  Tool    : ${tool}"
    echo "  Threads : ${THREADS}"
    if ${TASKSET} python "${script}" -o "${out}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1; then
        echo "  Result  ↓"
        python -m pyperf show "${out}" | sed 's/^/    /'
    else
        echo "  [FAIL] 跳过"
        [[ -f "${out}" ]] && rm -f "${out}"
    fi
    echo "----------------------------------------------------------------"
}

# ── 对比两个或多个结果 ────────────────────────────────────────────────────────
compare() {
    local title="$1"; shift
    local jsons=()
    for f in "$@"; do [[ -f "$f" ]] && jsons+=("$f"); done
    (( ${#jsons[@]} >= 2 )) || return
    echo ""
    echo "  [COMPARE] ${title}"
    python -m pyperf compare_to "${jsons[@]}" | sed 's/^/    /'
    echo ""
}


# ══ Dataset A: RoadNetCA ════════════════════════════════════════════════════
echo ""
echo "════════════════════  Dataset A: RoadNetCA  ════════════════════"

A_EG="${RESULT_DIR}/roadnetca_eg.json"
A_IG="${RESULT_DIR}/roadnetca_ig.json"
A_GT="${RESULT_DIR}/roadnetca_gt.json"

run_bench "RoadNetCA" "cpp_easygraph" "${SCRIPT_DIR}/test_eg.py" "${A_EG}"
run_bench "RoadNetCA" "igraph"        "${SCRIPT_DIR}/test_ig.py" "${A_IG}"
(( SKIP_GT == 0 )) && run_bench "RoadNetCA" "graph_tool" "${SCRIPT_DIR}/test_gt.py" "${A_GT}"

compare "RoadNetCA" "${A_EG}" "${A_IG}" "${A_GT}"


# ══ Dataset B: web-NotreDame ════════════════════════════════════════════════
echo ""
echo "════════════════════  Dataset B: web-NotreDame  ════════════════"

B_EG="${RESULT_DIR}/notredame_eg.json"
B_IG="${RESULT_DIR}/notredame_ig.json"
B_GT="${RESULT_DIR}/notredame_gt.json"

run_bench "web-NotreDame" "cpp_easygraph" "${SCRIPT_DIR}/test_enron_eg.py" "${B_EG}"
run_bench "web-NotreDame" "igraph"        "${SCRIPT_DIR}/test_enron_ig.py" "${B_IG}"
(( SKIP_GT == 0 )) && run_bench "web-NotreDame" "graph_tool" "${SCRIPT_DIR}/test_enron_gt.py" "${B_GT}"

compare "web-NotreDame" "${B_EG}" "${B_IG}" "${B_GT}"


# ══ Dataset C: soc-Epinions1 ════════════════════════════════════════════════
echo ""
echo "════════════════════  Dataset C: soc-Epinions1  ════════════════"

C_EG="${RESULT_DIR}/epinions_eg.json"

run_bench "soc-Epinions1" "cpp_easygraph" "${SCRIPT_DIR}/test_epinions_eg.py" "${C_EG}"


# ══ 汇总 ════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo " 结果目录: ${RESULT_DIR}/"
ls -1 "${RESULT_DIR}"/*.json 2>/dev/null | sed 's/^/  /' || true
echo "================================================================"
