#include <cstddef>
#include <memory>
#include <vector>

#include "./common/err.h"

namespace gpu_easygraph {

int closeness_centrality(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    std::vector<double>& CC
);



int betweenness_centrality(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    bool is_directed,
    bool normalized,
    bool endpoints,
    std::vector<double>& BC
);



int k_core(
    const std::vector<int>& V,
    const std::vector<int>& E,
    std::vector<int>& KC
);



int sssp_dijkstra(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<double>& W,
    const std::vector<int>& sources,
    int target,
    std::vector<double>& res
);



int pagerank(
    const std::vector<int>& V,
    const std::vector<int>& E,
    double alpha,
    int max_iter_num,
    double threshold,
    std::vector<double>& PR
);

enum class KatzStatus {
    success = 0,
    invalid_input,
    not_converged,
    cuda_error,
};

struct KatzResult {
    KatzStatus status;
    int iterations;
    double final_residual;
};

// Explicit owner for repeated deterministic Katz queries over one immutable
// incoming-CSR snapshot. This object is independent of Graph lifetime and does
// not alter Graph-level CSR behavior.
class PreparedKatzContext {
public:
    static std::unique_ptr<PreparedKatzContext> create(
        const std::vector<int>& incoming_offsets,
        const std::vector<int>& incoming_columns,
        KatzStatus& status
    );

    ~PreparedKatzContext();

    PreparedKatzContext(const PreparedKatzContext&) = delete;
    PreparedKatzContext& operator=(const PreparedKatzContext&) = delete;

    KatzResult run(
        double alpha,
        double beta,
        int max_iter,
        double epsilon,
        bool normalized,
        std::vector<double>& scores
    );

    int nodes() const { return nodes_; }

private:
    PreparedKatzContext() = default;

    int nodes_ = 0;
    int max_in_degree_ = 0;
    int* d_offsets_ = nullptr;
    int* d_columns_ = nullptr;
    double* d_current_ = nullptr;
    double* d_next_ = nullptr;
    double* d_values_ = nullptr;
    double* d_reduce_result_ = nullptr;
    void* d_reduce_temp_ = nullptr;
    size_t reduce_temp_bytes_ = 0;
};

// Deterministic incoming-Katz solver. A successful result means the L1
// difference between consecutive score vectors is below epsilon.
KatzResult katz_centrality(
    const std::vector<int>& incoming_offsets,
    const std::vector<int>& incoming_columns,
    double alpha,
    double beta,
    int max_iter,
    double epsilon,
    bool normalized,
    std::vector<double>& scores
);



int constraint(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask,
    std::vector<double>& constraint
);



int hierarchy(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& hierarchy
);



int effective_size(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& row,
    const std::vector<int>& col,
    int num_nodes,
    const std::vector<double>& W,
    bool is_directed,
    std::vector<int>& node_mask, 
    std::vector<double>& effective_size
);

} // namespace gpu_easygraph
