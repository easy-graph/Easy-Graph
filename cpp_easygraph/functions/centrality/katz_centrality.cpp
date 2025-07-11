#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "centrality.h"
#include "../../classes/graph.h"

namespace py = pybind11;

py::object cpp_katz_centrality(
    py::object G,
    py::object py_alpha,
    py::object py_beta,
    py::object py_max_iter,
    py::object py_tol,
    py::object py_normalized
) {
    try {
        Graph& graph = G.cast<Graph&>();
        auto csr = graph.gen_CSR();
        int n = csr->nodes.size();

        if (n == 0) {
            return py::dict();
        }

        // Initialize vectors
        std::vector<double> x0(n, 1.0);
        std::vector<double> x1(n);
        std::vector<double>* x_prev = &x0;
        std::vector<double>* x_next = &x1;

        // Process beta parameter
        std::vector<double> b(n);
        if (py::isinstance<py::float_>(py_beta) || py::isinstance<py::int_>(py_beta)) {
            double beta_val = py_beta.cast<double>();
            for (int i = 0; i < n; i++) {
                b[i] = beta_val;
            }
        } else if (py::isinstance<py::dict>(py_beta)) {
            py::dict beta_dict = py_beta.cast<py::dict>();
            for (int i = 0; i < n; i++) {
                node_t internal_id = csr->nodes[i];
                py::object node_obj = graph.id_to_node[py::cast(internal_id)];
                if (beta_dict.contains(node_obj)) {
                    b[i] = beta_dict[node_obj].cast<double>();
                } else {
                    b[i] = 1.0;
                }
            }
        } else {
            throw py::type_error("beta must be a float or a dict");
        }

        // Extract parameters
        double alpha = py_alpha.cast<double>();
        int max_iter = py_max_iter.cast<int>();
        double tol = py_tol.cast<double>();
        bool normalized = py_normalized.cast<bool>();

        // Iterative updates
        int iter = 0;
        for (; iter < max_iter; iter++) {
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                int start = csr->V[i];
                int end = csr->V[i + 1];
                for (int jj = start; jj < end; jj++) {
                    int j = csr->E[jj];
                    sum += (*x_prev)[j];
                }
                (*x_next)[i] = alpha * sum + b[i];
            }

            // Check convergence
            double change = 0.0;
            for (int i = 0; i < n; i++) {
                change += std::abs((*x_next)[i] - (*x_prev)[i]);
            }

            if (change < tol) {
                break;
            }

            std::swap(x_prev, x_next);
        }

        // Handle convergence failure
        if (iter == max_iter) {
            throw std::runtime_error("Katz centrality failed to converge in " + std::to_string(max_iter) + " iterations");
        }

        // Normalization
        std::vector<double>& x_final = *x_next;
        if (normalized) {
            double norm = 0.0;
            for (double val : x_final) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < n; i++) {
                    x_final[i] /= norm;
                }
            }
        }

        // Prepare results
        py::dict result;
        for (int i = 0; i < n; i++) {
            node_t internal_id = csr->nodes[i];
            py::object node_obj = graph.id_to_node[py::cast(internal_id)];
            result[node_obj] = x_final[i];
        }

        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}