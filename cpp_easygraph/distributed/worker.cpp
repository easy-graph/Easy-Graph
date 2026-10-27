#include "worker.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

#include "../common/pagerank_parallel.h"

namespace egdist {
PageRankResult PageRankWorker::run(double alpha, int max_iter, double tol, bool weighted) const {
    const auto& g = graph_;
    const auto& comm = *g.comm;
    comm.check_error(!std::isfinite(alpha) || alpha < 0 || alpha >= 1 ? "alpha must be in [0, 1)" :
                     max_iter <= 0 ? "max_iter must be positive" :
                     !std::isfinite(tol) || tol <= 0 ? "tol must be finite and positive" : "");
    std::ostringstream params;
    params << std::setprecision(17) << alpha << ',' << max_iter << ',' << tol << ',' << weighted;
    comm.agree(params.str());
    const auto n = g.global_num_vertices;
    if (!n) return {g.vertices, {}, 0};
    std::vector<double> old(g.vertices.size(), 1.0 / n), next(old.size()), normalizer(old.size());
    easygraph_parallel::for_each(old.size(), [&](std::size_t i) {
        if (!weighted) normalizer[i] = g.offsets[i + 1] - g.offsets[i];
        else for (auto e = g.offsets[i]; e < g.offsets[i + 1]; ++e) normalizer[i] += g.weights[e];
    });
    bool invalid_sum = false;
    for (auto sum : normalizer) invalid_sum = invalid_sum || !std::isfinite(sum);
    comm.check_error(invalid_sum ? "Outgoing weight sum overflowed" : "");
    std::vector<double> contributions(g.contribution_slots);
    std::vector<std::vector<Update>> buckets(comm.size());
    std::vector<double*> outgoing;
    outgoing.reserve(g.contribution_slots - old.size());
    for (std::size_t owner = 0; owner < buckets.size(); ++owner) {
        for (auto vertex : g.remote_targets[owner]) buckets[owner].push_back({vertex, 0.0});
        for (auto& update : buckets[owner]) outgoing.push_back(&update.value);
    }
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        easygraph_parallel::for_each(contributions.size(), [&](std::size_t i) {
            contributions[i] = 0.0;
        });
        const double local_dangling = easygraph_parallel::sum(old.size(), [&](std::size_t i) {
            return normalizer[i] == 0 ? old[i] : 0.0;
        });
        easygraph_parallel::for_each(old.size(), [&](std::size_t i) {
            if (normalizer[i] == 0) return;
            const double scale = weighted ? alpha * old[i] : alpha * old[i] / normalizer[i];
            for (auto e = g.offsets[i]; e < g.offsets[i + 1]; ++e) {
                const double value = weighted ? scale * (g.weights[e] / normalizer[i]) : scale;
                easygraph_parallel::add(contributions[g.target_slots[e]], value);
            }
        });
        easygraph_parallel::for_each(outgoing.size(), [&](std::size_t i) {
            *outgoing[i] = contributions[old.size() + i];
        });
        const auto received = comm.exchange(buckets);
        easygraph_parallel::for_each(received.size(), [&](std::size_t i) {
            const auto& update = received[i];
            easygraph_parallel::add(contributions[g.local_index.at(update.vertex)], update.value);
        });
        const double jump = (1.0 - alpha + alpha * comm.sum(local_dangling)) / n;
        const double local_error = easygraph_parallel::sum(old.size(), [&](std::size_t i) {
            next[i] = contributions[i] + jump;
            return std::abs(next[i] - old[i]);
        });
        old.swap(next);
        if (comm.sum(local_error) < tol * n) return {g.vertices, std::move(old), iteration};
    }
    throw std::runtime_error("Distributed PageRank did not converge within max_iter");
}
}
