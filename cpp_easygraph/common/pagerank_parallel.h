#pragma once

#include <cstddef>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace easygraph_parallel {
inline void configure(int threads) {
    if (threads < 1) throw std::invalid_argument("OpenMP thread count must be positive");
    #ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    #else
    if (threads != 1) throw std::runtime_error("OpenMP is unavailable in this build");
    #endif
}

template <typename Function>
void for_each(std::size_t count, Function function) {
    #pragma omp parallel for schedule(dynamic, 128)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) function(i);
}

template <typename Function>
double sum(std::size_t count, Function function) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) result += function(i);
    return result;
}

inline void add(double& target, double value) {
    #pragma omp atomic
    target += value;
}

inline bool enabled() {
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif
}

inline int max_threads() {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return 1;
    #endif
}

inline int team_size() {
    int result = 1;
    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        result = omp_get_num_threads();
    }
    #endif
    return result;
}
}
