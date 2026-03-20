// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2025 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// you should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef GRAPH_CLUSTERING_HH
#define GRAPH_CLUSTERING_HH

#include "config.h"

#include "idx_map.hh"
#include "samplers.hh"

#ifdef _OPENMP
#include "omp.h"
#endif

#ifndef __clang__
#include <ext/numeric>
using __gnu_cxx::power;
#else
template <class Value>
Value power(Value value, int n)
{
    return pow(value, n);
}
#endif

namespace graph_tool
{
using namespace boost;
using namespace std;

// calculates the number of triangles to which v belongs
template <class Graph, class EWeight, class VProp>
std::pair<typename property_traits<EWeight>::value_type,
          typename property_traits<EWeight>::value_type>
get_triangles(typename graph_traits<Graph>::vertex_descriptor v,
              EWeight& eweight, VProp& mark, const Graph& g)
{
    typedef typename property_traits<EWeight>::value_type val_t;
    val_t triangles = 0, k = 0, k2 = 0;

    if (out_degree(v, g) > 1)
    {
        for (auto e : out_edges_range(v, g))
        {
            auto u = target(e, g);
            if (u == v)
                continue;
            auto w = eweight[e];
            mark[u] = w;
            k += w;
            k2 += w * w;
        }

        for (auto e : out_edges_range(v, g))
        {
            auto u = target(e, g);
            if (u == v)
                continue;
            val_t t = 0;
            for (auto e2 : out_edges_range(u, g))
            {
                auto w = target(e2, g);
                auto mw = mark[w];
                if (mw > 0 && w != u)
                    t += mw * eweight[e2];
            }
            triangles += t * eweight[e];
        }

        for (auto u : adjacent_vertices_range(v, g))
            mark[u] = 0;
    }

    if (is_directed(g))
        return {triangles, k * k - k2};
    else
        return {triangles / 2, (k * k - k2) / 2};
}


// retrieves the global clustering coefficient
template <class Graph, class EWeight>
auto get_global_clustering(const Graph& g, EWeight eweight)
{
    typedef typename property_traits<EWeight>::value_type val_t;
    val_t triangles = 0, n = 0;
    vector<val_t> mark(num_vertices(g), 0);
    vector<std::pair<val_t, val_t>> ret(num_vertices(g));

    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
        firstprivate(mark) reduction(+:triangles, n)
    parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 auto temp = get_triangles(v, eweight, mark, g);
                 triangles += temp.first;
                 n += temp.second;
                 ret[v] = temp;
             });
    double c = double(triangles) / n;

    // "jackknife" variance
    double c_err = 0.0;
    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh())   \
        reduction(+:c_err)
    parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 auto cl = double(triangles - ret[v].first) /
                     (n - ret[v].second);
                 c_err += power(c - cl, 2);
             });

    c_err = sqrt(c_err);
    return std::make_tuple(c, c_err, triangles/3, n);
}

// sets the local clustering coefficient to a property
template <class Graph, class EWeight, class ClustMap>
void set_clustering_to_property(const Graph& g, EWeight eweight,
                                ClustMap clust_map)
{
    typedef typename property_traits<EWeight>::value_type val_t;
    typedef typename property_traits<ClustMap>::value_type cval_t;
    vector<val_t> mark(num_vertices(g), false);

    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
        firstprivate(mark)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             auto triangles = get_triangles(v, eweight, mark, g);
             cval_t clustering = (triangles.second > 0) ?
                 cval_t(triangles.first)/triangles.second :
                 0.0;
             clust_map[v] = clustering;
         });
}

template <class Graph, class RNG>
auto get_global_clustering_sampled(const Graph& g, size_t m, RNG& rng)
{
    std::vector<size_t> vs;
    std::vector<double> probs;
    idx_set<size_t, false, false> us(num_vertices(g));

    for (auto v : vertices_range(g))
    {
        us.clear();
        for (auto u : out_neighbors_range(v, g))
            us.insert(u);
        auto k = us.size();
        vs.push_back(v);
        if (is_directed(g))
            probs.push_back(k * (k - 1));
        else
            probs.push_back((k * (k - 1))/2);
    }

    Sampler<size_t> sampler(vs, probs);

    size_t ts = 0;
    for (size_t i = 0; i < m; ++i)
    {
        auto v = sampler(rng);
        us.clear();
        for (auto u : out_neighbors_range(v, g))
            us.insert(u);
        auto u = uniform_sample(us, rng);
        us.erase(u);
        auto w = uniform_sample(us, rng);
        if (edge(u, w, g).second)
            ts++;
    }
    return ts / double(m);
}


} //graph-tool namespace

#endif // GRAPH_CLUSTERING_HH
