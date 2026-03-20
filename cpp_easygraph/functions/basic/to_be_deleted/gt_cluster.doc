// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2025 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "graph_filtering.hh"

#include "graph.hh"
#include "graph_selectors.hh"
#include "graph_properties.hh"
#include "graph_filtering.hh"

#include "graph_clustering.hh"

#include "random.hh"

#define __MOD__ clustering
#define DEF_REGISTRY
#include "module_registry.hh"

#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

boost::python::tuple global_clustering(GraphInterface& g, std::any weight)
{
    typedef UnityPropertyMap<size_t,GraphInterface::edge_t> weight_map_t;
    auto weight_props = hana::append(edge_scalar_properties,
                                     hana::type<weight_map_t>());

    if (weight.has_value() && !belongs(edge_scalar_properties, weight))
        throw ValueException("weight edge property must have a scalar value type");

    if (!weight.has_value())
        weight = weight_map_t();

    boost::python::tuple oret;
    run_action<decltype(graph_tool::never_directed)>(false)
        (g,
         [&](auto&& graph, auto&& a2)
         {
             GILRelease gil_release;
             auto ret = get_global_clustering(std::forward<decltype(graph)>(graph),
                                              std::forward<decltype(a2)>(a2));
             gil_release.restore();
             oret = boost::python::make_tuple(get<0>(ret), get<1>(ret),
                                              get<2>(ret), get<3>(ret));
         },
         weight_props)(weight);
    return oret;
}

double global_clustering_sampled(GraphInterface& g, size_t m, rng_t& rng)
{
    double c = 0;
    run_action<decltype(graph_tool::never_directed)>()
        (g,
         [&](auto&& graph)
         {
             c = get_global_clustering_sampled(graph, m, rng);
         })();
    return c;
}

void local_clustering(GraphInterface& g, std::any prop, std::any weight)
{
    typedef UnityPropertyMap<size_t,GraphInterface::edge_t> weight_map_t;
    auto weight_props = hana::append(edge_scalar_properties,
                                     hana::type<weight_map_t>());

    if (weight.has_value() && !belongs(edge_scalar_properties, weight))
        throw ValueException("weight edge property must have a scalar value type");

    if(!weight.has_value())
        weight = weight_map_t();

    run_action<>()
        (g,
         [&](auto&& graph, auto&& a2, auto&& a3)
         {
             set_clustering_to_property(std::forward<decltype(graph)>(graph),
                                        std::forward<decltype(a2)>(a2),
                                        std::forward<decltype(a3)>(a3));
         },
         weight_props, writable_vertex_scalar_properties)(weight, prop);
}

using namespace boost::python;

BOOST_PYTHON_MODULE(libgraph_tool_clustering)
{
    docstring_options dopt(true, false);
    def("global_clustering", &global_clustering);
    def("global_clustering_sampled", &global_clustering_sampled);
    def("local_clustering", &local_clustering);
    __MOD__::EvokeRegistry();
}
