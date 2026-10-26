/// @file
/// @brief Planar Maximally Filtered Graph (structural/pmfg.py), on Boost's Boyer-Myrvold planarity
/// test.

#include <algorithm>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>

#include "../../include/cppbone/boost_graph.hpp"
#include "cppbone/cppbone.hpp"

namespace cppbone {

bool is_planar(const Graph &g) {
    return boost::boyer_myrvold_planarity_test(detail::to_boost(g, false));
}

Backbone pmfg(const Graph &data) {
    Graph g   = data;
    auto rows = g.edges();
    for (const auto &e : rows) {
        g.set(e.id, "in_backbone", false);
    }
    const auto w = g.edge_values("weight", NaN);
    std::stable_sort(rows.begin(), rows.end(),
                     [&](const EdgeRef &a, const EdgeRef &b) { return w[a.id] > w[b.id]; });
    // The growing backbone, on the input's node indices. Self-loops never affect planarity, so they
    // are only counted, not added to the graph that is tested.
    using Planar = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
    Planar backbone(g.number_of_nodes());
    int kept = 0;
    for (const auto &[u, v, id] : rows) {
        if (u != v) {
            auto [e, added] = boost::add_edge(u, v, backbone);
            if (!boost::boyer_myrvold_planarity_test(backbone)) {
                boost::remove_edge(e, backbone);
                continue;
            }
        }
        g.set(id, "in_backbone", true);
        if (++kept == 3 * (g.number_of_nodes() - 2)) {
            break;
        }
    }
    return {std::move(g),   "Planar Maximally Filtered Graph", "weight", false, {boolean_filter},
            FilterOn::Edges};
}

} // namespace cppbone
