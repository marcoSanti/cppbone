/// @file
/// @brief Edge betweenness centrality and the betweenness backbone (structural/betweenness.py), on
/// Boost's Brandes.

#include <boost/graph/betweenness_centrality.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <boost/property_map/property_map.hpp>

#include "../../include/cppbone/boost_graph.hpp"
#include "../../include/cppbone/detail.hpp"

namespace cppbone {

/// Boost's Brandes algorithm (single-threaded). For undirected graphs it halves the sums over
/// ordered pairs, which is networkx's unnormalised value; normalisation then divides by n (n - 1)
/// / 2.
std::vector<double> edge_betweenness_centrality(const Graph &g, const bool normalized,
                                                const std::string &weight) {
    const int n = g.number_of_nodes();
    std::vector bc(g.edge_capacity(), 0.0);
    auto b             = detail::to_boost(g);
    const auto edge_id = boost::get(boost::edge_index, b);
    const auto edge_bc = boost::make_iterator_property_map(bc.begin(), edge_id);
    if (weight.empty()) {
        boost::brandes_betweenness_centrality(b, boost::edge_centrality_map(edge_bc));
    } else {
        const auto w = g.edge_values(weight);
        using Edge   = detail::BoostGraph::edge_descriptor;
        const auto length =
            boost::make_function_property_map<Edge>([&](const Edge &e) { return w[edge_id[e]]; });
        boost::brandes_betweenness_centrality(
            b, boost::edge_centrality_map(edge_bc).weight_map(length));
    }
    if (normalized && n >= 2) {
        for (double &x : bc) {
            x *= 2.0 / (static_cast<double>(n) * (n - 1));
        }
    }
    return bc;
}

Backbone betweenness(const Graph &data, const bool weighted, const bool normalized) {
    Graph g                = data;
    const std::string name = weighted ? "weighted-betweenness" : "betweenness";
    const auto bc          = edge_betweenness_centrality(g, normalized, weighted ? "weight" : "");
    const auto edges       = g.edges();
    const int m            = static_cast<int>(edges.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        g.set(edges[i].id, name, bc[edges[i].id]);
    }
    return {std::move(g),
            weighted ? "Weighted Betweenness" : "Betweenness",
            name,
            false,
            {threshold_filter, fraction_filter},
            FilterOn::Edges};
}

} // namespace cppbone
