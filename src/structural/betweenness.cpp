/// @file
/// @brief Edge betweenness centrality and the betweenness backbone (structural/betweenness.py), on Boost's Brandes.

#include <boost/graph/betweenness_centrality.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <boost/property_map/property_map.hpp>

#include "boost_graph.hpp"
#include "detail.hpp"

namespace cppbone {

/// Boost's Brandes algorithm (single-threaded). For undirected graphs it halves the sums over ordered pairs, which
/// is networkx's unnormalised value; normalisation then divides by n (n - 1) / 2.
std::vector<double> edge_betweenness_centrality(const Graph& g, bool normalized, const std::string& weight) {
    int n = g.number_of_nodes();
    std::vector<double> bc(g.edge_capacity(), 0.0);
    auto b = detail::to_boost(g);
    auto edge_id = boost::get(boost::edge_index, b);
    auto edge_bc = boost::make_iterator_property_map(bc.begin(), edge_id);
    if (weight.empty()) {
        boost::brandes_betweenness_centrality(b, boost::edge_centrality_map(edge_bc));
    } else {
        auto w = g.edge_values(weight);
        using Edge = detail::BoostGraph::edge_descriptor;
        auto length = boost::make_function_property_map<Edge>([&](const Edge& e) { return w[edge_id[e]]; });
        boost::brandes_betweenness_centrality(b, boost::edge_centrality_map(edge_bc).weight_map(length));
    }
    if (normalized && n >= 2)
        for (double& x : bc) x *= 2.0 / (double(n) * (n - 1));
    return bc;
}

Backbone betweenness(const Graph& data, bool weighted, bool normalized) {
    Graph g = data;
    std::string name = weighted ? "weighted-betweenness" : "betweenness";
    auto bc = edge_betweenness_centrality(g, normalized, weighted ? "weight" : "");
    auto edges = g.edges();
    int m = (int)edges.size();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) g.set(edges[i].id, name, bc[edges[i].id]);
    return {std::move(g), weighted ? "Weighted Betweenness" : "Betweenness", name, false, {threshold_filter, fraction_filter},
            FilterOn::Edges};
}

}  // namespace cppbone
