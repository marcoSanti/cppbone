/// @file
/// @brief Maximum spanning tree backbone (structural/maximum_spanning_tree.py), on Boost's Kruskal.

#include <iterator>
#include <vector>

#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/property_map/function_property_map.hpp>

#include "boost_graph.hpp"
#include "cppbone/cppbone.hpp"

namespace cppbone {

Backbone maximum_spanning_tree(const Graph& data) {
    Graph g = data;
    auto w = g.edge_values("weight", NaN);
    auto b = detail::to_boost(g);
    auto edge_id = boost::get(boost::edge_index, b);
    using Edge = detail::BoostGraph::edge_descriptor;
    // The maximum spanning tree is the minimum spanning tree of the negated weights.
    auto negated = boost::make_function_property_map<Edge>([&](const Edge& e) { return -w[edge_id[e]]; });
    std::vector<Edge> tree;
    boost::kruskal_minimum_spanning_tree(b, std::back_inserter(tree), boost::weight_map(negated));
    for (const auto& e : g.edges()) g.set(e.id, "in_backbone", false);
    for (const auto& e : tree) g.set(edge_id[e], "in_backbone", true);
    return {std::move(g), "Maximum Spanning Tree", "weight", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace cppbone
