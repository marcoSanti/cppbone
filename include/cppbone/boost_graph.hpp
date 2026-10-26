#pragma once

/// @file
/// @brief Conversion of a Graph to a Boost.Graph adjacency_list, and the Dijkstra built on it.

#include <algorithm>
#include <vector>

#include <boost/graph/adjacency_list.hpp>

#include "cppbone/cppbone.hpp"

namespace cppbone::detail {

/// @brief Undirected Boost graph whose vertices are node indices and whose edges carry the Graph
/// edge id.
using BoostGraph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property,
                          boost::property<boost::edge_index_t, int>>;

/// @brief Copies a graph into a BoostGraph.
///
/// Edges are added in id order, so every vertex lists its edges in the same order as Graph::adj().
/// @param g Graph.
/// @param self_loops Whether to keep self-loops.
/// @return The Boost graph; get(boost::edge_index, b, e) is the Graph edge id of e.
inline BoostGraph to_boost(const Graph &g, const bool self_loops = true) {
    auto edges = g.edges();
    std::sort(edges.begin(), edges.end(),
              [](const EdgeRef &a, const EdgeRef &b) { return a.id < b.id; });
    BoostGraph b(g.number_of_nodes());
    for (const auto &e : edges) {
        if (auto [u, v] = g.ends(e.id); self_loops || u != v) {
            boost::add_edge(u, v, e.id, b);
        }
    }
    return b;
}

/// @brief Result of one single-source search.
struct ShortestPaths {
    std::vector<double>
        dist; ///< Distance from the source; INF when unreached. Beyond a cutoff, upper bounds.
    std::vector<int> parent_edge; ///< Graph edge id to the shortest-path-tree parent; -1 for the
                                  ///< source and unreached.
    std::vector<int> settled;     ///< Nodes in the order they were settled (all below the cutoff).
};

/// @brief Shortest paths from one node with Boost's Dijkstra, distances combined by sum or max.
///
/// A node's tree edge is the last edge that strictly shortened its distance.
/// @param b Graph from to_boost().
/// @param source Start node.
/// @param weight Edge distances indexed by Graph edge id; must be non-negative.
/// @param disjunction Sum (metric) or Max (ultrametric) path lengths.
/// @param out Result (resized as needed, so one object can be reused across searches).
/// @param cutoff Stop once the next node to settle is at distance >= cutoff. Nodes reached but not
/// settled keep an
///        upper bound >= cutoff in out.dist.
/// @param removed If given, edges with a non-zero entry are ignored.
/// @throws boost::negative_edge (a std::invalid_argument) for a negative distance.
void dijkstra(const BoostGraph &b, int source, const std::vector<double> &weight,
              Disjunction disjunction, ShortestPaths &out, double cutoff = INF,
              const std::vector<char> *removed = nullptr);

} // namespace cppbone::detail
