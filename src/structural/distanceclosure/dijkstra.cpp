/// @file
/// @brief Shortest paths with a pluggable path disjunction (distanceclosure/dijkstra.py), on Boost's Dijkstra.

#include <algorithm>
#include <functional>

#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/function_property_map.hpp>

#include "boost_graph.hpp"
#include "detail.hpp"

namespace cppbone {

namespace {

/// @brief Thrown by the visitor to end a search at the cutoff.
struct Cutoff {};

/// @brief Records the settled nodes and the tree edges, and ends the search at the cutoff.
struct Recorder : boost::default_dijkstra_visitor {
    detail::ShortestPaths* out;  ///< Where to record.
    double cutoff;               ///< Distance at which the search stops.

    /// @brief Called when a node is taken from the queue: stops at the cutoff, otherwise records it as settled.
    /// @param u Node.
    void examine_vertex(detail::BoostGraph::vertex_descriptor u, const detail::BoostGraph&) const {
        if (out->dist[u] >= cutoff) throw Cutoff{};
        out->settled.push_back((int)u);
    }

    /// @brief Called when an edge shortens the path to its target: the edge becomes the target's tree edge.
    /// @param e Edge.
    /// @param b Graph.
    void edge_relaxed(detail::BoostGraph::edge_descriptor e, const detail::BoostGraph& b) const {
        out->parent_edge[boost::target(e, b)] = boost::get(boost::edge_index, b, e);
    }
};

/// @brief Runs Boost's Dijkstra with a given way of combining distances along a path.
/// @tparam Combine Callable (path length so far, edge distance) -> new path length.
/// @param b Graph.
/// @param source Start node.
/// @param weight Edge distances by edge id.
/// @param combine How distances combine.
/// @param out Result.
/// @param cutoff Stop distance.
/// @param removed Edges to ignore, or null.
template <typename Combine>
void run(const detail::BoostGraph& b, int source, const std::vector<double>& weight, Combine combine,
         detail::ShortestPaths& out, double cutoff, const std::vector<char>* removed) {
    using Edge = detail::BoostGraph::edge_descriptor;
    auto edge_id = boost::get(boost::edge_index, b);
    // A removed edge has infinite length, so it never shortens a path.
    auto length = boost::make_function_property_map<Edge>([&](const Edge& e) {
        int id = edge_id[e];
        return removed && (*removed)[id] ? INF : weight[id];
    });
    try {
        boost::dijkstra_shortest_paths(b, source,
                                       boost::weight_map(length)
                                           .distance_map(out.dist.data())
                                           .distance_combine(combine)
                                           .distance_inf(INF)
                                           .distance_zero(0.0)
                                           .visitor(Recorder{{}, &out, cutoff}));
    } catch (const Cutoff&) {
    }
}

}  // namespace

namespace detail {

void dijkstra(const BoostGraph& b, int source, const std::vector<double>& weight, Disjunction disjunction,
              ShortestPaths& out, double cutoff, const std::vector<char>* removed) {
    int n = (int)boost::num_vertices(b);
    out.dist.assign(n, INF);
    out.parent_edge.assign(n, -1);
    out.settled.clear();
    if (disjunction == Disjunction::Sum)
        run(b, source, weight, std::plus<double>(), out, cutoff, removed);
    else
        run(b, source, weight, [](double d, double w) { return std::max(d, w); }, out, cutoff, removed);
}

}  // namespace detail

std::vector<double> single_source_dijkstra_path_length(const Graph& g, int source, const std::vector<double>& weight,
                                                       Disjunction disjunction, std::vector<int>* parent) {
    detail::ShortestPaths sp;
    detail::dijkstra(detail::to_boost(g), source, weight, disjunction, sp);
    if (parent) {
        parent->assign(g.number_of_nodes(), -1);
        for (int u = 0; u < g.number_of_nodes(); ++u)
            if (sp.parent_edge[u] >= 0) {
                auto [a, c] = g.ends(sp.parent_edge[u]);
                (*parent)[u] = a == u ? c : a;
            }
    }
    return sp.dist;
}

std::vector<std::vector<double>> all_pairs_dijkstra_path_length(const Graph& g, const std::string& weight,
                                                                Disjunction disjunction) {
    auto w = g.edge_values(weight);
    auto b = detail::to_boost(g);
    int n = g.number_of_nodes();
    std::vector<std::vector<double>> out(n);
    detail::Errors errors;
#pragma omp parallel
    {
        detail::ShortestPaths sp;
#pragma omp for schedule(dynamic, 16)
        for (int s = 0; s < n; ++s)
            errors.run([&] {
                detail::dijkstra(b, s, w, disjunction, sp);
                out[s] = sp.dist;
            });
    }
    errors.rethrow();
    return out;
}

}  // namespace cppbone
