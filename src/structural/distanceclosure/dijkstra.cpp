/// @file
/// @brief Shortest paths with a pluggable path disjunction (distanceclosure/dijkstra.py), on
/// Boost's Dijkstra.

#include <algorithm>
#include <functional>
#include <span>

#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/function_property_map.hpp>

#include "../../../include/cppbone/boost_graph.hpp"
#include "../../../include/cppbone/detail.hpp"

namespace cppbone {

namespace {

/// @brief Thrown by the visitor to end a search at the cutoff.
struct Cutoff {};

/// @brief Records the settled nodes and the tree edges, and ends the search at the cutoff.
struct Recorder : boost::default_dijkstra_visitor {
    detail::ShortestPaths *out; ///< Where to record.
    double cutoff;              ///< Distance at which the search stops.

    /// @brief Called when a node is taken from the queue: stops at the cutoff, otherwise records it
    /// as settled.
    /// @param u Node.
    [[maybe_unused]] void examine_vertex(const detail::BoostGraph::vertex_descriptor u,
                                         const detail::BoostGraph &) const {
        if (out->dist[u] >= cutoff) {
            throw Cutoff{};
        }
        out->settled.push_back(static_cast<int>(u));
    }

    /// @brief Called when an edge shortens the path to its target: the edge becomes the target's
    /// tree edge.
    /// @param e Edge.
    /// @param b Graph.
    [[maybe_unused]] void edge_relaxed(detail::BoostGraph::edge_descriptor e,
                                       const detail::BoostGraph &b) const {
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
/// @param removed Edges to ignore (non-zero entries); empty ignores none.
template <typename Combine>
void run(const detail::BoostGraph &b, int source, const std::span<const double> weight,
         Combine combine, detail::ShortestPaths &out, double cutoff,
         const std::span<const char> removed) {
    using Edge         = detail::BoostGraph::edge_descriptor;
    const auto edge_id = boost::get(boost::edge_index, b);
    // A removed edge has infinite length, so it never shortens a path.
    auto length        = boost::make_function_property_map<Edge>([&](const Edge &e) {
        const int id = edge_id[e];
        return !removed.empty() && removed[id] ? INF : weight[id];
    });
    try {
        boost::dijkstra_shortest_paths(b, source,
                                       boost::weight_map(length)
                                           .distance_map(out.dist.data())
                                           .distance_combine(combine)
                                           .distance_inf(INF)
                                           .distance_zero(0.0)
                                           .visitor(Recorder{{}, &out, cutoff}));
    } catch (const Cutoff &) {
    }
}

} // namespace

namespace detail {

void dijkstra(const BoostGraph &b, int source, const std::span<const double> weight,
              Disjunction disjunction, ShortestPaths &out, const double cutoff,
              const std::span<const char> removed) {
    const int n = static_cast<int>(boost::num_vertices(b));
    out.dist.assign(n, INF);
    out.parent_edge.assign(n, -1);
    out.settled.clear();
    if (disjunction == Disjunction::Sum) {
        run(b, source, weight, std::plus<>(), out, cutoff, removed);
    } else {
        run(
            b, source, weight, [](const double d, const double w) { return std::max(d, w); }, out,
            cutoff, removed);
    }
}

} // namespace detail

std::vector<double> single_source_dijkstra_path_length(const Graph &g, const int source,
                                                       const std::span<const double> weight,
                                                       const Disjunction disjunction,
                                                       std::vector<int> *parent) {
    detail::ShortestPaths sp;
    detail::dijkstra(detail::to_boost(g), source, weight, disjunction, sp);
    if (parent) {
        parent->assign(g.number_of_nodes(), -1);
        for (const int u : g.nodes()) {
            if (sp.parent_edge[u] >= 0) {
                const auto [a, c] = g.ends(sp.parent_edge[u]);
                (*parent)[u]      = a == u ? c : a;
            }
        }
    }
    return sp.dist;
}

std::vector<std::vector<double>> all_pairs_dijkstra_path_length(const Graph &g,
                                                                const std::string_view weight,
                                                                const Disjunction disjunction) {
    const auto w = g.edge_values(weight);
    const auto b = detail::to_boost(g);
    const int n  = g.number_of_nodes();
    std::vector<std::vector<double>> out(n);
    detail::Errors errors;
#pragma omp parallel
    {
        detail::ShortestPaths sp;
#pragma omp for schedule(dynamic, 16)
        for (int s = 0; s < n; ++s) {
            errors.run([&] {
                detail::dijkstra(b, s, w, disjunction, sp);
                out[s] = sp.dist;
            });
        }
    }
    errors.rethrow();
    return out;
}

} // namespace cppbone
