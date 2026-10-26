/// @file
/// @brief Globally and Locally Adaptive Backbone (hybrid/glanb.py), on Boost's Dijkstra.

#include <algorithm>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <cfloat>
#include <cmath>
#include <ranges>
#include <span>

#include "../../include/cppbone/boost_graph.hpp"
#include "../../include/cppbone/detail.hpp"

namespace cppbone {
/// @brief Three-way comparison of path lengths with igraph's tolerance (igraph_cmp_epsilon).
///
/// igraph treats shortest-path lengths within a relative 1e-10 as equal.
/// @param a First length.
/// @param b Second length.
/// @return 0 if a and b are equal within the tolerance, otherwise -1 if a < b and 1 if a > b.
static int cmp_epsilon(const double a, const double b) {
    constexpr double eps = 1e-10;
    if (a == b) {
        return 0;
    }
    const double diff     = a - b;
    const double abs_diff = std::abs(diff);
    const double sum      = std::abs(a) + std::abs(b);
    bool equal;
    if (a == 0 || b == 0 || sum < DBL_MIN) {
        equal = abs_diff < eps * DBL_MIN;
    } else if (!std::isfinite(sum)) {
        equal = abs_diff < eps * std::abs(a + b);
    } else {
        equal = abs_diff / sum < eps;
    }
    return equal ? 0 : (diff < 0 ? -1 : 1);
}

namespace {
/// @brief Records, during Boost's Dijkstra, the settle order and every predecessor of each node in
/// the shortest-path DAG (ties within igraph's tolerance included).
struct DagRecorder : boost::default_dijkstra_visitor {
    std::span<const double> length;      ///< Edge lengths by edge id.
    const double *dist;                  ///< The search's distance map.
    std::vector<std::vector<int>> *pred; ///< Predecessors of every node.
    std::vector<int> *order;             ///< Nodes in settle order.

    /// @brief Records a node as settled.
    /// @param u Node.
    [[maybe_unused]] void examine_vertex(const detail::BoostGraph::vertex_descriptor u,
                                         const detail::BoostGraph &) const {
        order->push_back(static_cast<int>(u));
    }

    /// @brief A strictly shorter path to the target: its source becomes the only predecessor.
    /// @param e Edge.
    /// @param b Graph.
    [[maybe_unused]] void edge_relaxed(const detail::BoostGraph::edge_descriptor &e,
                                       const detail::BoostGraph &b) const {
        (*pred)[boost::target(e, b)].assign(1, static_cast<int>(boost::source(e, b)));
    }

    /// @brief A path to an unsettled target that is not shorter: an equally short one adds a
    /// predecessor.
    /// @param e Edge.
    /// @param b Graph.
    [[maybe_unused]] void edge_not_relaxed(detail::BoostGraph::edge_descriptor e,
                                           const detail::BoostGraph &b) const {
        const auto u = boost::source(e, b), v = boost::target(e, b);
        if (const double alt = dist[u] + length[boost::get(boost::edge_index, b, e)];
            std::isfinite(alt) && cmp_epsilon(alt, dist[v]) == 0) {
            (*pred)[v].push_back(static_cast<int>(u));
        }
    }
};
} // namespace

/// @brief Counts the shortest paths from a source per first hop, without enumerating them.
///
/// Gives the numbers igraph's get_all_shortest_paths(source) would list: paths[x] counts the
/// shortest-path-DAG paths starting at x, so the paths through first hop v number paths[v]. The DAG
/// comes from Boost's Dijkstra with igraph's tolerant comparison of lengths.
/// @param bg Graph from detail::to_boost().
/// @param source Start node.
/// @param distance Edge lengths indexed by edge id.
/// @param pred Per-node scratch (one vector per node); receives each node's predecessors in the
/// DAG.
/// @return paths[x] for every node; paths[source] - 1 is the number of shortest paths from source.
static std::vector<double> paths_from(const detail::BoostGraph &bg, int source,
                                      const std::span<const double> distance,
                                      std::vector<std::vector<int>> &pred) {
    const int n = static_cast<int>(boost::num_vertices(bg));
    std::vector dist(n, INF);
    std::vector<int> order;
    for (auto &p : pred) {
        p.clear();
    }
    using Edge         = detail::BoostGraph::edge_descriptor;
    const auto edge_id = boost::get(boost::edge_index, bg);
    const auto length  = boost::make_function_property_map<Edge>(
        [&](const Edge &e) { return distance[edge_id[e]]; });
    auto less = [](const double a, const double b) { return cmp_epsilon(a, b) < 0; };
    boost::dijkstra_shortest_paths(
        bg, source,
        boost::weight_map(length)
            .distance_map(dist.data())
            .distance_compare(less)
            .distance_inf(INF)
            .distance_zero(0.0)
            .visitor(DagRecorder{{}, distance, dist.data(), &pred, &order}));
    std::vector<double> paths(n, 0);
    for (const int x : order | std::views::reverse) {
        paths[x] += 1;
        for (const int p : pred[x]) {
            paths[p] += paths[x];
        }
    }
    return paths;
}

Backbone glanb(const Graph &data, double c) {
    Graph g          = data;
    const auto edges = g.edges();
    const int m      = static_cast<int>(edges.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        g.set(edges[i].id, "distance", 1 / g.get(edges[i].id, "weight"));
    }
    const auto distance = g.edge_values("distance");
    const auto bg       = detail::to_boost(g);
    const int n         = g.number_of_nodes();
    // si[2 id + side]: S_ij computed from endpoint `side` (0 = ends().first), NaN when that
    // endpoint is skipped.
    std::vector si(2 * static_cast<size_t>(g.edge_capacity()), NaN);
#pragma omp parallel
    {
        std::vector<std::vector<int>> pred(n);
#pragma omp for schedule(dynamic, 4)
        for (int source = 0; source < n; ++source) {
            const int k_i = g.degree(source);
            if (k_i <= 1) {
                continue;
            }
            auto paths        = paths_from(bg, source, distance, pred);
            const double g_is = paths[source] - 1;
            for (const auto [v, id] : g.adj(source)) {
                const double g_ij = std::ranges::contains(pred[v], source) ? paths[v] : 0;
                const double I_ij = g_ij / g_is;
                si[2 * static_cast<size_t>(id) + (g.ends(id).first == source ? 0 : 1)] =
                    std::pow(1 - I_ij, std::pow(k_i - 1, c));
            }
        }
    }
    // Keep the smaller value, visiting the endpoints in node order like the sequential loop.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const auto &[u, v, id] = edges[i];
        const auto s           = detail::endpoint_min(
            u, v, [&](const int node) { return g.degree(node) <= 1; },
            [&](const int node) {
                return si[2 * static_cast<size_t>(id) + (g.ends(id).first == node ? 0 : 1)];
            });
        if (s) {
            g.set(id, "SI", *s);
        }
    }
    return {.graph         = std::move(g),
            .method_name   = "Globally and Locally Adaptive Backbone Filter",
            .property_name = "SI",
            .ascending     = true,
            .filters       = {threshold_filter, fraction_filter}};
}
} // namespace cppbone
