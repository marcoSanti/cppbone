/// @file
/// @brief Globally and Locally Adaptive Backbone (hybrid/glanb.py).

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <queue>
#include <tuple>

#include "cppbone/cppbone.hpp"

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

/// @brief Counts the shortest paths from a source per first hop, without enumerating them.
///
/// Gives the numbers igraph's get_all_shortest_paths(source) would list: paths[x] counts the
/// shortest-path-DAG paths starting at x, so the paths through first hop v number paths[v].
/// @param g Graph.
/// @param source Start node.
/// @param distance Edge lengths indexed by edge id.
/// @param pred Per-node scratch (one vector per node); receives each node's predecessors in the
/// DAG.
/// @return paths[x] for every node; paths[source] - 1 is the number of shortest paths from source.
static std::vector<double> paths_from(const Graph &g, int source,
                                      const std::vector<double> &distance,
                                      std::vector<std::vector<int>> &pred) {
    const int n = g.number_of_nodes();
    std::vector dist(n, INF);
    std::vector done(n, false);
    std::vector<int> order;
    for (auto &p : pred) {
        p.clear();
    }
    using Item = std::pair<double, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<>> q;
    dist[source] = 0;
    q.emplace(0.0, source);
    while (!q.empty()) {
        auto [d, v] = q.top();
        q.pop();
        if (done[v]) {
            continue;
        }
        done[v] = true;
        order.push_back(v);
        for (auto [u, id] : g.adj(v)) {
            if (done[u]) {
                continue;
            }
            double alt = d + distance[id];
            if (const int c = std::isinf(dist[u]) ? -1 : cmp_epsilon(alt, dist[u]); c < 0) {
                dist[u] = alt;
                pred[u].assign(1, v);
                q.emplace(alt, u);
            } else if (c == 0) {
                pred[u].push_back(v);
            }
        }
    }
    std::vector<double> paths(n, 0);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        paths[*it] += 1;
        for (const int p : pred[*it]) {
            paths[p] += paths[*it];
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
            auto paths        = paths_from(g, source, distance, pred);
            const double g_is = paths[source] - 1;
            for (auto [v, id] : g.adj(source)) {
                const bool first_hop =
                    std::find(pred[v].begin(), pred[v].end(), source) != pred[v].end();
                const double g_ij = first_hop ? paths[v] : 0;
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
        int a = std::min(u, v), b = std::max(u, v);
        bool has   = false;
        double cur = NaN;
        for (const int node : {a, b}) {
            if (node == b && a == b) {
                break; // a self-loop is visited once
            }
            if (g.degree(node) <= 1) {
                continue;
            }
            if (const double s_ij =
                    si[2 * static_cast<size_t>(id) + (g.ends(id).first == node ? 0 : 1)];
                !has || std::isnan(cur) || s_ij < cur) {
                cur = s_ij, has = true;
            }
        }
        if (has) {
            g.set(id, "SI", cur);
        }
    }
    return {std::move(g),
            "Globally and Locally Adaptive Backbone Filter",
            "SI",
            true,
            {threshold_filter, fraction_filter},
            FilterOn::Edges};
}
} // namespace cppbone
