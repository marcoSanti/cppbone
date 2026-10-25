// Globally and Locally Adaptive Backbone.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <queue>
#include <tuple>

#include "netbone/netbone.hpp"

namespace netbone {

// igraph_cmp_epsilon: igraph treats shortest-path lengths within a relative 1e-10 as equal.
static int cmp_epsilon(double a, double b) {
    const double eps = 1e-10;
    if (a == b) return 0;
    double diff = a - b, abs_diff = std::abs(diff), sum = std::abs(a) + std::abs(b);
    bool equal;
    if (a == 0 || b == 0 || sum < DBL_MIN)
        equal = abs_diff < eps * DBL_MIN;
    else if (!std::isfinite(sum))
        equal = abs_diff < eps * std::abs(a + b);
    else
        equal = abs_diff / sum < eps;
    return equal ? 0 : (diff < 0 ? -1 : 1);
}

// Number of shortest paths that igraph's get_all_shortest_paths(source) would list per first hop, without
// enumerating them: paths[x] = DAG paths starting at x, so paths through first hop v = paths[v].
static std::vector<double> paths_from(const Graph& g, int source, const std::vector<double>& distance,
                                      std::vector<std::vector<int>>& pred) {
    int n = g.number_of_nodes();
    std::vector<double> dist(n, INF);
    std::vector<bool> done(n, false);
    std::vector<int> order;
    for (auto& p : pred) p.clear();
    using Item = std::pair<double, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<>> q;
    dist[source] = 0;
    q.emplace(0.0, source);
    while (!q.empty()) {
        auto [d, v] = q.top();
        q.pop();
        if (done[v]) continue;
        done[v] = true;
        order.push_back(v);
        for (auto [u, id] : g.adj(v)) {
            if (done[u]) continue;
            double alt = d + distance[id];
            int c = std::isinf(dist[u]) ? -1 : cmp_epsilon(alt, dist[u]);
            if (c < 0) {
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
        for (int p : pred[*it]) paths[p] += paths[*it];
    }
    return paths;
}

Backbone glanb(const Graph& data, double c) {
    Graph g = data;
    for (const auto& e : g.edges()) g.set(e.id, "distance", 1 / g.get(e.id, "weight"));
    auto distance = g.edge_values("distance");
    std::vector<std::vector<int>> pred(g.number_of_nodes());
    for (int source = 0; source < g.number_of_nodes(); ++source) {
        int k_i = g.degree(source);
        if (k_i <= 1) continue;
        auto paths = paths_from(g, source, distance, pred);
        double g_is = paths[source] - 1;
        for (auto [v, id] : g.adj(source)) {
            bool first_hop = std::find(pred[v].begin(), pred[v].end(), source) != pred[v].end();
            double g_ij = first_hop ? paths[v] : 0;
            double I_ij = g_ij / g_is;
            double S_ij = std::pow(1 - I_ij, std::pow(k_i - 1, c));
            double cur = g.get(id, "SI");
            if (std::isnan(cur) || S_ij < cur) g.set(id, "SI", S_ij);
        }
    }
    return {g, "Globally and Locally Adaptive Backbone Filter", "SI", true, {threshold_filter, fraction_filter},
            FilterOn::Edges};
}

}  // namespace netbone
