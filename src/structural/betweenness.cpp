/// @file
/// @brief Edge betweenness centrality and the betweenness backbone (structural/betweenness.py).

#include <deque>
#include <queue>
#include <tuple>

#include "cppbone/cppbone.hpp"

namespace cppbone {

/// Brandes' algorithm, a line-by-line port of networkx (including its tie handling) so values match. Sources run
/// in parallel; their contributions are summed atomically, so the last bits depend on the schedule.
std::vector<double> edge_betweenness_centrality(const Graph& g, bool normalized, const std::string& weight) {
    int n = g.number_of_nodes();
    std::vector<double> bc(g.edge_capacity(), 0.0);
    std::vector<double> w = weight.empty() ? std::vector<double>() : g.edge_values(weight);
#pragma omp parallel
    {
        std::vector<double> sigma(n, 0), delta(n), dist(n, -1), seen(n, INF);
        std::vector<std::vector<std::pair<int, int>>> pred(n);  // (predecessor, edge id)
        std::vector<int> order;
        std::vector<bool> done(n, false);
#pragma omp for schedule(dynamic, 4)
        for (int s = 0; s < n; ++s) {
            // only the nodes the previous source reached were touched
            for (int v : order) pred[v].clear(), sigma[v] = 0, done[v] = false, dist[v] = -1, seen[v] = INF;
            order.clear();
            if (weight.empty()) {  // _single_source_shortest_path_basic
                std::deque<int> q{s};
                sigma[s] = 1;
                dist[s] = 0;
                while (!q.empty()) {
                    int v = q.front();
                    q.pop_front();
                    order.push_back(v);
                    for (auto [u, id] : g.adj(v)) {
                        if (dist[u] < 0) q.push_back(u), dist[u] = dist[v] + 1;
                        if (dist[u] == dist[v] + 1) sigma[u] += sigma[v], pred[u].emplace_back(v, id);
                    }
                }
            } else {  // _single_source_dijkstra_path_basic
                using Item = std::tuple<double, long, int, int, int>;  // dist, count, pred, node, edge
                std::priority_queue<Item, std::vector<Item>, std::greater<>> q;
                long c = 0;
                sigma[s] = 1;
                seen[s] = 0;
                q.emplace(0.0, c++, s, s, -1);
                while (!q.empty()) {
                    auto [d, _, p, v, eid] = q.top();
                    q.pop();
                    if (done[v]) continue;
                    sigma[v] += sigma[p];  // count paths
                    order.push_back(v);
                    done[v] = true;
                    for (auto [u, id] : g.adj(v)) {
                        double vu = d + w[id];
                        if (!done[u] && vu < seen[u]) {
                            seen[u] = vu;
                            q.emplace(vu, c++, v, u, id);
                            sigma[u] = 0;
                            pred[u].assign(1, {v, id});
                        } else if (vu == seen[u]) {  // handle equal paths
                            sigma[u] += sigma[v];
                            pred[u].emplace_back(v, id);
                        }
                    }
                }
            }
            for (int v : order) delta[v] = 0;
            for (auto it = order.rbegin(); it != order.rend(); ++it) {  // _accumulate_edges
                int x = *it;
                double coeff = (1 + delta[x]) / sigma[x];
                for (auto [v, id] : pred[x]) {
                    double cval = sigma[v] * coeff;
                    double& b = bc[id];
#pragma omp atomic
                    b += cval;
                    delta[v] += cval;
                }
            }
        }
    }
    if (n >= 2) {
        double scale = normalized ? 1.0 / (double(n) * (n - 1)) : 0.5;
        int m = (int)bc.size();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < m; ++i) bc[i] *= scale;
    }
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
