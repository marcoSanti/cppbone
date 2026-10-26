/// @file
/// @brief Dijkstra with a pluggable path disjunction (distanceclosure/dijkstra.py, adapted from networkx), plus the
///        reusable-workspace variant used by the parallel methods.

#include <algorithm>
#include <functional>
#include <queue>
#include <stdexcept>
#include <tuple>

#include "detail.hpp"

namespace cppbone {

std::vector<double> single_source_dijkstra_path_length(const Graph& g, int source, const std::vector<double>& weight,
                                                       Disjunction disjunction, std::vector<int>* parent) {
    int n = g.number_of_nodes();
    std::vector<double> dist(n, INF), seen(n, INF);
    std::vector<bool> done(n, false);
    if (parent) parent->assign(n, -1);
    // (distance, insertion count, node); the count breaks ties in insertion order like heapq + itertools.count.
    using Item = std::tuple<double, long, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<>> fringe;
    long c = 0;
    seen[source] = 0;
    fringe.emplace(0.0, c++, source);
    while (!fringe.empty()) {
        auto [d, _, v] = fringe.top();
        fringe.pop();
        if (done[v]) continue;  // already searched this node.
        done[v] = true;
        dist[v] = d;
        for (auto [u, id] : g.adj(v)) {
            double cost = weight[id];
            double vu_dist = disjunction == Disjunction::Sum ? d + cost : std::max(d, cost);
            if (done[u]) {
                if (vu_dist < dist[u]) throw std::invalid_argument("Contradictory paths found: negative weights?");
            } else if (vu_dist < seen[u]) {
                seen[u] = vu_dist;
                fringe.emplace(vu_dist, c++, u);
                if (parent) (*parent)[u] = v;
            }
        }
    }
    return dist;
}

namespace detail {

void dijkstra(const Graph& g, int source, const std::vector<double>& weight, Disjunction disjunction,
              DijkstraWorkspace& ws, double cutoff, const std::vector<char>* removed) {
    for (int v : ws.touched) ws.dist[v] = ws.seen[v] = INF, ws.done[v] = 0, ws.parent_edge[v] = -1;
    ws.touched.clear();
    ws.settled.clear();
    ws.heap.clear();
    // Same ordering as the priority_queue above: a min-heap on (distance, insertion count, node).
    auto push = [&](double d, long c, int v) {
        ws.heap.emplace_back(d, c, v);
        std::push_heap(ws.heap.begin(), ws.heap.end(), std::greater<>());
    };
    long c = 0;
    ws.seen[source] = 0;
    ws.touched.push_back(source);
    push(0.0, c++, source);
    while (!ws.heap.empty()) {
        std::pop_heap(ws.heap.begin(), ws.heap.end(), std::greater<>());
        auto [d, _, v] = ws.heap.back();
        ws.heap.pop_back();
        if (ws.done[v]) continue;
        if (d >= cutoff) break;
        ws.done[v] = 1;
        ws.dist[v] = d;
        ws.settled.push_back(v);
        for (auto [u, id] : g.adj(v)) {
            if (removed && (*removed)[id]) continue;
            double cost = weight[id];
            double vu_dist = disjunction == Disjunction::Sum ? d + cost : std::max(d, cost);
            if (ws.done[u]) {
                if (vu_dist < ws.dist[u]) throw std::invalid_argument("Contradictory paths found: negative weights?");
            } else if (vu_dist < ws.seen[u]) {
                if (ws.seen[u] == INF) ws.touched.push_back(u);
                ws.seen[u] = vu_dist;
                push(vu_dist, c++, u);
                ws.parent_edge[u] = id;
            }
        }
    }
}

}  // namespace detail

std::vector<std::vector<double>> all_pairs_dijkstra_path_length(const Graph& g, const std::string& weight,
                                                                Disjunction disjunction) {
    auto w = g.edge_values(weight);
    int n = g.number_of_nodes();
    std::vector<std::vector<double>> out(n);
    detail::Errors errors;
#pragma omp parallel for schedule(dynamic, 16)
    for (int s = 0; s < n; ++s)
        errors.run([&] { out[s] = single_source_dijkstra_path_length(g, s, w, disjunction); });
    errors.rethrow();
    return out;
}

}  // namespace cppbone
