// Dijkstra with a pluggable path disjunction, adapted (like the Python module) from networkx.
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <tuple>

#include "netbone/netbone.hpp"

namespace netbone {

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

std::vector<std::vector<double>> all_pairs_dijkstra_path_length(const Graph& g, const std::string& weight,
                                                                Disjunction disjunction) {
    auto w = g.edge_values(weight);
    std::vector<std::vector<double>> out;
    for (int n = 0; n < g.number_of_nodes(); ++n) out.push_back(single_source_dijkstra_path_length(g, n, w, disjunction));
    return out;
}

}  // namespace netbone
