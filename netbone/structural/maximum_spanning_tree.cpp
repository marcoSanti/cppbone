#include <algorithm>
#include <numeric>

#include "netbone/netbone.hpp"

namespace netbone {

static int find(std::vector<int>& p, int x) {
    while (p[x] != x) x = p[x] = p[p[x]];
    return x;
}

Backbone maximum_spanning_tree(const Graph& data) {
    Graph g = data;
    auto edges = g.edges();
    // Kruskal on decreasing weight; stable like Python's sorted(reverse=True).
    std::stable_sort(edges.begin(), edges.end(),
                     [&](const EdgeRef& a, const EdgeRef& b) { return g.get(a.id, "weight") > g.get(b.id, "weight"); });
    std::vector<int> parent(g.number_of_nodes());
    std::iota(parent.begin(), parent.end(), 0);
    for (const auto& e : edges) {
        int a = find(parent, e.u), b = find(parent, e.v);
        g.set(e.id, "in_backbone", a != b);
        if (a != b) parent[a] = b;
    }
    return {g, "Maximum Spanning Tree", "weight", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace netbone
