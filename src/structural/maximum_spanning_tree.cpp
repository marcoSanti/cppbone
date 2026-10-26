/// @file
/// @brief Maximum spanning tree backbone (structural/maximum_spanning_tree.py).

#include <algorithm>
#include <numeric>

#include "cppbone/cppbone.hpp"

namespace cppbone {

/// @brief Union-find root lookup with path halving.
/// @param p Parent array; updated to shorten the paths.
/// @param x Element.
/// @return The root of x's set.
static int find(std::vector<int>& p, int x) {
    while (p[x] != x) x = p[x] = p[p[x]];
    return x;
}

Backbone maximum_spanning_tree(const Graph& data) {
    Graph g = data;
    auto edges = g.edges();
    // Kruskal on decreasing weight; stable like Python's sorted(reverse=True).
    auto w = g.edge_values("weight", NaN);
    std::stable_sort(edges.begin(), edges.end(), [&](const EdgeRef& a, const EdgeRef& b) { return w[a.id] > w[b.id]; });
    std::vector<int> parent(g.number_of_nodes());
    std::iota(parent.begin(), parent.end(), 0);
    for (const auto& e : edges) {
        int a = find(parent, e.u), b = find(parent, e.v);
        g.set(e.id, "in_backbone", a != b);
        if (a != b) parent[a] = b;
    }
    return {std::move(g), "Maximum Spanning Tree", "weight", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace cppbone
