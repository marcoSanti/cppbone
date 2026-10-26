/// @file
/// @brief Global Sparsification (structural/gspar.py).

#include <algorithm>
#include <iterator>
#include <ranges>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

double jaccard(std::vector<int> a, std::vector<int> b) {
    for (auto *set : {&a, &b}) {
        std::ranges::sort(*set);
        set->erase(std::ranges::unique(*set).begin(), set->end());
    }
    std::vector<int> inter, uni;
    std::ranges::set_intersection(a, b, std::back_inserter(inter));
    std::ranges::set_union(a, b, std::back_inserter(uni));
    return static_cast<double>(inter.size()) / uni.size();
}

/// @brief Closed neighbourhood of a node: the node and its neighbours, sorted and deduplicated.
/// @param g Graph.
/// @param node Node index.
/// @return The sorted node indices.
static std::vector<int> get_neighbours(const Graph &g, const int node) {
    std::vector<int> out{node};
    out.append_range(g.adj(node) | std::views::transform(&Adjacent::node));
    std::ranges::sort(out);
    out.erase(std::ranges::unique(out).begin(), out.end());
    return out;
}

Backbone gspar(const Graph &data) {
    Graph g     = data;
    const int n = g.number_of_nodes();
    // jaccard() on precomputed neighbourhoods: the same counts without re-sorting per edge.
    std::vector<std::vector<int>> nb(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        nb[u] = get_neighbours(g, u);
    }
    const auto edges = g.edges();
    const int m      = static_cast<int>(edges.size());
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < m; ++i) {
        const auto &a = nb[edges[i].u];
        const auto &b = nb[edges[i].v];
        size_t inter = 0, x = 0, y = 0;
        while (x < a.size() && y < b.size()) {
            if (a[x] < b[y]) {
                ++x;
            } else if (b[y] < a[x]) {
                ++y;
            } else {
                ++inter, ++x, ++y;
            }
        }
        g.set(edges[i].id, "jaccard-sim", double(inter) / (a.size() + b.size() - inter));
    }
    return {.graph         = std::move(g),
            .method_name   = "Global Sparsification",
            .property_name = "jaccard-sim",
            .filters       = {threshold_filter, fraction_filter}};
}

} // namespace cppbone
