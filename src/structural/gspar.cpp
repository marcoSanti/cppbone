#include <algorithm>
#include <iterator>

#include "detail.hpp"

namespace cppbone {

double jaccard(std::vector<int> a, std::vector<int> b) {
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    std::sort(b.begin(), b.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    std::vector<int> inter, uni;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(inter));
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(uni));
    return double(inter.size()) / uni.size();
}

// Closed neighbourhood (node and its neighbours), sorted and deduplicated.
static std::vector<int> get_neighbours(const Graph& g, int node) {
    std::vector<int> out{node};
    for (auto [v, id] : g.adj(node)) out.push_back(v);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

Backbone gspar(const Graph& data) {
    Graph g = data;
    int n = g.number_of_nodes();
    // jaccard() on precomputed neighbourhoods: the same counts without re-sorting per edge.
    std::vector<std::vector<int>> nb(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) nb[u] = get_neighbours(g, u);
    auto edges = g.edges();
    int m = (int)edges.size();
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < m; ++i) {
        const auto& a = nb[edges[i].u];
        const auto& b = nb[edges[i].v];
        size_t inter = 0, x = 0, y = 0;
        while (x < a.size() && y < b.size()) {
            if (a[x] < b[y]) ++x;
            else if (b[y] < a[x]) ++y;
            else ++inter, ++x, ++y;
        }
        g.set(edges[i].id, "jaccard-sim", double(inter) / (a.size() + b.size() - inter));
    }
    return {std::move(g), "Global Sparsification", "jaccard-sim", false, {threshold_filter, fraction_filter}, FilterOn::Edges};
}

}  // namespace cppbone
