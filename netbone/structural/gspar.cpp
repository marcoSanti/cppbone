#include <algorithm>
#include <iterator>

#include "netbone/netbone.hpp"

namespace netbone {

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

static std::vector<int> get_neighbours(const Graph& g, int node) {
    std::vector<int> out{node};
    for (auto [v, id] : g.adj(node)) out.push_back(v);
    return out;
}

Backbone gspar(const Graph& data) {
    Graph g = data;
    for (const auto& e : g.edges()) g.set(e.id, "jaccard-sim", jaccard(get_neighbours(g, e.u), get_neighbours(g, e.v)));
    return {g, "Global Sparsification", "jaccard-sim", false, {threshold_filter, fraction_filter}, FilterOn::Edges};
}

}  // namespace netbone
