#include "detail.hpp"

namespace cppbone {

Backbone degree(const Graph& data, bool weighted) {
    Graph g = data;
    std::string name = weighted ? "weighted-degree" : "degree";
    int n = g.number_of_nodes();
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) g.node_set(u, name, weighted ? g.degree(u, "weight") : g.degree(u));
    return {std::move(g), weighted ? "Weighted Degree" : "Degree", name, false, {threshold_filter, fraction_filter},
            FilterOn::Nodes};
}

}  // namespace cppbone
