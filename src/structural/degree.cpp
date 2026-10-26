/// @file
/// @brief Degree backbone (structural/degree.py).

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

Backbone degree(const Graph &data, bool weighted) {
    Graph g                     = data;
    const std::string_view name = weighted ? "weighted-degree" : "degree";
    const int n                 = g.number_of_nodes();
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        g.node_set(u, name, weighted ? g.degree(u, "weight") : g.degree(u));
    }
    return {.graph         = std::move(g),
            .method_name   = weighted ? "Weighted Degree" : "Degree",
            .property_name = std::string(name),
            .filters       = {threshold_filter, fraction_filter},
            .filter_on     = FilterOn::Nodes};
}

} // namespace cppbone
