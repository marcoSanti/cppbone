#include "netbone/netbone.hpp"

namespace netbone {

Backbone degree(const Graph& data, bool weighted) {
    Graph g = data;
    std::string name = weighted ? "weighted-degree" : "degree";
    for (int u = 0; u < g.number_of_nodes(); ++u) g.node_set(u, name, weighted ? g.degree(u, "weight") : g.degree(u));
    return {g, weighted ? "Weighted Degree" : "Degree", name, false, {threshold_filter, fraction_filter},
            FilterOn::Nodes};
}

}  // namespace netbone
