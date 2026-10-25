#include "netbone/netbone.hpp"

namespace netbone {

Backbone plam(const Graph& data) {
    Graph g = data;
    for (const auto& e : g.edges()) g.set(e.id, "in_backbone", false);
    for (int node = 0; node < g.number_of_nodes(); ++node) {
        // get_max_weight_edge: first neighbour with the strictly largest weight
        double max_weight = -INF;
        int max_edge = -1;
        for (auto [v, id] : g.adj(node))
            if (g.get(id, "weight") > max_weight) max_weight = g.get(id, "weight"), max_edge = id;
        if (max_edge >= 0) g.set(max_edge, "in_backbone", true);
    }
    return {g, "Primary Linkage Analysis", "weight", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace netbone
