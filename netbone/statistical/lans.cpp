#include <algorithm>

#include "netbone/netbone.hpp"

namespace netbone {

double compute_pvalue(const Graph& g, int node, double w) {
    double u_degree = g.degree(node, "weight");
    double puv = w / u_degree;
    int count = 0;
    for (auto [v, id] : g.adj(node))
        if (g.get(id, "weight") / u_degree <= puv) ++count;
    return 1 - double(count) / g.degree(node);
}

Backbone lans(const Graph& data) {
    Graph g = data;
    for (const auto& e : g.edges()) {
        double w = g.get(e.id, "weight");
        g.set(e.id, "p_value", std::min(compute_pvalue(g, e.v, w), compute_pvalue(g, e.u, w)));
    }
    return {g, "Locally Adaptive Network Sparsification Filter", "p_value", true, {threshold_filter, fraction_filter},
            FilterOn::Edges};
}

}  // namespace netbone
