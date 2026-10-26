#include <algorithm>
#include <cmath>

#include "detail.hpp"

namespace cppbone {

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
    int n = g.number_of_nodes();
    auto w = g.edge_values("weight", NaN);
    // compute_pvalue counts the neighbours with weight / strength <= puv; with each node's ratios sorted that is
    // a binary search, on the same doubles, instead of a scan per edge.
    std::vector<double> strength(n);
    std::vector<std::vector<double>> ratios(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        strength[u] = g.degree(u, "weight");
        for (auto [v, id] : g.adj(u)) {
            double r = w[id] / strength[u];
            if (!std::isnan(r)) ratios[u].push_back(r);  // a NaN ratio never compares <=, so it is never counted
        }
        std::sort(ratios[u].begin(), ratios[u].end());
    }
    auto pvalue = [&](int u, double weight) {
        double puv = weight / strength[u];
        auto count = std::upper_bound(ratios[u].begin(), ratios[u].end(), puv,
                                      [](double x, double r) { return !(r <= x); }) - ratios[u].begin();
        return 1 - double(count) / g.degree(u);
    };
    auto edges = g.edges();
    int m = (int)edges.size();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const auto& e = edges[i];
        g.set(e.id, "p_value", std::min(pvalue(e.v, w[e.id]), pvalue(e.u, w[e.id])));
    }
    return {std::move(g), "Locally Adaptive Network Sparsification Filter", "p_value", true, {threshold_filter, fraction_filter},
            FilterOn::Edges};
}

}  // namespace cppbone
