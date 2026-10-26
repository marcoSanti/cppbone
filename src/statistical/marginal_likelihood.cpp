// Marginal Likelihood Filter (Navid Dianati): significance p-value per edge of an integer-weighted graph under a
// configuration-model null. Only the undirected variant is ported; the Python directed path cannot run
// (_pvalue_directed is called with the wrong keyword names).
#include <algorithm>
#include <stdexcept>

#include "detail.hpp"

namespace cppbone {

// Clip of binomtest p-values (MAX_NEG_LOG in the Python source).
static constexpr double MAX_NEG_LOG = 0;

// P(X >= w) for X ~ Binomial(q, ku*kv / (2 q^2)), statsmodels binom_test(alternative="larger").
static double pvalue_undirected(double w, double ku, double kv, double q) {
    double p = ku * kv * 1.0 / q / q / 2.0;
    return binom_sf(w - 1, q, p);
}

Backbone marginal_likelihood(const Graph& data) {
    for (const auto& e : data.edges())
        if (data.get(e.id, "weight") < 0) throw std::invalid_argument("All weights must be non-negative");

    int n = data.number_of_nodes();
    std::vector<double> ks(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) ks[u] = data.degree(u, "weight");
    double total_degree = 0;
    for (int u = 0; u < n; ++u) total_degree += ks[u];

    auto edges = data.edges();
    int m = (int)edges.size();
    std::vector<double> p(m);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < m; ++i) {
        const auto& e = edges[i];
        p[i] = std::max(MAX_NEG_LOG, pvalue_undirected(data.get(e.id, "weight"), ks[e.u], ks[e.v], total_degree / 2.0));
    }

    // The Python version round-trips through a source/target/weight/p_value DataFrame, so only those survive.
    Graph g;
    for (int i = 0; i < m; ++i) {
        const auto& e = edges[i];
        g.add_edge(data.label(e.u), data.label(e.v), {{"weight", data.get(e.id, "weight")}, {"p_value", p[i]}});
    }
    return {std::move(g), "Marginal Likelihood Filter", "p_value", true, {threshold_filter, fraction_filter}, FilterOn::Edges};
}

}  // namespace cppbone
