// Marginal Likelihood Filter (Navid Dianati): significance p-value per edge of an integer-weighted graph under a
// configuration-model null. Only the undirected variant is ported; the Python directed path cannot run
// (_pvalue_directed is called with the wrong keyword names).
#include <algorithm>
#include <stdexcept>

#include "netbone/netbone.hpp"

namespace netbone {

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

    double total_degree = 0;
    std::vector<double> ks(data.number_of_nodes());
    for (int u = 0; u < data.number_of_nodes(); ++u) total_degree += ks[u] = data.degree(u, "weight");

    // The Python version round-trips through a source/target/weight/p_value DataFrame, so only those survive.
    Graph g;
    for (const auto& e : data.edges()) {
        double w = data.get(e.id, "weight");
        double p = pvalue_undirected(w, ks[e.u], ks[e.v], total_degree / 2.0);
        g.add_edge(data.label(e.u), data.label(e.v),
                   {{"weight", w}, {"p_value", std::max(MAX_NEG_LOG, p)}});
    }
    return {g, "Marginal Likelihood Filter", "p_value", true, {threshold_filter, fraction_filter}, FilterOn::Edges};
}

}  // namespace netbone
