/// @file
/// @brief Marginal Likelihood Filter (Navid Dianati; statistical/marginal_likelihood.py).
///
/// Significance p-value per edge of an integer-weighted graph under a configuration-model null.
/// Only the undirected variant is ported; the Python directed path cannot run (_pvalue_directed is
/// called with the wrong keyword names).
#include <algorithm>
#include <stdexcept>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

/// @brief Lower clip of the p-values (MAX_NEG_LOG in the Python source).
static constexpr double MAX_NEG_LOG = 0;

/// @brief Upper-tail binomial p-value of one undirected edge (statsmodels binom_test,
/// alternative="larger").
/// @param w Edge weight.
/// @param ku Strength of one endpoint.
/// @param kv Strength of the other endpoint.
/// @param q Total weight (half the sum of strengths).
/// @return P(X >= w) for X ~ Binomial(q, ku kv / (2 q^2)).
static double pvalue_undirected(double w, double ku, double kv, double q) {
    const double p = ku * kv * 1.0 / q / q / 2.0;
    return binom_sf(w - 1, q, p);
}

Backbone marginal_likelihood(const Graph &data) {
    for (const auto &e : data.edges()) {
        if (data.get(e.id, "weight") < 0) {
            throw std::invalid_argument("All weights must be non-negative");
        }
    }

    const int n = data.number_of_nodes();
    std::vector<double> ks(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        ks[u] = data.degree(u, "weight");
    }
    double total_degree = 0;
    for (int u = 0; u < n; ++u) {
        total_degree += ks[u];
    }

    const auto edges = data.edges();
    const int m      = static_cast<int>(edges.size());
    std::vector<double> p(m);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < m; ++i) {
        const auto &[u, v, id] = edges[i];
        p[i] = std::max(MAX_NEG_LOG, pvalue_undirected(data.get(id, "weight"), ks[u], ks[v],
                                                       total_degree / 2.0));
    }

    // The Python version round-trips through a source/target/weight/p_value DataFrame, so only
    // those survive.
    Graph g;
    for (int i = 0; i < m; ++i) {
        const auto &[u, v, id] = edges[i];
        g.add_edge(data.label(u), data.label(v),
                   {{"weight", data.get(id, "weight")}, {"p_value", p[i]}});
    }
    return {std::move(g), "Marginal Likelihood Filter",        "p_value",
            true,         {threshold_filter, fraction_filter}, FilterOn::Edges};
}

} // namespace cppbone
