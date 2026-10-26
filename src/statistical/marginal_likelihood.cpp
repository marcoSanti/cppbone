/// @file
/// @brief Marginal Likelihood Filter (Navid Dianati; statistical/marginal_likelihood.py).
///
/// Significance p-value per edge of an integer-weighted graph under a configuration-model null.
/// Only the undirected variant is ported; the Python directed path cannot run (_pvalue_directed is
/// called with the wrong keyword names).
#include <algorithm>
#include <numeric>
#include <ranges>
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
    const auto edges = data.edges();
    if (std::ranges::any_of(edges,
                            [&](const EdgeRef &e) { return data.get(e.id, "weight") < 0; })) {
        throw std::invalid_argument("All weights must be non-negative");
    }

    const auto ks             = detail::strengths(data, "weight");
    const double total_degree = std::accumulate(ks.begin(), ks.end(), 0.0);

    const int m = static_cast<int>(edges.size());
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
    for (const auto &[e, p_value] : std::views::zip(edges, p)) {
        g.add_edge(data.label(e.u), data.label(e.v),
                   {{"weight", data.get(e.id, "weight")}, {"p_value", p_value}});
    }
    return {.graph         = std::move(g),
            .method_name   = "Marginal Likelihood Filter",
            .property_name = "p_value",
            .ascending     = true,
            .filters       = {threshold_filter, fraction_filter}};
}

} // namespace cppbone
