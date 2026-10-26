/// @file
/// @brief Noise Corrected filter (statistical/noise_corrected.py).

#include <cmath>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

Backbone noise_corrected(const Graph &data, bool approximation) {
    Graph g          = data;
    const auto edges = g.edges();
    double n         = 0;
    for (const auto &e : edges) {
        n += g.get(e.id, "weight");
    }
    const auto strength = detail::strengths(g, "weight");

    const int m = static_cast<int>(edges.size());
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < m; ++i) {
        const auto &[u, v, id]              = edges[i];
        const double w                      = g.get(id, "weight");
        const double ni                     = strength[u];
        const double nj                     = strength[v];
        const double mean_prior_probability = ((ni * nj) / n) * (1 / n);
        if (approximation) {
            g.set(id, "p_value", 1 - binom_cdf(w, n, mean_prior_probability));
        } else {
            const double kappa = n / (ni * nj);
            const double score = ((kappa * w) - 1) / ((kappa * w) + 1);
            const double var_prior_probability =
                (1 / (n * n)) * (ni * nj * (n - ni) * (n - nj)) / ((n * n) * (n - 1));
            const double alpha_prior =
                (((mean_prior_probability * mean_prior_probability) / var_prior_probability) *
                 (1 - mean_prior_probability)) -
                mean_prior_probability;
            const double beta_prior = (mean_prior_probability / var_prior_probability) *
                                          (1 - (mean_prior_probability * mean_prior_probability)) -
                                      (1 - mean_prior_probability);
            const double alpha_post   = alpha_prior + w;
            const double beta_post    = n - w + beta_prior;
            const double expected_pij = alpha_post / (alpha_post + beta_post);
            const double variance_nij = expected_pij * (1 - expected_pij) * n;
            const double d = (1.0 / (ni * nj)) - (n * ((ni + nj) / ((ni * nj) * (ni * nj))));
            const double variance_cij =
                variance_nij * std::pow((2 * (kappa + (w * d))) / std::pow((kappa * w) + 1, 2), 2);
            g.set(id, "nc_sdev", std::sqrt(variance_cij));
            g.set(id, "score", score);
        }
    }
    return {.graph         = std::move(g),
            .method_name   = "Noise Corrected Filter",
            .property_name = "p_value",
            .ascending     = true,
            .filters       = {threshold_filter, fraction_filter}};
}

} // namespace cppbone
