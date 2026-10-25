#include <cmath>

#include "netbone/netbone.hpp"

namespace netbone {

Backbone noise_corrected(const Graph& data, bool approximation) {
    Graph g = data;
    double n = 0;
    for (const auto& e : g.edges()) n += g.get(e.id, "weight");

    for (const auto& e : g.edges()) {
        double w = g.get(e.id, "weight");
        double ni = g.degree(e.u, "weight");
        double nj = g.degree(e.v, "weight");
        double mean_prior_probability = ((ni * nj) / n) * (1 / n);
        if (approximation) {
            g.set(e.id, "p_value", 1 - binom_cdf(w, n, mean_prior_probability));
        } else {
            double kappa = n / (ni * nj);
            double score = ((kappa * w) - 1) / ((kappa * w) + 1);
            double var_prior_probability = (1 / (n * n)) * (ni * nj * (n - ni) * (n - nj)) / ((n * n) * (n - 1));
            double alpha_prior = (((mean_prior_probability * mean_prior_probability) / var_prior_probability) *
                                  (1 - mean_prior_probability)) -
                                 mean_prior_probability;
            double beta_prior = (mean_prior_probability / var_prior_probability) *
                                    (1 - (mean_prior_probability * mean_prior_probability)) -
                                (1 - mean_prior_probability);
            double alpha_post = alpha_prior + w;
            double beta_post = n - w + beta_prior;
            double expected_pij = alpha_post / (alpha_post + beta_post);
            double variance_nij = expected_pij * (1 - expected_pij) * n;
            double d = (1.0 / (ni * nj)) - (n * ((ni + nj) / ((ni * nj) * (ni * nj))));
            double variance_cij = variance_nij * std::pow((2 * (kappa + (w * d))) / std::pow((kappa * w) + 1, 2), 2);
            g.set(e.id, "nc_sdev", std::sqrt(variance_cij));
            g.set(e.id, "score", score);
        }
    }
    return {g, "Noise Corrected Filter", "p_value", true, {threshold_filter, fraction_filter}, FilterOn::Edges};
}

}  // namespace netbone
