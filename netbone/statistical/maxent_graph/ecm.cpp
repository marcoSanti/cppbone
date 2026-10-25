// (Undirected) Enhanced Configuration Model, ported from maxent_graph (ecm.py, MaxentGraph.py, ecm_main.py).
// Same parametrisation as Python: x = exp(a), y = sigmoid(b). Python minimises the negative log-likelihood with
// scipy trust-krylov and JAX autodiff; here gradient and Hessian are closed form (degree/strength residuals and
// per-pair covariances, chained through the transform) and the solver is Levenberg-Marquardt. Steps are judged on
// the change in the objective summed term by term, since the objective itself (~1e8 on real data) cannot resolve
// the small decreases near the optimum.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "netbone/netbone.hpp"

namespace netbone {

namespace {

constexpr double EPS = std::numeric_limits<double>::epsilon();

// In-place Cholesky solve of H d = r; false if H is not positive definite.
bool cholesky_solve(std::vector<double> H, std::vector<double>& r, int n) {
    for (int j = 0; j < n; ++j) {
        double d = H[j * n + j];
        for (int k = 0; k < j; ++k) d -= H[j * n + k] * H[j * n + k];
        if (!(d > 0)) return false;
        d = std::sqrt(d);
        H[j * n + j] = d;
        for (int i = j + 1; i < n; ++i) {
            double v = H[i * n + j];
            for (int k = 0; k < j; ++k) v -= H[i * n + k] * H[j * n + k];
            H[i * n + j] = v / d;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < i; ++k) r[i] -= H[i * n + k] * r[k];
        r[i] /= H[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int k = i + 1; k < n; ++k) r[i] -= H[k * n + i] * r[k];
        r[i] /= H[i * n + i];
    }
    return true;
}

struct ECM {
    std::vector<double> k, s;  // observed degree and strength sequences
    int n = 0;

    static double sigmoid(double b) { return 1 / (1 + std::exp(-b)); }
    static double log_sigmoid(double b) { return b >= 0 ? -std::log1p(std::exp(-b)) : b - std::log1p(std::exp(b)); }

    struct Pair {
        double p, sij, one_m_yy, yy, r;  // r = xx yy / (1 - yy); log-likelihood term is -log1p(r)
    };
    Pair pair(const std::vector<double>& v, int i, int j) const {
        double yi = sigmoid(v[n + i]), yj = sigmoid(v[n + j]);
        double yy = yi * yj;
        double one_m_yy = (1 - yi) + yi * (1 - yj);  // 1 - yi yj without cancellation
        double xxyy = std::exp(v[i] + v[j]) * yy;
        double denom = one_m_yy + xxyy;
        double p = xxyy / denom;
        return {p, p / one_m_yy, one_m_yy, yy, xxyy / one_m_yy};
    }

    std::vector<double> get_initial_guess(std::mt19937& rng) const {  // option 5, clipped to the bounds
        double num_edges = 0;
        for (double ki : k) num_edges += ki;
        num_edges /= 2;
        std::uniform_real_distribution<double> unif(0, 1);
        std::vector<double> v(2 * n);
        for (int i = 0; i < n; ++i) {
            v[i] = std::log(std::max(EPS, k[i] / std::sqrt(num_edges)));
            double y = std::clamp(unif(rng), EPS, 1 - EPS);
            v[n + i] = -std::log(1 / y - 1);
        }
        return v;
    }

    // Expected (k, s) sequence, gradient and Hessian of the negative log-likelihood in (a, b).
    void derivatives(const std::vector<double>& v, std::vector<double>& expected, std::vector<double>& grad,
                     std::vector<double>& H) const {
        int m = 2 * n;
        expected.assign(m, 0);
        H.assign(size_t(m) * m, 0);
        std::vector<double> jac(n);  // d log y / d b = 1 - y
        for (int i = 0; i < n; ++i) jac[i] = 1 - sigmoid(v[n + i]);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j) {
                Pair q = pair(v, i, j);
                expected[i] += q.p, expected[j] += q.p;
                expected[n + i] += q.sij, expected[n + j] += q.sij;
                // covariances of (edge present, edge weight) w.r.t. the natural parameters (a, log y)
                double var_a = q.p * (1 - q.p);
                double cov = q.sij * (1 - q.p);
                double var_w = q.p * (1 + q.yy) / (q.one_m_yy * q.one_m_yy) - q.sij * q.sij;
                for (int r : {i, j})
                    for (int c : {i, j}) {
                        H[size_t(r) * m + c] += var_a;
                        H[size_t(r) * m + n + c] += cov * jac[c];
                        H[size_t(n + r) * m + c] += cov * jac[r];
                        H[size_t(n + r) * m + n + c] += var_w * jac[r] * jac[c];
                    }
            }
        grad.resize(m);
        for (int i = 0; i < n; ++i) {
            double g_logy = expected[n + i] - s[i];
            grad[i] = expected[i] - k[i];
            grad[n + i] = jac[i] * g_logy;
            H[size_t(n + i) * m + n + i] -= g_logy * (1 - jac[i]) * jac[i];  // d2 log y / d b2 = -y (1 - y)
        }
    }

    // f(w) - f(v) for the negative log-likelihood, summed from per-term differences.
    double delta_nll(const std::vector<double>& v, const std::vector<double>& w) const {
        double d = 0;
        for (int i = 0; i < n; ++i) {
            d -= k[i] * (w[i] - v[i]);
            d -= s[i] * (log_sigmoid(w[n + i]) - log_sigmoid(v[n + i]));
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j) {
                double r0 = pair(v, i, j).r, r1 = pair(w, i, j).r;
                d += std::log1p((r1 - r0) / (1 + r0));
            }
        return d;
    }

    double max_relative_error(const std::vector<double>& expected) const {
        double mx = 0;
        for (int i = 0; i < n; ++i) {
            mx = std::max(mx, std::abs(expected[i] - k[i]) / (1 + std::abs(k[i])));
            mx = std::max(mx, std::abs(expected[n + i] - s[i]) / (1 + std::abs(s[i])));
        }
        return mx;
    }

    std::vector<double> solve(std::vector<double> v) const {
        int m = 2 * n;
        std::vector<double> expected, g, H;
        derivatives(v, expected, g, H);
        double scale = 0;
        for (int i = 0; i < m; ++i) scale = std::max(scale, std::abs(H[size_t(i) * m + i]));
        double lambda = 1e-6 * scale;
        bool success = false;
        for (int it = 0; it < 1000; ++it) {
            double gnorm = 0;
            for (double x : g) gnorm += x * x;
            // same stopping rule as scipy trust-krylov (gtol=1e-8), or an essentially exact fit
            if (std::sqrt(gnorm) < 1e-8 || max_relative_error(expected) < 1e-12) {
                success = true;
                break;
            }
            bool stepped = false;
            while (lambda < 1e30 && !stepped) {
                std::vector<double> Hr = H, d = g;
                for (int i = 0; i < m; ++i) Hr[size_t(i) * m + i] += lambda;
                if (!cholesky_solve(Hr, d, m)) {
                    lambda = std::max(lambda * 10, 1e-12);
                    continue;
                }
                std::vector<double> w(m);
                for (int i = 0; i < m; ++i) w[i] = v[i] - d[i];
                double df = delta_nll(v, w);
                if (std::isfinite(df) && df < 0) {
                    v.swap(w);
                    lambda = std::max(lambda / 3, 1e-12);
                    stepped = true;
                } else {
                    lambda *= 10;
                }
            }
            if (!stepped) break;  // no decrease possible: numerical optimum
            derivatives(v, expected, g, H);
        }
        if (!success) {
            if (max_relative_error(expected) < 0.5)
                std::cerr << "RuntimeWarning: Didn't succeed according to algorithm, but max relative error is low.\n";
            else
                throw std::runtime_error("Didn't succeed in minimization.");
        }
        return v;
    }
};

}  // namespace

Backbone ecm(const Graph& data, std::uint32_t seed) {
    Graph g = data;
    auto w = g.edge_values("weight");
    ECM model;
    model.n = g.number_of_nodes();
    model.k.assign(model.n, 0);
    model.s.assign(model.n, 0);
    for (const auto& e : g.edges()) {
        if (e.u == e.v) continue;  // ignore self-loops
        double we = w[e.id];
        for (int i : {e.u, e.v}) {
            model.k[i] += we > 0;
            model.s[i] += we;
        }
    }
    std::mt19937 rng(seed);
    auto solution = model.solve(model.get_initial_guess(rng));
    int n = model.n;
    for (const auto& e : g.edges()) {
        if (e.u == e.v || w[e.id] == 0) continue;
        double xx = std::exp(solution[e.u] + solution[e.v]);
        double yy = ECM::sigmoid(solution[n + e.u]) * ECM::sigmoid(solution[n + e.v]);
        double pij = xx * yy / (1 - yy + xx * yy);
        // probability this weight would be at least this large, given null model
        g.set(e.id, "p_value", pij * std::pow(yy, w[e.id] - 1));
    }
    return {g, "Enhanced Configuration Model Filter", "p_value", true, {threshold_filter, fraction_filter},
            FilterOn::Edges};
}

}  // namespace netbone
