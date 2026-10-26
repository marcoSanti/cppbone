/// @file
/// @brief (Undirected) Enhanced Configuration Model, ported from maxent_graph (ecm.py,
/// MaxentGraph.py, ecm_main.py).
///
/// Same parametrisation as Python: x = exp(a), y = sigmoid(b). Python minimises the negative
/// log-likelihood with scipy trust-krylov and JAX autodiff; here gradient and Hessian are closed
/// form (degree/strength residuals and per-pair covariances, chained through the transform) and the
/// solver is Levenberg-Marquardt. Steps are judged on the change in the objective summed term by
/// term, since the objective itself (~1e8 on real data) cannot resolve the small decreases near the
/// optimum.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "detail.hpp"

namespace cppbone {

namespace {

/// @brief Machine epsilon, the margin that keeps the initial guess inside the parameter bounds.
constexpr double EPS = std::numeric_limits<double>::epsilon();

/// @brief Solves H d = r by Cholesky factorisation (the column updates run in parallel).
/// @param H Symmetric n x n matrix, row-major (taken by value: it is factorised in place).
/// @param r Right-hand side; replaced by the solution d.
/// @param n Matrix size.
/// @return false if H is not positive definite (r is then partially overwritten).
bool cholesky_solve(std::vector<double> H, std::vector<double> &r, int n) {
    for (int j = 0; j < n; ++j) {
        double d = H[j * n + j];
        for (int k = 0; k < j; ++k) {
            d -= H[j * n + k] * H[j * n + k];
        }
        if (!(d > 0)) {
            return false;
        }
        d            = std::sqrt(d);
        H[j * n + j] = d;
#pragma omp parallel for schedule(static) if (n - j > 256)
        for (int i = j + 1; i < n; ++i) {
            double v = H[i * n + j];
            for (int k = 0; k < j; ++k) {
                v -= H[i * n + k] * H[j * n + k];
            }
            H[i * n + j] = v / d;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < i; ++k) {
            r[i] -= H[i * n + k] * r[k];
        }
        r[i] /= H[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int k = i + 1; k < n; ++k) {
            r[i] -= H[k * n + i] * r[k];
        }
        r[i] /= H[i * n + i];
    }
    return true;
}

/// @brief The ECM likelihood for one network: observed sequences, derivatives and the solver.
///
/// The parameter vector v holds a_i = log x_i in v[0..n) and b_i = logit y_i in v[n..2n).
struct ECM {
    std::vector<double> k; ///< Observed degree of every node.
    std::vector<double> s; ///< Observed strength of every node.
    int n = 0;             ///< Number of nodes.

    /// @brief Logistic function.
    /// @param b Argument.
    /// @return 1 / (1 + exp(-b)).
    static double sigmoid(double b) { return 1 / (1 + std::exp(-b)); }

    /// @brief Logarithm of the logistic function, without overflow for large |b|.
    /// @param b Argument.
    /// @return log(sigmoid(b)).
    static double log_sigmoid(double b) {
        return b >= 0 ? -std::log1p(std::exp(-b)) : b - std::log1p(std::exp(b));
    }

    /// @brief Model quantities of one node pair.
    struct Pair {
        double p;        ///< Probability that the edge exists.
        double sij;      ///< Expected weight.
        double one_m_yy; ///< 1 - y_i y_j, computed without cancellation.
        double yy;       ///< y_i y_j.
        double r; ///< x_i x_j y_i y_j / (1 - y_i y_j); the pair's log-likelihood term is -log1p(r).
    };

    /// @brief Computes the model quantities of a pair.
    /// @param v Parameter vector.
    /// @param i First node; callers pass the larger index first so results do not depend on the
    /// order.
    /// @param j Second node.
    /// @return The pair's quantities.
    [[nodiscard]] Pair pair(const std::vector<double> &v, int i, int j) const {
        double yi = sigmoid(v[n + i]), yj = sigmoid(v[n + j]);
        double yy       = yi * yj;
        double one_m_yy = (1 - yi) + yi * (1 - yj); // 1 - yi yj without cancellation
        double xxyy     = std::exp(v[i] + v[j]) * yy;
        double denom    = one_m_yy + xxyy;
        double p        = xxyy / denom;
        return {p, p / one_m_yy, one_m_yy, yy, xxyy / one_m_yy};
    }

    /// @brief Starting point of the solver: maxent_graph's option 5, clipped to the bounds.
    /// @param rng Random generator for the y part.
    /// @return x_i = k_i / sqrt(edges) and uniformly random y_i, as a parameter vector.
    std::vector<double> get_initial_guess(std::mt19937 &rng) const {
        double num_edges = 0;
        for (double ki : k) {
            num_edges += ki;
        }
        num_edges /= 2;
        std::uniform_real_distribution<double> unif(0, 1);
        std::vector<double> v(2 * n);
        for (int i = 0; i < n; ++i) {
            v[i]     = std::log(std::max(EPS, k[i] / std::sqrt(num_edges)));
            double y = std::clamp(unif(rng), EPS, 1 - EPS);
            v[n + i] = -std::log(1 / y - 1);
        }
        return v;
    }

    /// @brief Expected sequences, gradient and Hessian of the negative log-likelihood at a point.
    /// @param v Parameter vector.
    /// @param expected Receives the expected degrees (first n) and strengths (last n).
    /// @param grad Receives the gradient with respect to v.
    /// @param H Receives the 2n x 2n Hessian, row-major.
    void derivatives(const std::vector<double> &v, std::vector<double> &expected,
                     std::vector<double> &grad, std::vector<double> &H) const {
        const int m = 2 * n;
        expected.assign(m, 0);
        H.assign(static_cast<size_t>(m) * m, 0);
        std::vector<double> jac(n); // d log y / d b = 1 - y
        for (int i = 0; i < n; ++i) {
            jac[i] = 1 - sigmoid(v[n + i]);
        }
        // Each node r gathers its own rows over every partner c, in increasing c: that is the order
        // in which the sequential loop over pairs (i > j) adds to them, so rows are independent and
        // the sums are unchanged. Each pair is evaluated twice (once per endpoint) to avoid sharing
        // rows between threads.
#pragma omp parallel for schedule(dynamic, 16)
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                if (c == r) {
                    continue;
                }
                Pair q = pair(v, std::max(r, c), std::min(r, c));
                expected[r] += q.p;
                expected[n + r] += q.sij;
                // covariances of (edge present, edge weight) w.r.t. the natural parameters (a, log
                // y)
                double var_a = q.p * (1 - q.p);
                double cov   = q.sij * (1 - q.p);
                double var_w = q.p * (1 + q.yy) / (q.one_m_yy * q.one_m_yy) - q.sij * q.sij;
                for (int col : {r, c}) {
                    H[size_t(r) * m + col] += var_a;
                    H[size_t(r) * m + n + col] += cov * jac[col];
                    H[size_t(n + r) * m + col] += cov * jac[r];
                    H[size_t(n + r) * m + n + col] += var_w * jac[r] * jac[col];
                }
            }
        }
        grad.resize(m);
        for (int i = 0; i < n; ++i) {
            double g_logy = expected[n + i] - s[i];
            grad[i]       = expected[i] - k[i];
            grad[n + i]   = jac[i] * g_logy;
            H[size_t(n + i) * m + n + i] -=
                g_logy * (1 - jac[i]) * jac[i]; // d2 log y / d b2 = -y (1 - y)
        }
    }

    /// @brief Change of the negative log-likelihood between two points, summed term by term for
    /// precision.
    /// @param v Current point.
    /// @param w Candidate point.
    /// @return f(w) - f(v).
    [[nodiscard]] double delta_nll(const std::vector<double> &v,
                                   const std::vector<double> &w) const {
        double d = 0;
        for (int i = 0; i < n; ++i) {
            d -= k[i] * (w[i] - v[i]);
            d -= s[i] * (log_sigmoid(w[n + i]) - log_sigmoid(v[n + i]));
        }
        // Row sums in parallel, then added in row order: deterministic, though rounded differently
        // than one running sum.
        std::vector<double> rows(n, 0);
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                double r0 = pair(v, i, j).r, r1 = pair(w, i, j).r;
                rows[i] += std::log1p((r1 - r0) / (1 + r0));
            }
        }
        for (double x : rows) {
            d += x;
        }
        return d;
    }

    /// @brief Largest relative error between the expected and observed sequences.
    /// @param expected Expected degrees and strengths, as from derivatives().
    /// @return max over nodes of |expected - observed| / (1 + |observed|).
    [[nodiscard]] double max_relative_error(const std::vector<double> &expected) const {
        double mx = 0;
        for (int i = 0; i < n; ++i) {
            mx = std::max(mx, std::abs(expected[i] - k[i]) / (1 + std::abs(k[i])));
            mx = std::max(mx, std::abs(expected[n + i] - s[i]) / (1 + std::abs(s[i])));
        }
        return mx;
    }

    /// @brief Minimises the negative log-likelihood with Levenberg-Marquardt.
    ///
    /// Stops when the gradient norm drops below 1e-8 (scipy's gtol) or the fit is essentially
    /// exact.
    /// @param v Starting point.
    /// @return The fitted parameter vector.
    /// @throws std::runtime_error if it fails with a max relative error of 0.5 or more (a warning
    /// is printed for
    ///         smaller errors).
    [[nodiscard]] std::vector<double> solve(std::vector<double> v) const {
        int m = 2 * n;
        std::vector<double> expected, g, H;
        derivatives(v, expected, g, H);
        double scale = 0;
        for (int i = 0; i < m; ++i) {
            scale = std::max(scale, std::abs(H[size_t(i) * m + i]));
        }
        double lambda = 1e-6 * scale;
        bool success  = false;
        for (int it = 0; it < 1000; ++it) {
            double gnorm = 0;
            for (double x : g) {
                gnorm += x * x;
            }
            // same stopping rule as scipy trust-krylov (gtol=1e-8), or an essentially exact fit
            if (std::sqrt(gnorm) < 1e-8 || max_relative_error(expected) < 1e-12) {
                success = true;
                break;
            }
            bool stepped = false;
            while (lambda < 1e30 && !stepped) {
                std::vector<double> Hr = H, d = g;
                for (int i = 0; i < m; ++i) {
                    Hr[size_t(i) * m + i] += lambda;
                }
                if (!cholesky_solve(Hr, d, m)) {
                    lambda = std::max(lambda * 10, 1e-12);
                    continue;
                }
                std::vector<double> w(m);
                for (int i = 0; i < m; ++i) {
                    w[i] = v[i] - d[i];
                }
                double df = delta_nll(v, w);
                if (std::isfinite(df) && df < 0) {
                    v.swap(w);
                    lambda  = std::max(lambda / 3, 1e-12);
                    stepped = true;
                } else {
                    lambda *= 10;
                }
            }
            if (!stepped) {
                break; // no decrease possible: numerical optimum
            }
            derivatives(v, expected, g, H);
        }
        if (!success) {
            if (max_relative_error(expected) < 0.5) {
                std::cerr << "RuntimeWarning: Didn't succeed according to algorithm, but max "
                             "relative error is low.\n";
            } else {
                throw std::runtime_error("Didn't succeed in minimization.");
            }
        }
        return v;
    }
};

} // namespace

Backbone ecm(const Graph &data, std::uint32_t seed) {
    Graph g = data;
    auto w  = g.edge_values("weight");
    ECM model;
    model.n = g.number_of_nodes();
    model.k.assign(model.n, 0);
    model.s.assign(model.n, 0);
    for (const auto &e : g.edges()) {
        if (e.u == e.v) {
            continue; // ignore self-loops
        }
        double we = w[e.id];
        for (int i : {e.u, e.v}) {
            model.k[i] += we > 0;
            model.s[i] += we;
        }
    }
    std::mt19937 rng(seed);
    const auto solution = model.solve(model.get_initial_guess(rng));
    const int n         = model.n;
    for (const auto &[u, v, id] : g.edges()) {
        if (u == v || w[id] == 0) {
            continue;
        }
        const double xx  = std::exp(solution[u] + solution[v]);
        const double yy  = ECM::sigmoid(solution[n + u]) * ECM::sigmoid(solution[n + v]);
        const double pij = xx * yy / (1 - yy + xx * yy);
        // probability this weight would be at least this large, given null model
        g.set(id, "p_value", pij * std::pow(yy, w[id] - 1));
    }
    return {std::move(g), "Enhanced Configuration Model Filter", "p_value",
            true,         {threshold_filter, fraction_filter},   FilterOn::Edges};
}

} // namespace cppbone
