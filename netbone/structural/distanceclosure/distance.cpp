// Pairwise Jaccard proximity between the rows of a dense matrix (Rion Brattig Correia).
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

#include "netbone/netbone.hpp"

namespace netbone {

using Row = std::vector<double>;

static double jaccard_coef_scipy(const Row& u, const Row& v, double min_support) {
    double su = 0, sv = 0, nonzero = 0, unequal = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        su += u[i], sv += v[i];
        bool nz = u[i] != 0 || v[i] != 0;
        nonzero += nz;
        unequal += nz && u[i] != v[i];
    }
    if (su + sv < min_support) return 0.;
    return 1 - (nonzero ? unequal / nonzero : 0);  // scipy jaccard distance is 0 for two all-zero rows
}

static double jaccard_coef_binary(const Row& u, const Row& v, double min_support) {
    double inter = 0, uni = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        inter += u[i] != 0 && v[i] != 0;
        uni += u[i] != 0 || v[i] != 0;
    }
    if (uni < min_support) return 0.;
    return inter / uni;
}

static double jaccard_coef_set(const Row& u, const Row& v, double min_support) {
    std::set<double> a(u.begin(), u.end()), b(v.begin(), v.end());
    double inter = 0;
    for (double x : a) inter += b.count(x);
    double union_len = a.size() + b.size() - inter;
    if (union_len < min_support) return 0.;
    return inter / union_len;
}

static double jaccard_coef_weighted(const Row& u, const Row& v, double min_support) {
    double sum_min = 0, sum_max = 0;
    for (size_t i = 0; i < u.size(); ++i) sum_min += std::min(u[i], v[i]), sum_max += std::max(u[i], v[i]);
    if (sum_max < min_support) return 0.;
    return sum_min / sum_max;
}

// min_support < 0 selects each metric's Python default (1, or 10 for weighted).
Matrix pairwise_proximity(const Matrix& m, const std::string& metric, double min_support) {
    for (const auto& r : m)
        for (double x : r)
            if (x < 0) throw std::invalid_argument("Matrix cannot have negative numbers");

    double (*coef)(const Row&, const Row&, double);
    double default_support = 1;
    if (metric == "jaccard" || metric == "scipy" || metric == "j") {
        coef = jaccard_coef_scipy;
    } else if (metric == "jaccard_binary" || metric == "jb") {
        coef = jaccard_coef_binary;
    } else if (metric == "jaccard_set" || metric == "js") {
        std::cerr << "UserWarning: Zeros will be considered as a item set.\n";
        coef = jaccard_coef_set;
    } else if (metric == "jaccard_weighted" || metric == "weighted_jaccard" || metric == "wj") {
        coef = jaccard_coef_weighted;
        default_support = 10;
    } else {
        throw std::invalid_argument("Metric kind should be one of: 'jaccard' 'scipy' 'jaccard_binary' 'jb' "
                                    "'jaccard_set' 'js' 'jaccard_weighted' 'weighted_jaccard' 'wj'");
    }
    if (min_support < 0) min_support = default_support;

    size_t n = m.size();
    Matrix pm(n, Row(n, 1.0));
    for (size_t i = 0; i + 1 < n; ++i)
        for (size_t j = i + 1; j < n; ++j) pm[i][j] = pm[j][i] = coef(m[i], m[j], min_support);
    return pm;
}

}  // namespace netbone
