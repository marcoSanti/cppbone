/// @file
/// @brief Pairwise Jaccard proximity between the rows of a dense matrix (Rion Brattig Correia;
///        distanceclosure/distance.py).

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

#include "cppbone/cppbone.hpp"

namespace cppbone {

/// @brief One row of a dense matrix.
using Row = std::vector<double>;

/// @brief Jaccard proximity as scipy computes it: 1 - (positions where exactly one row is non-zero)
/// / (positions
///        where either is non-zero).
///
/// Current scipy (1.18) compares the rows as booleans (non-zero or not), not their values.
/// @param u First row.
/// @param v Second row.
/// @param min_support Rows whose sums add up to less than this get 0.
/// @return The proximity; 1 for two all-zero rows.
static double jaccard_coef_scipy(const Row &u, const Row &v, const double min_support) {
    double su = 0, sv = 0, nonzero = 0, unequal = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        su += u[i], sv += v[i];
        const bool a = u[i] != 0;
        const bool b = v[i] != 0;
        nonzero += a || b;
        unequal += a != b;
    }
    if (su + sv < min_support) {
        return 0.;
    }
    return 1 -
           (nonzero ? unequal / nonzero : 0); // scipy jaccard distance is 0 for two all-zero rows
}

/// @brief Binary Jaccard proximity: positions non-zero in both rows over positions non-zero in
/// either.
/// @param u First row.
/// @param v Second row.
/// @param min_support Pairs whose union is smaller than this get 0.
/// @return The proximity.
static double jaccard_coef_binary(const Row &u, const Row &v, double min_support) {
    double inter = 0, uni = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        inter += u[i] != 0 && v[i] != 0;
        uni += u[i] != 0 || v[i] != 0;
    }
    if (uni < min_support) {
        return 0.;
    }
    return inter / uni;
}

/// @brief Set Jaccard proximity: the rows' distinct values compared as sets (zeros count as an
/// item).
/// @param u First row.
/// @param v Second row.
/// @param min_support Pairs whose union is smaller than this get 0.
/// @return The proximity.
static double jaccard_coef_set(const Row &u, const Row &v, double min_support) {
    const std::set a(u.begin(), u.end());
    const std::set b(v.begin(), v.end());
    double inter = 0;
    for (double x : a) {
        inter += b.count(x);
    }
    const double union_len = a.size() + b.size() - inter;
    if (union_len < min_support) {
        return 0.;
    }
    return inter / union_len;
}

/// @brief Weighted Jaccard proximity: sum of element-wise minima over sum of element-wise maxima.
/// @param u First row.
/// @param v Second row.
/// @param min_support Pairs whose sum of maxima is smaller than this get 0.
/// @return The proximity.
static double jaccard_coef_weighted(const Row &u, const Row &v, const double min_support) {
    double sum_min = 0, sum_max = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        sum_min += std::min(u[i], v[i]), sum_max += std::max(u[i], v[i]);
    }
    if (sum_max < min_support) {
        return 0.;
    }
    return sum_min / sum_max;
}

/// A negative min_support selects each metric's Python default (1, or 10 for weighted).
Matrix pairwise_proximity(const Matrix &m, const std::string &metric, double min_support) {
    for (const auto &r : m) {
        for (const double x : r) {
            if (x < 0) {
                throw std::invalid_argument("Matrix cannot have negative numbers");
            }
        }
    }

    double (*coef)(const Row &, const Row &, double);
    double default_support = 1;
    if (metric == "jaccard" || metric == "scipy" || metric == "j") {
        coef = jaccard_coef_scipy;
    } else if (metric == "jaccard_binary" || metric == "jb") {
        coef = jaccard_coef_binary;
    } else if (metric == "jaccard_set" || metric == "js") {
        std::cerr << "UserWarning: Zeros will be considered as a item set.\n";
        coef = jaccard_coef_set;
    } else if (metric == "jaccard_weighted" || metric == "weighted_jaccard" || metric == "wj") {
        coef            = jaccard_coef_weighted;
        default_support = 10;
    } else {
        throw std::invalid_argument(
            "Metric kind should be one of: 'jaccard' 'scipy' 'jaccard_binary' 'jb' "
            "'jaccard_set' 'js' 'jaccard_weighted' 'weighted_jaccard' 'wj'");
    }
    if (min_support < 0) {
        min_support = default_support;
    }

    const size_t n = m.size();
    Matrix pm(n, Row(n, 1.0));
    for (size_t i = 0; i + 1 < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            pm[i][j] = pm[j][i] = coef(m[i], m[j], min_support);
        }
    }
    return pm;
}

} // namespace cppbone
