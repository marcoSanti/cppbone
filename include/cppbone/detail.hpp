#pragma once

/// @file
/// @brief Internal helpers shared by the method implementations: OpenMP glue, node strengths, the
/// endpoint minimum and a dense matrix.
///
/// Loops stay OpenMP 2.0 compatible (signed int indices, no atomic read/write, no min/max
/// reductions) for MSVC.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cppbone/cppbone.hpp"

/// @brief Implementation details, not part of the public API.
namespace cppbone::detail {

/// @brief Returns how many threads the next parallel region will use.
/// @return omp_get_max_threads(), or 1 without OpenMP.
inline int thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/// @brief Returns the calling thread's number inside a parallel region.
/// @return omp_get_thread_num(), or 0 without OpenMP.
inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/// @brief Carries exceptions out of an OpenMP region, which they must not leave on their own.
///
/// run() records the first exception thrown by any thread; rethrow() raises it after the region.
class Errors {
public:
    /// @brief Runs a callable, recording the exception it throws if it is the first one.
    /// @param f Callable taking no arguments.
    template <typename F> void run(F &&f) {
        try {
            f();
        } catch (...) {
#pragma omp critical(cppbone_errors)
            if (!first_) {
                first_ = std::current_exception();
            }
        }
    }

    /// @brief Rethrows the recorded exception, if any.
    void rethrow() const {
        if (first_) {
            std::rethrow_exception(first_);
        }
    }

private:
    std::exception_ptr first_; ///< First exception recorded.
};

/// @brief Weighted degree (strength) of every node, computed in parallel.
/// @param g Graph.
/// @param weight Attribute to sum (see Graph::degree).
/// @return One strength per node.
[[nodiscard]] inline std::vector<double> strengths(const Graph &g, const std::string_view weight) {
    const int n = g.number_of_nodes();
    std::vector<double> s(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        s[u] = g.degree(u, weight);
    }
    return s;
}

/// @brief Smallest per-endpoint score of an edge, as the Python loops over nodes compute it.
///
/// The endpoints are visited in node order (a self-loop once) and the ones `skip` rejects are
/// passed over; a NaN kept so far is replaced by the next value.
/// @param u One endpoint.
/// @param v The other endpoint.
/// @param skip Callable (node) -> bool: true leaves this endpoint out.
/// @param score Callable (node) -> double: the edge's score seen from this endpoint.
/// @return The minimum, or nothing when both endpoints are skipped.
template <typename Skip, typename Score>
[[nodiscard]] std::optional<double> endpoint_min(const int u, const int v, Skip &&skip,
                                                 Score &&score) {
    const auto [a, b] = std::minmax(u, v);
    std::optional<double> best;
    for (const int node : {a, b}) {
        if (node == b && a == b) {
            break; // a self-loop is visited once
        }
        if (skip(node)) {
            continue;
        }
        if (const double x = score(node); !best || std::isnan(*best) || x < *best) {
            best = x;
        }
    }
    return best;
}

/// @brief Dense row-major matrix of doubles, indexed as m[row, column].
class DenseMatrix {
public:
    /// @brief Creates an empty matrix.
    DenseMatrix() = default;

    /// @brief Creates a rows x cols matrix.
    /// @param rows Row count.
    /// @param cols Column count.
    /// @param value Initial value of every entry.
    DenseMatrix(const std::size_t rows, const std::size_t cols, const double value = 0)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    /// @brief Accesses one entry (const or not, following the matrix).
    /// @param r Row.
    /// @param c Column.
    /// @return Reference to the entry.
    template <typename Self>
    [[nodiscard]] auto &&operator[](this Self &&self, const std::size_t r, const std::size_t c) {
        return std::forward_like<Self>(self.data_[r * self.cols_ + c]);
    }

    /// @brief Views one row.
    /// @param r Row.
    /// @return The row's entries (read-only for a const matrix).
    template <typename Self> [[nodiscard]] auto row(this Self &self, const std::size_t r) {
        return std::span(self.data_).subspan(r * self.cols_, self.cols_);
    }

    /// @brief Returns the number of rows.
    /// @return Row count.
    [[nodiscard]] std::size_t rows() const { return rows_; }

    /// @brief Returns the number of columns.
    /// @return Column count.
    [[nodiscard]] std::size_t cols() const { return cols_; }

    /// @brief Returns the entries, row after row (for LAPACK).
    /// @return Pointer to the first entry.
    [[nodiscard]] double *data() { return data_.data(); }

    /// @brief Resizes the matrix and sets every entry.
    /// @param rows Row count.
    /// @param cols Column count.
    /// @param value Value of every entry.
    void assign(const std::size_t rows, const std::size_t cols, const double value) {
        rows_ = rows, cols_ = cols;
        data_.assign(rows * cols, value);
    }

private:
    std::size_t rows_ = 0;     ///< Row count.
    std::size_t cols_ = 0;     ///< Column count.
    std::vector<double> data_; ///< Entries, row-major.
};

} // namespace cppbone::detail
