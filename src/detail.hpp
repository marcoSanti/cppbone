#pragma once

/// @file
/// @brief Internal helpers shared by the method implementations: OpenMP glue.
///
/// Loops stay OpenMP 2.0 compatible (signed int indices, no atomic read/write, no min/max reductions) for MSVC.

#include <exception>
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
    template <typename F>
    void run(F&& f) {
        try {
            f();
        } catch (...) {
#pragma omp critical(cppbone_errors)
            if (!first_) first_ = std::current_exception();
        }
    }

    /// @brief Rethrows the recorded exception, if any.
    void rethrow() const {
        if (first_) std::rethrow_exception(first_);
    }

private:
    std::exception_ptr first_;  ///< First exception recorded.
};

}  // namespace cppbone::detail
