#pragma once

/// @file
/// @brief Internal helpers shared by the method implementations: OpenMP glue and a reusable Dijkstra.
///
/// Loops stay OpenMP 2.0 compatible (signed int indices, no atomic read/write, no min/max reductions) for MSVC.

#include <exception>
#include <tuple>
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

/// @brief Per-thread scratch for repeated single-source searches; only the touched entries are reset between sources.
struct DijkstraWorkspace {
    std::vector<double> dist;       ///< Final distance of the settled nodes, INF otherwise.
    std::vector<double> seen;       ///< Best tentative distance found so far.
    std::vector<char> done;         ///< Whether a node is settled.
    std::vector<int> parent_edge;   ///< Edge to the shortest-path-tree parent, -1 for the source and unreached nodes.
    std::vector<int> settled;       ///< Nodes in the order they were settled.
    std::vector<int> touched;       ///< Nodes whose entries must be reset before the next search.
    std::vector<std::tuple<double, long, int>> heap;  ///< Min-heap of (distance, insertion count, node).

    /// @brief Allocates the scratch arrays for a graph.
    /// @param n Number of nodes.
    explicit DijkstraWorkspace(int n) : dist(n, INF), seen(n, INF), done(n, 0), parent_edge(n, -1) {}
};

/// @brief single_source_dijkstra_path_length on a reusable workspace, with the parent recorded as an edge id.
///
/// Uses the same tie-breaking (insertion order) as the public function.
/// @param g Graph.
/// @param source Start node.
/// @param weight Edge distances indexed by edge id.
/// @param disjunction How distances combine along a path.
/// @param ws Workspace; on return ws.dist, ws.parent_edge and ws.settled describe the search.
/// @param cutoff Stop once the next node to settle is at distance >= cutoff; nodes not settled keep dist = INF.
/// @param removed If given, edges with a non-zero entry are ignored.
/// @throws std::invalid_argument when a negative distance makes the paths contradictory.
void dijkstra(const Graph& g, int source, const std::vector<double>& weight, Disjunction disjunction,
              DijkstraWorkspace& ws, double cutoff = INF, const std::vector<char>* removed = nullptr);

}  // namespace cppbone::detail
