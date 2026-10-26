#pragma once

// Internal helpers shared by the method implementations: OpenMP glue and a reusable Dijkstra.
// Loops stay OpenMP 2.0 compatible (signed int indices, no atomic read/write, no min/max reductions) for MSVC.

#include <exception>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cppbone/cppbone.hpp"

namespace cppbone::detail {

inline int thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Exceptions must not leave an OpenMP region: run() records the first one, rethrow() raises it afterwards.
class Errors {
public:
    template <typename F>
    void run(F&& f) {
        try {
            f();
        } catch (...) {
#pragma omp critical(cppbone_errors)
            if (!first_) first_ = std::current_exception();
        }
    }
    void rethrow() const {
        if (first_) std::rethrow_exception(first_);
    }

private:
    std::exception_ptr first_;
};

// Per-thread scratch for repeated single-source searches; only the touched entries are reset between sources.
struct DijkstraWorkspace {
    std::vector<double> dist, seen;
    std::vector<char> done;
    std::vector<int> parent_edge;  // edge to the shortest-path-tree parent, -1 for the source and unreached nodes
    std::vector<int> settled;      // nodes in the order they were settled
    std::vector<int> touched;
    std::vector<std::tuple<double, long, int>> heap;

    explicit DijkstraWorkspace(int n) : dist(n, INF), seen(n, INF), done(n, 0), parent_edge(n, -1) {}
};

// single_source_dijkstra_path_length on a workspace, with the same tie-breaking and the parent as an edge id.
// Stops once the next node to settle is at distance >= cutoff: nodes not settled keep dist = INF.
// Edges with (*removed)[id] set are ignored.
void dijkstra(const Graph& g, int source, const std::vector<double>& weight, Disjunction disjunction,
              DijkstraWorkspace& ws, double cutoff = INF, const std::vector<char>* removed = nullptr);

}  // namespace cppbone::detail
