/// @file
/// @brief Multiple Linkage Analysis (structural/mlam.py).

#include <algorithm>
#include <cmath>
#include <functional>
#include <ranges>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

/// @brief A node's edges sorted by their share of its strength, largest first (stable).
/// @param g Graph with a "weight" edge attribute.
/// @param node Node index.
/// @return (edge id, weight as a percentage of the node's strength) pairs.
static std::vector<std::pair<int, double>> get_neighbor_weights(const Graph &g, int node) {
    double total = 0;
    for (const auto [v, id] : g.adj(node)) {
        total += g.get(id, "weight");
    }
    auto out = g.adj(node) | std::views::transform([&](const Adjacent &a) {
                   return std::pair{a.edge, g.get(a.edge, "weight") / total * 100};
               }) |
               std::ranges::to<std::vector>();
    std::ranges::stable_sort(out, std::greater<>(), [](const auto &p) { return p.second; });
    return out;
}

/// @brief Ideal distribution of a node with i + 1 equal links.
/// @param i Index of the last kept link.
/// @param total Length of the distribution (the node's degree).
/// @return 100 / (i + 1) for the first i + 1 entries, 0 after.
static std::vector<double> get_ideal_distribution(int i, int total) {
    std::vector<double> a(total, 0);
    std::ranges::fill_n(a.begin(), i + 1, 100.0 / (i + 1));
    return a;
}

/// @brief Coefficient of determination (squared Pearson correlation) of two series, rounded to 2
/// decimals.
/// @param f First series.
/// @param y Second series, same length.
/// @return r^2 rounded to 2 decimals; NaN when either series is constant.
static double compute_cod(const std::vector<double> &f, const std::vector<double> &y) {
    const double n = f.size();
    double mf = 0, my = 0;
    for (size_t i = 0; i < f.size(); ++i) {
        mf += f[i] / n, my += y[i] / n;
    }
    double sfy = 0, sff = 0, syy = 0;
    for (size_t i = 0; i < f.size(); ++i) {
        sfy += (f[i] - mf) * (y[i] - my);
        sff += (f[i] - mf) * (f[i] - mf);
        syy += (y[i] - my) * (y[i] - my);
    }
    const double corr = sfy / std::sqrt(sff * syy);
    return std::round(corr * corr * 100) / 100;
}

Backbone mlam(const Graph &data) {
    Graph g     = data;
    const int n = g.number_of_nodes();
    // Nodes run in parallel; each records the edges it keeps, applied afterwards.
    std::vector<std::vector<int>> kept(n);
#pragma omp parallel for schedule(dynamic, 64)
    for (int node = 0; node < n; ++node) {
        const auto neighbors_weights = get_neighbor_weights(g, node);
        const auto real_distribution =
            neighbors_weights | std::views::values | std::ranges::to<std::vector>();
        const int neighbors_count = static_cast<int>(neighbors_weights.size());
        int edge_index            = 0;
        if (neighbors_count != 1) {
            double old_cod = 0;
            for (int i = 0; i < neighbors_count; ++i) {
                const double new_cod =
                    compute_cod(real_distribution, get_ideal_distribution(i, neighbors_count));
                if (std::isnan(new_cod)) {
                    break;
                }
                if (old_cod <= new_cod) {
                    old_cod    = new_cod;
                    edge_index = i;
                } else {
                    break;
                }
                if (i == neighbors_count - 1) {
                    edge_index = i;
                }
            }
        }
        kept[node] = neighbors_weights | std::views::keys | std::views::take(edge_index + 1) |
                     std::ranges::to<std::vector>();
    }
    std::vector<char> keep(g.edge_capacity(), 0);
    for (const int id : kept | std::views::join) {
        keep[id] = 1;
    }
    const auto edges = g.edges();
    const int m      = static_cast<int>(edges.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        g.set(edges[i].id, "in_backbone", keep[edges[i].id] != 0);
    }
    return {.graph         = std::move(g),
            .method_name   = "Multiple Linkage Analysis",
            .property_name = "weight",
            .filters       = {boolean_filter}};
}

} // namespace cppbone
