#include <algorithm>
#include <cmath>

#include "netbone/netbone.hpp"

namespace netbone {

// Neighbours (edge ids) sorted by normalised weight (percent of strength), descending and stable.
static std::vector<std::pair<int, double>> get_neighbor_weights(const Graph& g, int node) {
    std::vector<std::pair<int, double>> out;
    double total = 0;
    for (auto [v, id] : g.adj(node)) total += g.get(id, "weight");
    for (auto [v, id] : g.adj(node)) out.emplace_back(id, g.get(id, "weight") / total * 100);
    std::stable_sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.second > b.second; });
    return out;
}

static std::vector<double> get_ideal_distribution(int i, int total) {
    std::vector<double> a(total, 0);
    for (int j = 0; j <= i; ++j) a[j] = 100.0 / (i + 1);
    return a;
}

// Coefficient of determination (squared Pearson correlation) rounded to 2 decimals; NaN for a constant input.
static double compute_cod(const std::vector<double>& f, const std::vector<double>& y) {
    double n = f.size(), mf = 0, my = 0;
    for (size_t i = 0; i < f.size(); ++i) mf += f[i] / n, my += y[i] / n;
    double sfy = 0, sff = 0, syy = 0;
    for (size_t i = 0; i < f.size(); ++i) {
        sfy += (f[i] - mf) * (y[i] - my);
        sff += (f[i] - mf) * (f[i] - mf);
        syy += (y[i] - my) * (y[i] - my);
    }
    double corr = sfy / std::sqrt(sff * syy);
    return std::round(corr * corr * 100) / 100;
}

Backbone mlam(const Graph& data) {
    Graph g = data;
    for (const auto& e : g.edges()) g.set(e.id, "in_backbone", false);
    for (int node = 0; node < g.number_of_nodes(); ++node) {
        auto neighbors_weights = get_neighbor_weights(g, node);
        std::vector<double> real_distribution;
        for (auto& p : neighbors_weights) real_distribution.push_back(p.second);
        int neighbors_count = (int)neighbors_weights.size();
        int edge_index = 0;
        double old_cod = 0;
        if (neighbors_count != 1) {
            for (int i = 0; i < neighbors_count; ++i) {
                double new_cod = compute_cod(real_distribution, get_ideal_distribution(i, neighbors_count));
                if (std::isnan(new_cod)) break;
                if (old_cod <= new_cod) {
                    old_cod = new_cod;
                    edge_index = i;
                } else {
                    break;
                }
                if (i == neighbors_count - 1) edge_index = i;
            }
        }
        for (int j = 0; j < neighbors_count && j <= edge_index; ++j) g.set(neighbors_weights[j].first, "in_backbone", true);
    }
    return {g, "Multiple Linkage Analysis", "weight", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace netbone
