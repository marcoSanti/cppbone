#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>

#include "netbone/netbone.hpp"

namespace netbone {

static int find(std::vector<int>& p, int x) {
    while (p[x] != x) x = p[x] = p[p[x]];
    return x;
}

Backbone doubly_stochastic(const Graph& data) {
    // The Python version works on nx.to_pandas_edgelist rows, so edge orientation follows Graph::edges().
    auto rows = data.edges();
    auto label_cmp = [](const std::string& a, const std::string& b) { return label_less(a, b); };
    std::map<std::string, int, decltype(label_cmp)> src(label_cmp), dst(label_cmp);
    for (const auto& e : rows) src[data.label(e.u)], dst[data.label(e.v)];
    int i0 = 0;
    for (auto& kv : src) kv.second = i0++;
    i0 = 0;
    for (auto& kv : dst) kv.second = i0++;
    std::vector<std::string> src_labels, dst_labels;
    for (auto& kv : src) src_labels.push_back(kv.first);
    for (auto& kv : dst) dst_labels.push_back(kv.first);

    // pivot_table(values="weight", index="source", columns="target", aggfunc="sum", fill_value=0) + .0001
    size_t R = src.size(), C = dst.size();
    std::vector<std::vector<double>> table(R, std::vector<double>(C, 0));
    for (const auto& e : rows) table[src[data.label(e.u)]][dst[data.label(e.v)]] += data.get(e.id, "weight");
    for (auto& r : table)
        for (double& x : r) x += .0001;

    // Sinkhorn-Knopp
    auto row_sums = [&] {
        std::vector<double> s(R, 0);
        for (size_t r = 0; r < R; ++r) s[r] = std::accumulate(table[r].begin(), table[r].end(), 0.0);
        return s;
    };
    auto stddev = [](const std::vector<double>& v) {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size(), var = 0;
        for (double x : v) var += (x - mean) * (x - mean);
        return std::sqrt(var / v.size());
    };
    auto rs = row_sums();
    int attempts = 0;
    while (stddev(rs) > 1e-12) {
        for (size_t r = 0; r < R; ++r)
            for (double& x : table[r]) x /= rs[r];
        std::vector<double> cs(C, 0);
        for (size_t r = 0; r < R; ++r)
            for (size_t c = 0; c < C; ++c) cs[c] += table[r][c];
        for (size_t r = 0; r < R; ++r)
            for (size_t c = 0; c < C; ++c) table[r][c] /= cs[c];
        rs = row_sums();
        if (++attempts > 1000) {
            std::cerr << "RuntimeWarning: Matrix could not be reduced to doubly stochastic. See Sec. 3 of Sinkhorn 1964\n";
            return {Graph{}, "Doubly Stochastic Filter", "score", false,
                    {boolean_filter, threshold_filter, fraction_filter}, FilterOn::Edges};
        }
    }

    // melt (column-major), keep source < target and value > 0, sort by value desc, inner-merge with the edge list.
    std::map<std::pair<std::string, std::string>, double> weight_of;
    for (const auto& e : rows) weight_of.emplace(std::make_pair(data.label(e.u), data.label(e.v)), data.get(e.id, "weight"));
    struct Row {
        std::string source, target;
        double value, weight;
        bool in_backbone = false;
    };
    std::vector<Row> melted;
    for (size_t c = 0; c < C; ++c)
        for (size_t r = 0; r < R; ++r)
            if (label_less(src_labels[r], dst_labels[c]) && table[r][c] > 0)
                melted.push_back({src_labels[r], dst_labels[c], table[r][c], 0});
    std::stable_sort(melted.begin(), melted.end(), [](const Row& a, const Row& b) { return a.value > b.value; });
    std::vector<Row> merged;
    for (auto& r : melted) {
        auto it = weight_of.find({r.source, r.target});
        if (it != weight_of.end()) r.weight = it->second, merged.push_back(r);
    }

    // Add edges by decreasing score until the graph is connected and spans every node of the merged table.
    std::map<std::string, int> ids;
    for (const auto& r : merged) ids.emplace(r.source, (int)ids.size()), ids.emplace(r.target, (int)ids.size());
    int doubly_nodes = (int)ids.size();
    std::vector<int> parent(doubly_nodes);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<bool> present(doubly_nodes, false);
    int nodes = 0, components = 0;
    for (size_t i = 0; components != 1 || nodes < doubly_nodes; ++i) {
        if (i == merged.size()) break;
        int a = ids[merged[i].source], b = ids[merged[i].target];
        for (int x : {a, b})
            if (!present[x]) present[x] = true, ++nodes, ++components;
        int ra = find(parent, a), rb = find(parent, b);
        if (ra != rb) parent[ra] = rb, --components;
        merged[i].in_backbone = true;
    }

    Graph g;
    for (const auto& r : merged)
        if (r.source != r.target)
            g.add_edge(r.source, r.target, {{"weight", r.weight}, {"score", r.value}, {"in_backbone", r.in_backbone}});
    return {g, "Doubly Stochastic Filter", "score", false, {boolean_filter, threshold_filter, fraction_filter},
            FilterOn::Edges};
}

}  // namespace netbone
