/// @file
/// @brief Doubly Stochastic filter (structural/doubly_stochastic.py).

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>

#include <boost/pending/disjoint_sets.hpp>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

Backbone doubly_stochastic(const Graph &data) {
    // The Python version works on nx.to_pandas_edgelist rows, so edge orientation follows
    // Graph::edges().
    auto rows      = data.edges();
    auto label_cmp = [](const std::string &a, const std::string &b) { return label_less(a, b); };
    std::map<std::string, int, decltype(label_cmp)> src(label_cmp), dst(label_cmp);
    for (const auto &e : rows) {
        src[data.label(e.u)], dst[data.label(e.v)];
    }
    int i0 = 0;
    for (auto &[_, v] : src) {
        v = i0++;
    }
    i0 = 0;
    for (auto &[_, v] : dst) {
        v = i0++;
    }
    std::vector<std::string> src_labels, dst_labels;
    for (auto &[k, _] : src) {
        src_labels.push_back(k);
    }
    for (auto &[k, _] : dst) {
        dst_labels.push_back(k);
    }

    // pivot_table(values="weight", index="source", columns="target", aggfunc="sum", fill_value=0) +
    // .0001
    size_t R = src.size(), C = dst.size();
    std::vector table(R, std::vector<double>(C, 0));
    for (const auto &[u, v, id] : rows) {
        table[src[data.label(u)]][dst[data.label(v)]] += data.get(id, "weight");
    }
    int rows_n = static_cast<int>(R), cols_n = static_cast<int>(C);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_n; ++r) {
        for (double &x : table[r]) {
            x += .0001;
        }
    }

    // Sinkhorn-Knopp
    auto row_sums = [&] {
        std::vector<double> s(R, 0);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows_n; ++r)
            s[r] = std::accumulate(table[r].begin(), table[r].end(), 0.0);
        return s;
    };
    auto stddev = [](const std::vector<double> &v) {
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double var        = 0;
        for (const double x : v) {
            var += (x - mean) * (x - mean);
        }
        return std::sqrt(var / v.size());
    };
    auto rs      = row_sums();
    int attempts = 0;
    while (stddev(rs) > 1e-12) {
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows_n; ++r) {
            for (double &x : table[r]) {
                x /= rs[r];
            }
        }
        // Column sums in blocks of columns: each column still adds its rows in order, so the sums
        // are exact.
        std::vector<double> cs(C, 0);
        constexpr int block = 512;
        int blocks          = (cols_n + block - 1) / block;
#pragma omp parallel for schedule(dynamic)
        for (int bi = 0; bi < blocks; ++bi) {
            int c0 = bi * block, c1 = std::min(cols_n, c0 + block);
            for (int r = 0; r < rows_n; ++r) {
                for (int c = c0; c < c1; ++c) {
                    cs[c] += table[r][c];
                }
            }
        }
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows_n; ++r) {
            for (int c = 0; c < cols_n; ++c) {
                table[r][c] /= cs[c];
            }
        }
        rs = row_sums();
        if (++attempts > 1000) {
            std::cerr << "RuntimeWarning: Matrix could not be reduced to doubly stochastic. See "
                         "Sec. 3 of Sinkhorn 1964\n";
            return {Graph{},
                    "Doubly Stochastic Filter",
                    "score",
                    false,
                    {boolean_filter, threshold_filter, fraction_filter},
                    FilterOn::Edges};
        }
    }

    // melt (column-major), keep source < target and value > 0, sort by value desc, inner-merge with
    // the edge list.
    std::map<std::pair<std::string, std::string>, double> weight_of;
    for (const auto &[u, v, id] : rows) {
        weight_of.emplace(std::make_pair(data.label(u), data.label(v)), data.get(id, "weight"));
    }
    struct Row {
        std::string source, target;
        double value, weight;
        bool in_backbone = false;
    };
    std::vector<Row> melted;
    for (size_t c = 0; c < C; ++c) {
        for (size_t r = 0; r < R; ++r) {
            if (label_less(src_labels[r], dst_labels[c]) && table[r][c] > 0) {
                melted.push_back({src_labels[r], dst_labels[c], table[r][c], 0});
            }
        }
    }
    std::stable_sort(melted.begin(), melted.end(),
                     [](const Row &a, const Row &b) { return a.value > b.value; });
    std::vector<Row> merged;
    for (auto &r : melted) {
        if (auto it = weight_of.find({r.source, r.target}); it != weight_of.end()) {
            r.weight = it->second, merged.push_back(r);
        }
    }

    // Add edges by decreasing score until the graph is connected and spans every node of the merged
    // table.
    std::map<std::string, int> ids;
    for (const auto &r : merged) {
        ids.emplace(r.source, static_cast<int>(ids.size())),
            ids.emplace(r.target, static_cast<int>(ids.size()));
    }
    int doubly_nodes = static_cast<int>(ids.size());
    std::vector<int> rank(doubly_nodes), parent(doubly_nodes);
    boost::disjoint_sets sets(rank.data(), parent.data());
    for (int x = 0; x < doubly_nodes; ++x) {
        sets.make_set(x);
    }
    std::vector present(doubly_nodes, false);
    int nodes = 0, components = 0;
    for (size_t i = 0; components != 1 || nodes < doubly_nodes; ++i) {
        if (i == merged.size()) {
            break;
        }
        int a = ids[merged[i].source], b = ids[merged[i].target];
        for (int x : {a, b}) {
            if (!present[x]) {
                present[x] = true, ++nodes, ++components;
            }
        }
        int rb = sets.find_set(b);
        if (int ra = sets.find_set(a); ra != rb) {
            sets.link(ra, rb), --components;
        }
        merged[i].in_backbone = true;
    }

    Graph g;
    for (const auto &[source, target, value, weight, in_backbone] : merged) {
        if (source != target) {
            g.add_edge(source, target,
                       {{"weight", weight}, {"score", value}, {"in_backbone", in_backbone}});
        }
    }
    return {std::move(g),
            "Doubly Stochastic Filter",
            "score",
            false,
            {boolean_filter, threshold_filter, fraction_filter},
            FilterOn::Edges};
}

} // namespace cppbone
