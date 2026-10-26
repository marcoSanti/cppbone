/// @file
/// @brief Compare implementation (compare.py) and the two-sample KS statistic.

#include "cppbone/compare.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <ranges>
#include <stdexcept>

#include "cppbone/cppbone.hpp"

namespace cppbone {

void Table::to_csv(std::ostream &out) const {
    out << index_name;
    for (const auto &c : columns) {
        out << ',' << c;
    }
    out << '\n';
    for (const auto &[label, row] : std::views::zip(index, data)) {
        out << label;
        for (const double v : row) {
            out << ',' << format_double(v);
        }
        out << '\n';
    }
}

void Compare::set_filter(Filter filter, std::vector<double> values) {
    filter_        = filter;
    filter_values_ = std::move(values);
    if (filter == fraction_filter) {
        value_name_ = "Fraction of Edges";
    } else if (filter == threshold_filter) {
        value_name_ = "P-value";
    }
}

void Compare::need_values() {
    if (filter_ == boolean_filter) {
        filter_values_.assign(backbones_.size(), 0);
    }
    if (filter_values_.empty()) {
        throw std::invalid_argument("Please enter the filter values.");
    }
}

Graph Compare::extract(const size_t i, const double value) const {
    return filter_(backbones_[i], value, false, "weight", false);
}

Table Compare::properties() {
    need_values();
    const auto measure = [&](const Graph &b) {
        return props_ | std::views::values |
               std::views::transform([&](const Property &p) { return p(network_, b); }) |
               std::ranges::to<std::vector>();
    };
    Table t;
    t.index.emplace_back("Original");
    t.index.append_range(backbones_ | std::views::transform(&Backbone::method_name));
    t.columns = props_ | std::views::keys | std::ranges::to<std::vector>();
    t.data.push_back(measure(network_));
    for (size_t i = 0; i < backbones_.size(); ++i) {
        t.data.push_back(measure(extract(i, filter_values_[i])));
    }
    return t;
}

std::vector<std::pair<std::string, Table>> Compare::properties_progression() {
    if (filter_ == boolean_filter) {
        throw std::invalid_argument("Can't apply the boolean filter in this function.");
    }
    if (filter_values_.empty()) {
        throw std::invalid_argument("Please enter the filter values.");
    }
    std::vector<std::pair<std::string, Table>> res;
    const auto methods =
        backbones_ | std::views::transform(&Backbone::method_name) | std::ranges::to<std::vector>();
    for (const auto &name : props_ | std::views::keys) {
        res.emplace_back(
            name, Table{.index_name = value_name_, .index = {}, .columns = methods, .data = {}});
    }
    for (const double value : filter_values_) {
        for (auto &table : res | std::views::values) {
            table.index.push_back(format_double(value));
            table.data.emplace_back();
        }
        for (size_t i = 0; i < backbones_.size(); ++i) {
            const Graph b = extract(i, value);
            for (auto &&[table, prop] :
                 std::views::zip(res | std::views::values, props_ | std::views::values)) {
                table.data.back().push_back(prop(network_, b));
            }
        }
    }
    return res;
}

/// @brief Intersects two graphs by label.
/// @param acc Graph to restrict.
/// @param g Graph to intersect with.
/// @return The nodes and edges of acc that are also in g, with acc's attributes.
static Graph intersect(const Graph &acc, const Graph &g) {
    Graph out =
        acc.subgraph(acc.nodes() |
                     std::views::transform([&](const int u) { return g.node(acc.label(u)) >= 0; }) |
                     std::ranges::to<std::vector<bool>>());
    for (const auto &[u, v, id] : out.edges()) {
        const int b = g.node(out.label(v));
        if (const int a = g.node(out.label(u)); !g.has_edge(a, b)) {
            out.remove_edge(id);
        }
    }
    return out;
}

/// @brief Removes the nodes without edges.
/// @param g Graph.
/// @return The subgraph on the nodes of degree > 0.
static Graph drop_isolates(const Graph &g) {
    return g.subgraph(g.nodes() |
                      std::views::transform([&](const int u) { return g.degree(u) > 0; }) |
                      std::ranges::to<std::vector<bool>>());
}

Compare::KsResult Compare::distribution_ks_statistic(bool increasing, std::vector<bool> consent) {
    need_values();
    if (consent.empty()) {
        consent.assign(backbones_.size(), false);
    }
    const bool any = std::ranges::contains(consent, true);

    std::vector<Graph> extracted;
    std::optional<Graph> consensual;
    for (size_t i = 0; i < backbones_.size(); ++i) {
        extracted.push_back(extract(i, filter_values_[i]));
        if (consent[i]) {
            consensual = consensual ? intersect(*consensual, extracted.back()) : extracted.back();
        }
    }
    if (consensual) {
        consensual = drop_isolates(*consensual);
    }

    KsResult res;
    res.ks_statistics.index_name = "";
    res.ks_statistics.index =
        backbones_ | std::views::transform(&Backbone::method_name) | std::ranges::to<std::vector>();
    if (any) {
        res.ks_statistics.index.emplace_back("Consensual Backbone");
    }
    res.ks_statistics.data.assign(res.ks_statistics.index.size(), {});
    for (const auto &[name, prop] : dists_) {
        res.ks_statistics.columns.push_back(name);
        const auto values0 = prop(network_);
        auto &dist         = res.dist[name];
        dist["Original"]   = cumulative_dist(values0, increasing);
        for (size_t i = 0; i < backbones_.size(); ++i) {
            const auto values1              = prop(extracted[i]);
            dist[backbones_[i].method_name] = cumulative_dist(values1, increasing);
            res.ks_statistics.data[i].push_back(ks_statistic(values0, values1));
        }
        if (any) {
            const auto values1          = prop(*consensual);
            dist["Consensual Backbone"] = cumulative_dist(values1, increasing);
            res.ks_statistics.data.back().push_back(ks_statistic(values0, values1));
        }
    }
    if (any) {
        res.consensual_backbone = consensual;
    }
    return res;
}

Graph Compare::consent() {
    need_values();
    Graph acc;
    for (size_t i = 0; i < backbones_.size(); ++i) {
        Graph b = extract(i, filter_values_[i]);
        acc     = i == 0 ? b : intersect(acc, b);
    }
    return drop_isolates(acc);
}

double ks_statistic(std::vector<double> a, std::vector<double> b) {
    if (a.empty() || b.empty()) {
        return NaN;
    }
    std::ranges::sort(a);
    std::ranges::sort(b);
    double d = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const double x = std::min(a[i], b[j]);
        while (i < a.size() && a[i] == x) {
            ++i;
        }
        while (j < b.size() && b[j] == x) {
            ++j;
        }
        d = std::max(
            d, std::abs(static_cast<double>(i) / a.size() - static_cast<double>(j) / b.size()));
    }
    return d;
}

} // namespace cppbone
