#include "netbone/compare.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdexcept>

#include "netbone/netbone.hpp"

namespace netbone {

void Table::to_csv(std::ostream& out) const {
    out << index_name;
    for (const auto& c : columns) out << ',' << c;
    out << '\n';
    for (size_t r = 0; r < index.size(); ++r) {
        out << index[r];
        for (double v : data[r]) out << ',' << format_double(v);
        out << '\n';
    }
}

void Compare::set_filter(Filter filter, std::vector<double> values) {
    filter_ = filter;
    filter_values_ = std::move(values);
    if (filter == fraction_filter)
        value_name_ = "Fraction of Edges";
    else if (filter == threshold_filter)
        value_name_ = "P-value";
}

void Compare::need_values() {
    if (filter_ == boolean_filter) filter_values_.assign(backbones_.size(), 0);
    if (filter_values_.empty()) throw std::invalid_argument("Please enter the filter values.");
}

Graph Compare::extract(size_t i, double value) const { return filter_(backbones_[i], value, false, "weight", false); }

Table Compare::properties() {
    need_values();
    Table t;
    t.index.push_back("Original");
    for (const auto& b : backbones_) t.index.push_back(b.method_name);
    for (const auto& p : props_) t.columns.push_back(p.first);
    t.data.assign(t.index.size(), std::vector<double>(props_.size()));
    for (size_t j = 0; j < props_.size(); ++j) t.data[0][j] = props_[j].second(network_, network_);
    for (size_t i = 0; i < backbones_.size(); ++i) {
        Graph b = extract(i, filter_values_[i]);
        for (size_t j = 0; j < props_.size(); ++j) t.data[i + 1][j] = props_[j].second(network_, b);
    }
    return t;
}

std::vector<std::pair<std::string, Table>> Compare::properties_progression() {
    if (filter_ == boolean_filter) throw std::invalid_argument("Cann't apply the boolean filter in this function.");
    if (filter_values_.empty()) throw std::invalid_argument("Please enter the filter values.");
    std::vector<std::pair<std::string, Table>> res;
    for (const auto& p : props_) {
        Table t;
        t.index_name = value_name_;
        for (const auto& b : backbones_) t.columns.push_back(b.method_name);
        res.emplace_back(p.first, t);
    }
    for (double value : filter_values_) {
        for (auto& r : res) {
            r.second.index.push_back(format_double(value));
            r.second.data.emplace_back();
        }
        for (size_t i = 0; i < backbones_.size(); ++i) {
            Graph b = extract(i, value);
            for (size_t j = 0; j < props_.size(); ++j) res[j].second.data.back().push_back(props_[j].second(network_, b));
        }
    }
    return res;
}

// Keep the nodes and edges of `acc` that are also in `g`.
static Graph intersect(const Graph& acc, const Graph& g) {
    std::vector<bool> keep(acc.number_of_nodes());
    for (int u = 0; u < acc.number_of_nodes(); ++u) keep[u] = g.node(acc.label(u)) >= 0;
    Graph out = acc.subgraph(keep);
    for (const auto& e : out.edges()) {
        int a = g.node(out.label(e.u)), b = g.node(out.label(e.v));
        if (!g.has_edge(a, b)) out.remove_edge(e.id);
    }
    return out;
}

static Graph drop_isolates(const Graph& g) {
    std::vector<bool> keep(g.number_of_nodes());
    for (int u = 0; u < g.number_of_nodes(); ++u) keep[u] = g.degree(u) > 0;
    return g.subgraph(keep);
}

Compare::KsResult Compare::distribution_ks_statistic(bool increasing, std::vector<bool> consent) {
    need_values();
    if (consent.empty()) consent.assign(backbones_.size(), false);
    bool any = std::find(consent.begin(), consent.end(), true) != consent.end();

    std::vector<Graph> extracted;
    std::optional<Graph> consensual;
    for (size_t i = 0; i < backbones_.size(); ++i) {
        extracted.push_back(extract(i, filter_values_[i]));
        if (consent[i]) consensual = consensual ? intersect(*consensual, extracted.back()) : extracted.back();
    }
    if (consensual) consensual = drop_isolates(*consensual);

    KsResult res;
    res.ks_statistics.index_name = "";
    for (const auto& b : backbones_) res.ks_statistics.index.push_back(b.method_name);
    if (any) res.ks_statistics.index.push_back("Consensual Backbone");
    res.ks_statistics.data.assign(res.ks_statistics.index.size(), {});
    for (const auto& [name, prop] : dists_) {
        res.ks_statistics.columns.push_back(name);
        auto values0 = prop(network_);
        auto& dist = res.dist[name];
        dist["Original"] = cumulative_dist(values0, increasing);
        for (size_t i = 0; i < backbones_.size(); ++i) {
            auto values1 = prop(extracted[i]);
            dist[backbones_[i].method_name] = cumulative_dist(values1, increasing);
            res.ks_statistics.data[i].push_back(ks_statistic(values0, values1));
        }
        if (any) {
            auto values1 = prop(*consensual);
            dist["Consensual Backbone"] = cumulative_dist(values1, increasing);
            res.ks_statistics.data.back().push_back(ks_statistic(values0, values1));
        }
    }
    if (any) res.consensual_backbone = consensual;
    return res;
}

Graph Compare::consent() {
    need_values();
    Graph acc;
    for (size_t i = 0; i < backbones_.size(); ++i) {
        Graph b = extract(i, filter_values_[i]);
        acc = i == 0 ? b : intersect(acc, b);
    }
    return drop_isolates(acc);
}

double ks_statistic(std::vector<double> a, std::vector<double> b) {
    if (a.empty() || b.empty()) return NaN;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    double d = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        double x = std::min(a[i], b[j]);
        while (i < a.size() && a[i] == x) ++i;
        while (j < b.size() && b[j] == x) ++j;
        d = std::max(d, std::abs(double(i) / a.size() - double(j) / b.size()));
    }
    return d;
}

}  // namespace netbone
