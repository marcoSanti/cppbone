#pragma once

#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "netbone/backbone.hpp"
#include "netbone/graph.hpp"

namespace netbone {

// Stand-in for the pandas DataFrames Compare returns: data[row][column].
struct Table {
    std::string index_name;
    std::vector<std::string> index;
    std::vector<std::string> columns;
    std::vector<std::vector<double>> data;

    void to_csv(std::ostream& out) const;
};

using Property = std::function<double(const Graph& original, const Graph& graph)>;
// Python passes these through add_property too; C++ needs a separate slot for the one-argument form.
using Distribution = std::function<std::vector<double>(const Graph& graph)>;
using CumulativeDist = std::vector<std::pair<double, double>>;

class Compare {
public:
    void set_network(const Graph& network) { network_ = network; }
    void add_backbone(const Backbone& backbone) { backbones_.push_back(backbone); }
    void add_property(const std::string& name, Property property) { props_.emplace_back(name, std::move(property)); }
    void add_distribution(const std::string& name, Distribution d) { dists_.emplace_back(name, std::move(d)); }
    void set_filter(Filter filter, std::vector<double> values = {});

    Table properties();
    std::vector<std::pair<std::string, Table>> properties_progression();

    struct KsResult {
        Table ks_statistics;
        std::map<std::string, std::map<std::string, CumulativeDist>> dist;  // property -> method -> cdf
        std::optional<Graph> consensual_backbone;
    };
    KsResult distribution_ks_statistic(bool increasing = true, std::vector<bool> consent = {});
    Graph consent();

private:
    Graph extract(size_t i, double value) const;
    void need_values();

    Graph network_;
    std::vector<Backbone> backbones_;
    std::vector<std::pair<std::string, Property>> props_;
    std::vector<std::pair<std::string, Distribution>> dists_;
    std::string value_name_ = "p_value";
    Filter filter_ = threshold_filter;
    std::vector<double> filter_values_;
};

// scipy.stats.kstest(a, b) statistic (two-sample).
double ks_statistic(std::vector<double> a, std::vector<double> b);

}  // namespace netbone
