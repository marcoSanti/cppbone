#pragma once

/// @file
/// @brief Compare: side-by-side properties, progressions and KS statistics of several backbones
/// (compare.py).

#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cppbone/backbone.hpp"
#include "cppbone/graph.hpp"

namespace cppbone {

/// @brief Labelled table standing in for the pandas DataFrames Compare returns: data[row][column].
struct Table {
    std::string index_name;                ///< Name of the row index column.
    std::vector<std::string> index;        ///< Row labels.
    std::vector<std::string> columns;      ///< Column labels.
    std::vector<std::vector<double>> data; ///< Values, one vector per row.

    /// @brief Writes the table as CSV: a header row, then one row per index label.
    /// @param out Destination stream.
    void to_csv(std::ostream &out) const;
};

/// @brief A property of a backbone measured against the original network, e.g. edge_fraction.
using Property       = std::function<double(const Graph &original, const Graph &graph)>;
/// @brief A per-node or per-edge sample of a graph, e.g. its degrees.
///
/// Python passes these through add_property too; C++ needs a separate slot for the one-argument
/// form.
using Distribution   = std::function<std::vector<double>(const Graph &graph)>;
/// @brief Points (x, y) of a cumulative distribution, as returned by cumulative_dist.
using CumulativeDist = std::vector<std::pair<double, double>>;

/// @brief Compares several backbones of one network (the Python Compare class).
///
/// Set the network, add backbones, properties or distributions, choose a filter and its values,
/// then ask for the tables.
class Compare {
public:
    /// @brief Sets the original network the backbones are compared with.
    /// @param network Original network.
    void set_network(const Graph &network) { network_ = network; }

    /// @brief Adds a backbone to compare.
    /// @param backbone Backbone (copied).
    void add_backbone(const Backbone &backbone) { backbones_.push_back(backbone); }

    /// @brief Adds a property, which becomes a column of properties() and a table of
    /// properties_progression().
    /// @param name Column name.
    /// @param property Function of (original network, extracted backbone).
    void add_property(const std::string &name, Property property) {
        props_.emplace_back(name, std::move(property));
    }

    /// @brief Adds a distribution for distribution_ks_statistic().
    /// @param name Column name.
    /// @param d Function returning the sample of a graph.
    void add_distribution(const std::string &name, Distribution d) {
        dists_.emplace_back(name, std::move(d));
    }

    /// @brief Chooses the filter used to extract the backbones, and its values.
    /// @param filter Filter applied to every backbone.
    /// @param values One value per backbone for properties() and the KS statistics, or the sequence
    /// of values for
    ///        properties_progression(). Not needed for boolean_filter.
    void set_filter(Filter filter, std::vector<double> values = {});

    /// @brief Measures every property on the original network and on every extracted backbone.
    /// @return Rows "Original" and one per backbone (by method name), one column per property.
    /// @throws std::invalid_argument if the filter values are missing.
    Table properties();

    /// @brief Measures every property for each filter value.
    /// @return One table per property: one row per filter value, one column per backbone.
    /// @throws std::invalid_argument with boolean_filter or without filter values.
    std::vector<std::pair<std::string, Table>> properties_progression();

    /// @brief Result of distribution_ks_statistic().
    struct KsResult {
        Table ks_statistics; ///< KS statistic per backbone (rows) and distribution (columns).
        /// @brief Cumulative distributions by distribution name, then by "Original" or method name.
        std::map<std::string, std::map<std::string, CumulativeDist>> dist;
        std::optional<Graph>
            consensual_backbone; ///< Intersection of the consented backbones, if any.
    };

    /// @brief Compares the distributions of every backbone with the original network's.
    /// @param increasing Order of the returned cumulative distributions (see cumulative_dist).
    /// @param consent One flag per backbone: the flagged ones are intersected into a consensual
    /// backbone, which
    ///        gets its own row. Empty means none.
    /// @return The KS statistics, the cumulative distributions and the consensual backbone.
    /// @throws std::invalid_argument if the filter values are missing.
    KsResult distribution_ks_statistic(bool increasing = true, std::vector<bool> consent = {});

    /// @brief Intersection of all extracted backbones: edges present in every one, isolated nodes
    /// dropped.
    /// @return The consensual backbone.
    /// @throws std::invalid_argument if the filter values are missing.
    Graph consent();

private:
    /// @brief Applies the chosen filter to one backbone, silently.
    /// @param i Backbone index.
    /// @param value Filter value.
    /// @return The extracted graph.
    Graph extract(size_t i, double value) const;

    /// @brief Fills in the values for boolean_filter and checks that values exist for the other
    /// filters.
    /// @throws std::invalid_argument if they are missing.
    void need_values();

    Graph network_;                                           ///< Original network.
    std::vector<Backbone> backbones_;                         ///< Backbones compared.
    std::vector<std::pair<std::string, Property>> props_;     ///< Properties, by name.
    std::vector<std::pair<std::string, Distribution>> dists_; ///< Distributions, by name.
    std::string value_name_ = "p_value";        ///< Index name of the progression tables.
    Filter filter_          = threshold_filter; ///< Filter used to extract the backbones.
    std::vector<double> filter_values_;         ///< Filter values (see set_filter).
};

/// @brief Two-sample Kolmogorov-Smirnov statistic (scipy.stats.kstest(a, b)).
/// @param a First sample.
/// @param b Second sample.
/// @return Largest distance between the two empirical CDFs; NaN if a sample is empty.
double ks_statistic(std::vector<double> a, std::vector<double> b);

} // namespace cppbone
