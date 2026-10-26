#pragma once

/// @file
/// @brief The Backbone result type and the three filters that turn it into a graph (filters.py).

#include <iosfwd>
#include <string>
#include <vector>

#include "cppbone/graph.hpp"

namespace cppbone {

/// @brief Whether a backbone scores edges or nodes.
enum class FilterOn {
    Edges,  ///< The property is an edge attribute; filters select edges.
    Nodes   ///< The property is a node attribute; filters select nodes and keep the induced subgraph.
};

struct Backbone;

/// @brief Signature shared by every filter, so filters can be compared and stored like the Python function objects.
using Filter = Graph (*)(const Backbone& backbone, double value, bool narrate, const std::string& secondary_property,
                         bool secondary_property_ascending);

/// @brief Keeps the edges whose "in_backbone" attribute is set (filters.py).
/// @param backbone Backbone to filter.
/// @param value Ignored.
/// @param narrate Print the method's description first.
/// @param secondary_property Unused (kept for the Filter signature).
/// @param secondary_property_ascending Unused (kept for the Filter signature).
/// @return The kept edges, or an empty graph (after printing the accepted filters) if the method does not support
///         this filter.
Graph boolean_filter(const Backbone& backbone, double value = 0, bool narrate = true,
                     const std::string& secondary_property = "weight", bool secondary_property_ascending = false);

/// @brief Keeps the edges (or nodes) whose property passes a threshold (filters.py).
///
/// The property must be below the value for ascending properties (p-values) and above it otherwise. Edges come
/// out sorted by the property, ties broken by the secondary property.
/// @param backbone Backbone to filter.
/// @param value Threshold.
/// @param narrate Print the method's description first.
/// @param secondary_property Attribute that orders edges with equal property values.
/// @param secondary_property_ascending Sort the secondary property in ascending order.
/// @return The kept edges or the subgraph on the kept nodes, or an empty graph (after printing the accepted
///         filters) if the method does not support this filter.
Graph threshold_filter(const Backbone& backbone, double value, bool narrate = true,
                       const std::string& secondary_property = "weight", bool secondary_property_ascending = false);

/// @brief Keeps the best fraction of the edges (or nodes) by property (filters.py).
///
/// Sorts by the property (best first, NaN last), ties broken by the secondary property, and keeps the first
/// ceil(value * count).
/// @param backbone Backbone to filter.
/// @param value Fraction to keep, in [0, 1].
/// @param narrate Print the method's description first.
/// @param secondary_property Attribute that orders edges with equal property values.
/// @param secondary_property_ascending Sort the secondary property in ascending order.
/// @return The kept edges or the subgraph on the kept nodes, or an empty graph (after printing the accepted
///         filters) if the method does not support this filter.
Graph fraction_filter(const Backbone& backbone, double value, bool narrate = true,
                      const std::string& secondary_property = "weight", bool secondary_property_ascending = false);

/// @brief Returns a filter's Python name.
/// @param f Filter.
/// @return "boolean_filter", "threshold_filter", "fraction_filter", or "custom_filter" for any other function.
std::string filter_name(Filter f);

/// @brief Result of a backbone method: the scored graph and how to filter it (the Python Backbone class).
struct Backbone {
    Graph graph;                           ///< Input graph annotated with the method's scores.
    std::string method_name;               ///< Human-readable method name, e.g. "Disparity Filter".
    std::string property_name;             ///< Attribute holding the score.
    bool ascending = false;                ///< true when smaller scores are better (p-values).
    std::vector<Filter> filters;           ///< Filters this method supports.
    FilterOn filter_on = FilterOn::Edges;  ///< Whether the score is on edges or nodes.

    /// @brief Writes the scored edges (or nodes) as CSV, like Backbone.to_dataframe().
    /// @param out Destination stream.
    void to_csv(std::ostream& out) const;

    /// @brief Prints the method's description (a placeholder citation for the methods Python does not describe).
    void narrate() const;

    /// @brief Returns the filters this method supports.
    /// @return The compatible filters.
    const std::vector<Filter>& compatible_filters() const { return filters; }
};

}  // namespace cppbone
