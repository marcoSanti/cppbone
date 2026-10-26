/// @file
/// @brief The boolean, threshold and fraction filters (filters.py).

#include <algorithm>
#include <cmath>
#include <compare>
#include <iostream>
#include <ranges>
#include <utility>

#include "cppbone/backbone.hpp"

namespace cppbone {

std::string filter_name(const Filter f) {
    if (f == boolean_filter) {
        return "boolean_filter";
    }
    if (f == threshold_filter) {
        return "threshold_filter";
    }
    if (f == fraction_filter) {
        return "fraction_filter";
    }
    return "custom_filter";
}

/// @brief Checks that a backbone supports a filter, printing the accepted filters when it does not.
/// @param b Backbone.
/// @param f Filter.
/// @return true if f is among b's compatible filters.
static bool accepts(const Backbone &b, const Filter f) {
    const auto &fs = b.compatible_filters();
    if (std::ranges::contains(fs, f)) {
        return true;
    }
    std::cout << "The accepted filters for " << b.method_name << " are: ";
    const char *separator = "";
    for (const Filter accepted : fs) {
        std::cout << std::exchange(separator, ", ") << filter_name(accepted);
    }
    std::cout << '\n';
    return false;
}

/// @brief Three-way comparison of two sort keys like pandas sort_values: NaN goes last whatever the
/// direction.
/// @param a First key.
/// @param b Second key.
/// @param ascending Sort direction.
/// @return less if a sorts first, greater if b does, equivalent if they tie.
static std::weak_ordering cmp_key(const double a, const double b, const bool ascending) {
    const bool na = std::isnan(a);
    if (const bool nb = std::isnan(b); na || nb) {
        return na <=> nb; // NaN sorts after any number
    }
    if (a == b) {
        return std::weak_ordering::equivalent;
    }
    return (a < b) == ascending ? std::weak_ordering::less : std::weak_ordering::greater;
}

/// @brief Builds a graph from a list of edges of another graph, keeping their attributes and order.
/// @param g Graph the edges belong to.
/// @param rows Edges to copy.
/// @return The new graph.
static Graph from_edges(const Graph &g, const std::vector<EdgeRef> &rows) {
    Graph out;
    for (const auto &[u, v, id] : rows) {
        out.add_edge(g.label(u), g.label(v), g.edge_attrs(id));
    }
    return out;
}

/// @brief Lists a backbone's edges best first: by its property, ties broken by a secondary
/// attribute (stable).
/// @param b Backbone.
/// @param secondary Secondary attribute.
/// @param secondary_ascending Sort the secondary attribute in ascending order.
/// @return The sorted edges.
static std::vector<EdgeRef> sorted_edges(const Backbone &b, const std::string_view secondary,
                                         const bool secondary_ascending) {
    const Graph &g  = b.graph;
    auto rows       = g.edges();
    // keys looked up once, not per comparison
    const auto key  = g.edge_values(b.property_name, NaN);
    const auto key2 = g.edge_values(secondary, NaN);
    std::ranges::stable_sort(
        rows,
        [&](const int x, const int y) {
            const auto primary = cmp_key(key[x], key[y], b.ascending);
            return std::is_lt(std::is_eq(primary) ? cmp_key(key2[x], key2[y], secondary_ascending)
                                                  : primary);
        },
        &EdgeRef::id);
    return rows;
}

/// @brief Lists a node backbone's nodes best first by its property (stable).
/// @param b Backbone with FilterOn::Nodes.
/// @return The sorted node indices.
static std::vector<int> sorted_nodes(const Backbone &b) {
    const Graph &g = b.graph;
    auto nodes     = g.nodes() | std::ranges::to<std::vector>();
    const auto key =
        nodes | std::views::transform([&](const int u) { return g.node_get(u, b.property_name); }) |
        std::ranges::to<std::vector>();
    std::ranges::stable_sort(nodes, [&](const int x, const int y) {
        return std::is_lt(cmp_key(key[x], key[y], b.ascending));
    });
    return nodes;
}

/// @brief Tells whether a score passes a threshold.
/// @param x Score.
/// @param value Threshold.
/// @param ascending true when smaller scores are better.
/// @return x < value for ascending scores, x > value otherwise.
static bool passes(const double x, const double value, const bool ascending) {
    return ascending ? x < value : x > value;
}

Graph boolean_filter(const Backbone &backbone, double, const bool narrate, std::string_view, bool) {
    if (!accepts(backbone, boolean_filter)) {
        return {};
    }
    if (narrate) {
        backbone.narrate();
    }
    const auto in_backbone = [&](const EdgeRef &e) {
        const double in = backbone.graph.get(e.id, "in_backbone");
        return !std::isnan(in) && in != 0;
    };
    return from_edges(backbone.graph, backbone.graph.edges() | std::views::filter(in_backbone) |
                                          std::ranges::to<std::vector>());
}

Graph threshold_filter(const Backbone &backbone, const double value, const bool narrate,
                       const std::string_view secondary_property,
                       const bool secondary_property_ascending) {
    if (!accepts(backbone, threshold_filter)) {
        return {};
    }
    if (narrate) {
        backbone.narrate();
    }
    const Graph &g = backbone.graph;
    if (backbone.filter_on == FilterOn::Edges) {
        const auto pass = [&](const EdgeRef &e) {
            return passes(g.get(e.id, backbone.property_name), value, backbone.ascending);
        };
        return from_edges(g,
                          sorted_edges(backbone, secondary_property, secondary_property_ascending) |
                              std::views::filter(pass) | std::ranges::to<std::vector>());
    }
    return g.subgraph(g.nodes() | std::views::transform([&](const int u) {
                          return passes(g.node_get(u, backbone.property_name), value,
                                        backbone.ascending);
                      }) |
                      std::ranges::to<std::vector<bool>>());
}

Graph fraction_filter(const Backbone &backbone, const double value, const bool narrate,
                      const std::string_view secondary_property,
                      const bool secondary_property_ascending) {
    if (!accepts(backbone, fraction_filter)) {
        return {};
    }
    if (narrate) {
        backbone.narrate();
    }
    const Graph &g = backbone.graph;
    if (backbone.filter_on == FilterOn::Edges) {
        auto rows = sorted_edges(backbone, secondary_property, secondary_property_ascending);
        rows.resize(std::min(rows.size(), static_cast<size_t>(std::ceil(value * rows.size()))));
        return from_edges(g, rows);
    }
    const auto nodes = sorted_nodes(backbone);
    const size_t n =
        std::min(nodes.size(), static_cast<size_t>(std::ceil(value * g.number_of_nodes())));
    std::vector keep(g.number_of_nodes(), false);
    for (const int u : nodes | std::views::take(n)) {
        keep[u] = true;
    }
    return g.subgraph(keep);
}

} // namespace cppbone
