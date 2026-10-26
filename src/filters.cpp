/// @file
/// @brief The boolean, threshold and fraction filters (filters.py).

#include <algorithm>
#include <cmath>
#include <iostream>

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
    if (std::find(fs.begin(), fs.end(), f) != fs.end()) {
        return true;
    }
    std::cout << "The accepted filters for " << b.method_name << " are: ";
    for (size_t i = 0; i < fs.size(); ++i) {
        std::cout << (i ? ", " : "") << filter_name(fs[i]);
    }
    std::cout << '\n';
    return false;
}

/// @brief Three-way comparison of two sort keys like pandas sort_values: NaN goes last whatever the
/// direction.
/// @param a First key.
/// @param b Second key.
/// @param ascending Sort direction.
/// @return -1 if a sorts first, 1 if b does, 0 if they tie.
static int cmp_key(const double a, const double b, const bool ascending) {
    const bool nb = std::isnan(b);
    if (const bool na = std::isnan(a); na || nb) {
        return na == nb ? 0 : (na ? 1 : -1);
    }
    if (a == b) {
        return 0;
    }
    return (a < b) == ascending ? -1 : 1;
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
static std::vector<EdgeRef> sorted_edges(const Backbone &b, const std::string &secondary,
                                         const bool secondary_ascending) {
    const Graph &g  = b.graph;
    auto rows       = g.edges();
    const auto key  = g.edge_values(b.property_name, NaN);
    // keys looked up once, not per comparison
    const auto key2 = g.edge_values(secondary, NaN);
    std::stable_sort(rows.begin(), rows.end(), [&](const EdgeRef &x, const EdgeRef &y) {
        int c = cmp_key(key[x.id], key[y.id], b.ascending);
        if (c == 0) {
            c = cmp_key(key2[x.id], key2[y.id], secondary_ascending);
        }
        return c < 0;
    });
    return rows;
}

/// @brief Lists a node backbone's nodes best first by its property (stable).
/// @param b Backbone with FilterOn::Nodes.
/// @return The sorted node indices.
static std::vector<int> sorted_nodes(const Backbone &b) {
    const Graph &g = b.graph;
    std::vector<int> nodes(g.number_of_nodes());
    for (int u = 0; u < g.number_of_nodes(); ++u) {
        nodes[u] = u;
    }
    std::vector<double> key(g.number_of_nodes());
    for (int u = 0; u < g.number_of_nodes(); ++u) {
        key[u] = g.node_get(u, b.property_name);
    }
    std::stable_sort(nodes.begin(), nodes.end(), [&](const int x, const int y) {
        return cmp_key(key[x], key[y], b.ascending) < 0;
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

Graph boolean_filter(const Backbone &backbone, double, const bool narrate, const std::string &,
                     bool) {
    if (!accepts(backbone, boolean_filter)) {
        return {};
    }
    if (narrate) {
        backbone.narrate();
    }
    std::vector<EdgeRef> rows;
    for (const auto &e : backbone.graph.edges()) {
        if (const double in = backbone.graph.get(e.id, "in_backbone"); !std::isnan(in) && in != 0) {
            rows.push_back(e);
        }
    }
    return from_edges(backbone.graph, rows);
}

Graph threshold_filter(const Backbone &backbone, const double value, const bool narrate,
                       const std::string &secondary_property,
                       const bool secondary_property_ascending) {
    if (!accepts(backbone, threshold_filter)) {
        return {};
    }
    if (narrate) {
        backbone.narrate();
    }
    const Graph &g = backbone.graph;
    if (backbone.filter_on == FilterOn::Edges) {
        std::vector<EdgeRef> rows;
        for (const auto &e :
             sorted_edges(backbone, secondary_property, secondary_property_ascending)) {
            if (passes(g.get(e.id, backbone.property_name), value, backbone.ascending)) {
                rows.push_back(e);
            }
        }
        return from_edges(g, rows);
    }
    std::vector keep(g.number_of_nodes(), false);
    for (int u = 0; u < g.number_of_nodes(); ++u) {
        keep[u] = passes(g.node_get(u, backbone.property_name), value, backbone.ascending);
    }
    return g.subgraph(keep);
}

Graph fraction_filter(const Backbone &backbone, const double value, const bool narrate,
                      const std::string &secondary_property,
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
    for (size_t i = 0; i < n; ++i) {
        keep[nodes[i]] = true;
    }
    return g.subgraph(keep);
}

} // namespace cppbone
