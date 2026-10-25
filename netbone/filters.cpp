#include <algorithm>
#include <cmath>
#include <iostream>

#include "netbone/backbone.hpp"

namespace netbone {

std::string filter_name(Filter f) {
    if (f == boolean_filter) return "boolean_filter";
    if (f == threshold_filter) return "threshold_filter";
    if (f == fraction_filter) return "fraction_filter";
    return "custom_filter";
}

static bool accepts(const Backbone& b, Filter f) {
    const auto& fs = b.compatible_filters();
    if (std::find(fs.begin(), fs.end(), f) != fs.end()) return true;
    std::cout << "The accepted filters for " << b.method_name << " are: ";
    for (size_t i = 0; i < fs.size(); ++i) std::cout << (i ? ", " : "") << filter_name(fs[i]);
    std::cout << '\n';
    return false;
}

// pandas sort_values: NaN goes last whatever the direction.
static int cmp_key(double a, double b, bool ascending) {
    bool na = std::isnan(a), nb = std::isnan(b);
    if (na || nb) return na == nb ? 0 : (na ? 1 : -1);
    if (a == b) return 0;
    return (a < b) == ascending ? -1 : 1;
}

static Graph from_edges(const Graph& g, const std::vector<EdgeRef>& rows) {
    Graph out;
    for (const auto& e : rows) out.add_edge(g.label(e.u), g.label(e.v), g.edge_attrs(e.id));
    return out;
}

static std::vector<EdgeRef> sorted_edges(const Backbone& b, const std::string& secondary, bool secondary_ascending) {
    const Graph& g = b.graph;
    auto rows = g.edges();
    std::stable_sort(rows.begin(), rows.end(), [&](const EdgeRef& x, const EdgeRef& y) {
        int c = cmp_key(g.get(x.id, b.property_name), g.get(y.id, b.property_name), b.ascending);
        if (c == 0) c = cmp_key(g.get(x.id, secondary), g.get(y.id, secondary), secondary_ascending);
        return c < 0;
    });
    return rows;
}

static std::vector<int> sorted_nodes(const Backbone& b) {
    const Graph& g = b.graph;
    std::vector<int> nodes(g.number_of_nodes());
    for (int u = 0; u < g.number_of_nodes(); ++u) nodes[u] = u;
    std::stable_sort(nodes.begin(), nodes.end(), [&](int x, int y) {
        return cmp_key(g.node_get(x, b.property_name), g.node_get(y, b.property_name), b.ascending) < 0;
    });
    return nodes;
}

static bool passes(double x, double value, bool ascending) { return ascending ? x < value : x > value; }

Graph boolean_filter(const Backbone& backbone, double, bool narrate, const std::string&, bool) {
    if (!accepts(backbone, boolean_filter)) return {};
    if (narrate) backbone.narrate();
    std::vector<EdgeRef> rows;
    for (const auto& e : backbone.graph.edges()) {
        double in = backbone.graph.get(e.id, "in_backbone");
        if (!std::isnan(in) && in != 0) rows.push_back(e);
    }
    return from_edges(backbone.graph, rows);
}

Graph threshold_filter(const Backbone& backbone, double value, bool narrate, const std::string& secondary_property,
                       bool secondary_property_ascending) {
    if (!accepts(backbone, threshold_filter)) return {};
    if (narrate) backbone.narrate();
    const Graph& g = backbone.graph;
    if (backbone.filter_on == FilterOn::Edges) {
        std::vector<EdgeRef> rows;
        for (const auto& e : sorted_edges(backbone, secondary_property, secondary_property_ascending))
            if (passes(g.get(e.id, backbone.property_name), value, backbone.ascending)) rows.push_back(e);
        return from_edges(g, rows);
    }
    std::vector<bool> keep(g.number_of_nodes(), false);
    for (int u = 0; u < g.number_of_nodes(); ++u)
        keep[u] = passes(g.node_get(u, backbone.property_name), value, backbone.ascending);
    return g.subgraph(keep);
}

Graph fraction_filter(const Backbone& backbone, double value, bool narrate, const std::string& secondary_property,
                      bool secondary_property_ascending) {
    if (!accepts(backbone, fraction_filter)) return {};
    if (narrate) backbone.narrate();
    const Graph& g = backbone.graph;
    if (backbone.filter_on == FilterOn::Edges) {
        auto rows = sorted_edges(backbone, secondary_property, secondary_property_ascending);
        rows.resize(std::min(rows.size(), (size_t)std::ceil(value * rows.size())));
        return from_edges(g, rows);
    }
    auto nodes = sorted_nodes(backbone);
    size_t n = std::min(nodes.size(), (size_t)std::ceil(value * g.number_of_nodes()));
    std::vector<bool> keep(g.number_of_nodes(), false);
    for (size_t i = 0; i < n; ++i) keep[nodes[i]] = true;
    return g.subgraph(keep);
}

}  // namespace netbone
