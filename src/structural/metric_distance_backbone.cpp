/// @file
/// @brief Metric distance backbone (structural/metric_distance_backbone.py), shared with the
/// ultrametric variant.

#include "cppbone/cppbone.hpp"

namespace cppbone {

/// @brief Metric or ultrametric distance backbone, shared by both public functions.
///
/// Sets distance = 1 / weight and marks "in_backbone" on the edges kept by metric_backbone or
/// ultrametric_backbone.
/// @param data Input graph with a "weight" edge attribute.
/// @param metric true for metric (sum) path lengths, false for ultrametric (max).
/// @return Backbone on "distance", filtered with boolean_filter.
Backbone distance_backbone(const Graph &data, bool metric) {
    Graph g = data;
    for (const auto &e : g.edges()) {
        g.set(e.id, "distance", 1 / g.get(e.id, "weight"));
    }
    const Graph b = metric ? metric_backbone(g, "distance") : ultrametric_backbone(g, "distance");
    for (const auto &[u, v, id] : g.edges()) {
        g.set(id, "in_backbone", b.has_edge(u, v)); // same node ids: b is a copy
    }
    return {.graph         = std::move(g),
            .method_name   = metric ? "Metric Distance Filter" : "Ultrametric Distance Filter",
            .property_name = "distance",
            .filters       = {boolean_filter}};
}

Backbone metric_distance_backbone(const Graph &data) { return distance_backbone(data, true); }

} // namespace cppbone
