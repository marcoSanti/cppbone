#include "netbone/netbone.hpp"

namespace netbone {

// Shared by the metric and ultrametric variants: distance = 1/weight, in_backbone = edge kept by the closure.
Backbone distance_backbone(const Graph& data, bool metric) {
    Graph g = data;
    for (const auto& e : g.edges()) g.set(e.id, "distance", 1 / g.get(e.id, "weight"));
    Graph b = metric ? metric_backbone(g, "distance") : ultrametric_backbone(g, "distance");
    for (const auto& e : g.edges()) g.set(e.id, "in_backbone", b.has_edge(e.u, e.v));  // same node ids: b is a copy
    return {g, metric ? "Metric Distance Filter" : "Ultrametric Distance Filter", "distance", false, {boolean_filter},
            FilterOn::Edges};
}

Backbone metric_distance_backbone(const Graph& data) { return distance_backbone(data, true); }

}  // namespace netbone
