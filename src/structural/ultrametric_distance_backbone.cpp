/// @file
/// @brief Ultrametric distance backbone (structural/ultrametric_distance_backbone.py).

#include "cppbone/cppbone.hpp"

namespace cppbone {

/// @brief Metric or ultrametric distance backbone (defined and documented in
/// metric_distance_backbone.cpp).
Backbone distance_backbone(const Graph &data, bool metric);

Backbone ultrametric_distance_backbone(const Graph &data) { return distance_backbone(data, false); }

} // namespace cppbone
