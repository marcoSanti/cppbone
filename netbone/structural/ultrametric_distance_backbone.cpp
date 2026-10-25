#include "netbone/netbone.hpp"

namespace netbone {

Backbone distance_backbone(const Graph& data, bool metric);  // metric_distance_backbone.cpp

Backbone ultrametric_distance_backbone(const Graph& data) { return distance_backbone(data, false); }

}  // namespace netbone
