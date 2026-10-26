/// @file
/// @brief Global threshold backbone (structural/global_threshold.py).

#include "cppbone/cppbone.hpp"

namespace cppbone {

Backbone global_threshold(const Graph &data) {
    return {.graph         = data,
            .method_name   = "Global Threshold Filter",
            .property_name = "weight",
            .filters       = {fraction_filter, threshold_filter}};
}

} // namespace cppbone
