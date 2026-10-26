#include "cppbone/cppbone.hpp"

namespace cppbone {

Backbone global_threshold(const Graph& data) {
    return {data, "Global Threshold Filter", "weight", false, {fraction_filter, threshold_filter}, FilterOn::Edges};
}

}  // namespace cppbone
