/// @file
/// @brief Backbone output: CSV export and narration.

#include "cppbone/backbone.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <string_view>

namespace cppbone {

void Backbone::to_csv(std::ostream &out) const {
    if (filter_on == FilterOn::Edges) {
        graph.to_csv(out);
    } else {
        graph.nodes_to_csv(out);
    }
}

void Backbone::narrate() const {
    using namespace std::string_view_literals;
    static constexpr std::array known = {
        "Disparity Filter"sv,           "Enhanced Configuration Model Filter"sv,
        "Marginal Likelihood Filter"sv, "Locally Adaptive Network Sparsification Filter"sv,
        "Noise Corrected Filter"sv,     "High Salience Skeleton Filter"sv,
        "Modularity Filter"sv,          "Ultrametric Distance Filter"sv,
        "Maximum Spanning Tree"sv,      "Metric Distance Filter"sv,
        "H-Backbone Filter"sv,          "Doubly Stochastic Filter"sv,
        "Global Threshold Filter"sv};
    std::cout << (std::ranges::contains(known, method_name) ? method_name : "Citation here")
              << '\n';
}

} // namespace cppbone
