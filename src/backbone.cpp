#include "cppbone/backbone.hpp"

#include <iostream>

namespace cppbone {

void Backbone::to_csv(std::ostream& out) const {
    if (filter_on == FilterOn::Edges)
        graph.to_csv(out);
    else
        graph.nodes_to_csv(out);
}

void Backbone::narrate() const {
    static const char* known[] = {"Disparity Filter",
                                  "Enhanced Configuration Model Filter",
                                  "Marginal Likelihood Filter",
                                  "Locally Adaptive Network Sparsification Filter",
                                  "Noise Corrected Filter",
                                  "High Salience Skeleton Filter",
                                  "Modularity Filter",
                                  "Ultrametric Distance Filter",
                                  "Maximum Spanning Tree",
                                  "Metric Distance Filter",
                                  "H-Backbone Filter",
                                  "Doubly Stochastic Filter",
                                  "Global Threshold Filter"};
    for (const char* k : known)
        if (method_name == k) {
            std::cout << method_name << '\n';
            return;
        }
    std::cout << "Citation here\n";
}

}  // namespace cppbone
