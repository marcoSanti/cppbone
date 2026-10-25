// The README example: one backbone method and the three filters.
#include <iostream>

#include "netbone/netbone.hpp"

int main(int argc, char** argv) {
    namespace nb = netbone;
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " edges.csv   (columns: source,target,weight)\n";
        return 1;
    }
    // load the network
    nb::Graph g = nb::Graph::read_csv(argv[1]);

    // apply the chosen backbone extraction method
    nb::Backbone b = nb::high_salience_skeleton(g);

    // extract the backbone based on the default threshold
    nb::Graph backbone1 = nb::boolean_filter(b);
    // extract the backbone based on a threshold (0.7)
    nb::Graph backbone2 = nb::threshold_filter(b, 0.7);
    // extract the backbone keeping a fraction of edges (0.15)
    nb::Graph backbone3 = nb::fraction_filter(b, 0.15);

    std::cout << "original: " << g.number_of_nodes() << " nodes, " << g.number_of_edges() << " edges\n";
    for (const nb::Graph* x : {&backbone1, &backbone2, &backbone3})
        std::cout << "backbone: " << x->number_of_nodes() << " nodes, " << x->number_of_edges() << " edges\n";
}
